"""Stable public facade for the unwired OpenHands UDS relay profile."""

from __future__ import annotations

import importlib.metadata
import inspect
import os

import httpx

from openloop.openhands.runtime_profile import PINNED_OPENHANDS_VERSION
from openloop.tools.openhands_relay_client import (
    AGENT_SESSION_HEADER,
    LOGICAL_RELAY_HOST,
    RELAY_CAPABILITY_HEADER,
    OpenHandsRelayClientError,
    OpenHandsRelayError,
    RelayClientEndpoint,
    RelayMode,
    create_relay_workspace,
    relay_websocket_callback_client_factory,
)
from openloop.tools.openhands_relay_profile import (
    CONTAINER_RELAY_CAPABILITY_FILE,
    CONTAINER_RELAY_CONFIG_FILE,
    DEFAULT_HAPROXY_RELAY_IMAGE,
    MAX_GENERATION,
    CompiledOpenHandsRelay,
    OpenHandsRelayProfileError,
    RelayRuntimePolicy,
    RelaySecretFile,
    compile_openhands_relay,
    install_relay_artifacts,
)


_REQUIRED_DISTRIBUTIONS = (
    "openhands-sdk",
    "openhands-workspace",
    "openhands-agent-server",
    "openhands-tools",
)


def probe_relay_compatibility() -> None:
    """Fail closed unless the exact forked OpenHands 1.36.0 seam is present."""
    for distribution in _REQUIRED_DISTRIBUTIONS:
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise OpenHandsRelayError(f"{distribution} is not installed") from exc
        if installed != PINNED_OPENHANDS_VERSION:
            raise OpenHandsRelayError(
                f"{distribution} {installed} is incompatible; "
                f"expected {PINNED_OPENHANDS_VERSION}"
            )

    from openhands.sdk.conversation.impl.remote_conversation import (
        RemoteConversation,
        WebSocketCallbackClient,
    )
    from openhands.sdk.workspace import RemoteWorkspace

    workspace_fields = {
        "host",
        "api_key",
        "working_dir",
        "read_timeout",
        "max_connections",
    }
    if not workspace_fields.issubset(RemoteWorkspace.model_fields):
        raise OpenHandsRelayError("OpenHands RemoteWorkspace fields are incompatible")
    if not isinstance(getattr(RemoteWorkspace, "client", None), property):
        raise OpenHandsRelayError("OpenHands RemoteWorkspace client seam changed")

    conversation_parameters = inspect.signature(RemoteConversation.__init__).parameters
    required_conversation_parameters = {
        "agent",
        "workspace",
        "conversation_id",
        "callbacks",
        "delete_on_close",
        "websocket_client_factory",
    }
    if not required_conversation_parameters.issubset(conversation_parameters):
        raise OpenHandsRelayError("OpenHands RemoteConversation seam changed")
    factory_parameter = conversation_parameters["websocket_client_factory"]
    if factory_parameter.default is not None:
        raise OpenHandsRelayError(
            "OpenHands WebSocket client factory default is incompatible"
        )

    readiness_required = os.getenv("OPENHANDS_REMOTE_WS_READY_REQUIRED")
    if readiness_required is not None and readiness_required.lower() in {
        "0",
        "false",
        "no",
    }:
        raise OpenHandsRelayError(
            "OpenHands WebSocket readiness required policy cannot be disabled"
        )
    readiness_timeout = os.getenv("OPENHANDS_REMOTE_WS_READY_TIMEOUT")
    if readiness_timeout is not None:
        try:
            parsed_readiness_timeout = float(readiness_timeout)
        except ValueError as exc:
            raise OpenHandsRelayError(
                "OpenHands WebSocket readiness timeout must remain 30 seconds"
            ) from exc
        if parsed_readiness_timeout != 30.0:
            raise OpenHandsRelayError(
                "OpenHands WebSocket readiness timeout must remain 30 seconds"
            )

    websocket_parameters = tuple(
        inspect.signature(WebSocketCallbackClient.__init__).parameters
    )
    if websocket_parameters != (
        "self",
        "host",
        "conversation_id",
        "callback",
        "api_key",
        "on_reconnect",
    ):
        raise OpenHandsRelayError("OpenHands WebSocket callback signature changed")
    if not inspect.iscoroutinefunction(WebSocketCallbackClient._client_loop):
        raise OpenHandsRelayError("OpenHands WebSocket callback loop changed")
    for method in ("start", "stop", "wait_until_ready"):
        if not callable(getattr(WebSocketCallbackClient, method, None)):
            raise OpenHandsRelayError(
                f"OpenHands WebSocket callback {method} seam changed"
            )

    import websockets

    if not callable(getattr(websockets, "unix_connect", None)):
        raise OpenHandsRelayError("websockets.unix_connect is unavailable")
    if "uds" not in inspect.signature(httpx.HTTPTransport).parameters:
        raise OpenHandsRelayError("httpx HTTPTransport UDS seam changed")


__all__ = [
    "AGENT_SESSION_HEADER",
    "CONTAINER_RELAY_CAPABILITY_FILE",
    "CONTAINER_RELAY_CONFIG_FILE",
    "DEFAULT_HAPROXY_RELAY_IMAGE",
    "LOGICAL_RELAY_HOST",
    "MAX_GENERATION",
    "RELAY_CAPABILITY_HEADER",
    "CompiledOpenHandsRelay",
    "OpenHandsRelayClientError",
    "OpenHandsRelayError",
    "OpenHandsRelayProfileError",
    "RelayClientEndpoint",
    "RelayMode",
    "RelayRuntimePolicy",
    "RelaySecretFile",
    "compile_openhands_relay",
    "create_relay_workspace",
    "install_relay_artifacts",
    "probe_relay_compatibility",
    "relay_websocket_callback_client_factory",
]
