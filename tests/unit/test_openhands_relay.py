from __future__ import annotations

import inspect

import pytest

import openloop.tools.openhands_relay as relay
from openloop.tools.openhands_relay import (
    OpenHandsRelayError,
    RelayClientEndpoint,
    compile_openhands_relay,
    create_relay_workspace,
    install_relay_artifacts,
    probe_relay_compatibility,
    relay_websocket_callback_client_factory,
)


def _signatures_with_136_seams(monkeypatch, *, factory_default=None) -> None:
    from openhands.sdk.conversation.impl.remote_conversation import (
        RemoteConversation,
        WebSocketCallbackClient,
    )

    real_signature = relay.inspect.signature
    callback_init = WebSocketCallbackClient.__init__

    def compatible_signature(target):
        signature = real_signature(target)
        parameters = list(signature.parameters.values())
        if target is callback_init:
            if "on_reconnect" not in signature.parameters:
                parameters.append(
                    inspect.Parameter(
                        "on_reconnect",
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
            return signature.replace(parameters=parameters)
        if target is not RemoteConversation.__init__:
            return signature
        existing = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter.name == "websocket_client_factory"
            ),
            None,
        )
        if existing is not None:
            parameters[existing] = parameters[existing].replace(default=factory_default)
            return signature.replace(parameters=parameters)
        insertion = next(
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        )
        parameters.insert(
            insertion,
            inspect.Parameter(
                "websocket_client_factory",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=factory_default,
            ),
        )
        return signature.replace(parameters=parameters)

    monkeypatch.setattr(relay.inspect, "signature", compatible_signature)


def _pinned_versions(monkeypatch) -> None:
    monkeypatch.setattr(
        relay.importlib.metadata,
        "version",
        lambda _distribution: "1.36.0",
    )


def test_facade_exports_only_production_profile_entry_points() -> None:
    assert RelayClientEndpoint is relay.RelayClientEndpoint
    assert compile_openhands_relay is relay.compile_openhands_relay
    assert create_relay_workspace is relay.create_relay_workspace
    assert install_relay_artifacts is relay.install_relay_artifacts
    assert (
        relay_websocket_callback_client_factory
        is relay.relay_websocket_callback_client_factory
    )
    assert "render_haproxy_config" not in relay.__all__
    assert "write_haproxy_config" not in relay.__all__


def test_probe_passes_for_pinned_environment_with_factory_seam(
    monkeypatch,
) -> None:
    _pinned_versions(monkeypatch)
    _signatures_with_136_seams(monkeypatch)
    probe_relay_compatibility()


def test_probe_rejects_unpatched_sdk_factory_seam(monkeypatch) -> None:
    from openhands.sdk.conversation.impl.remote_conversation import (
        RemoteConversation,
    )

    if (
        "websocket_client_factory"
        in inspect.signature(RemoteConversation.__init__).parameters
    ):
        pytest.skip("the pinned OpenLoop SDK fork is already installed")
    _pinned_versions(monkeypatch)
    with pytest.raises(OpenHandsRelayError, match="RemoteConversation seam"):
        probe_relay_compatibility()


def test_probe_rejects_wrong_distribution_version(monkeypatch) -> None:
    def fake_version(distribution: str) -> str:
        if distribution == "openhands-sdk":
            return "1.37.0"
        return "1.36.0"

    monkeypatch.setattr(relay.importlib.metadata, "version", fake_version)
    with pytest.raises(OpenHandsRelayError, match="1.37.0 is incompatible"):
        probe_relay_compatibility()


def test_probe_rejects_missing_distribution(monkeypatch) -> None:
    def fake_version(distribution: str) -> str:
        if distribution == "openhands-tools":
            raise relay.importlib.metadata.PackageNotFoundError(distribution)
        return "1.36.0"

    monkeypatch.setattr(relay.importlib.metadata, "version", fake_version)
    with pytest.raises(OpenHandsRelayError, match="openhands-tools is not installed"):
        probe_relay_compatibility()


def test_probe_rejects_changed_workspace_client(monkeypatch) -> None:
    from openhands.sdk.workspace import RemoteWorkspace

    _pinned_versions(monkeypatch)
    _signatures_with_136_seams(monkeypatch)
    monkeypatch.setattr(RemoteWorkspace, "client", None)
    with pytest.raises(OpenHandsRelayError, match="client seam"):
        probe_relay_compatibility()


def test_probe_rejects_non_none_factory_default(monkeypatch) -> None:
    _pinned_versions(monkeypatch)
    _signatures_with_136_seams(monkeypatch, factory_default=object())
    with pytest.raises(OpenHandsRelayError, match="factory default"):
        probe_relay_compatibility()


def test_probe_rejects_changed_websocket_constructor(monkeypatch) -> None:
    from openhands.sdk.conversation.impl.remote_conversation import (
        WebSocketCallbackClient,
    )

    _pinned_versions(monkeypatch)
    _signatures_with_136_seams(monkeypatch)

    def changed_init(self, endpoint):
        del self, endpoint

    monkeypatch.setattr(WebSocketCallbackClient, "__init__", changed_init)
    with pytest.raises(OpenHandsRelayError, match="callback signature"):
        probe_relay_compatibility()


def test_probe_rejects_missing_unix_connect(monkeypatch) -> None:
    import websockets

    _pinned_versions(monkeypatch)
    _signatures_with_136_seams(monkeypatch)
    monkeypatch.setattr(websockets, "unix_connect", None)
    with pytest.raises(OpenHandsRelayError, match="unix_connect"):
        probe_relay_compatibility()


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("OPENHANDS_REMOTE_WS_READY_REQUIRED", "false", "readiness required"),
        ("OPENHANDS_REMOTE_WS_READY_TIMEOUT", "10", "readiness timeout"),
    ],
)
def test_probe_rejects_weakened_websocket_readiness_environment(
    monkeypatch,
    name: str,
    value: str,
    match: str,
) -> None:
    _pinned_versions(monkeypatch)
    _signatures_with_136_seams(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(OpenHandsRelayError, match=match):
        probe_relay_compatibility()
