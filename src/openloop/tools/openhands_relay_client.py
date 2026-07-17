"""Native UDS client values for the fixed OpenHands generation relay."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import httpx


LOGICAL_RELAY_HOST = "http://openhands-relay.invalid"
RELAY_CAPABILITY_HEADER = "X-OpenLoop-Relay-Capability"
AGENT_SESSION_HEADER = "X-Session-API-Key"
_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}\Z")
_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_FATAL_WEBSOCKET_CLOSE_CODES = frozenset({4001, 4004})
_FATAL_WEBSOCKET_UPGRADE_STATUSES = frozenset({401, 403, 404})

logger = logging.getLogger(__name__)


def _websocket_close_code(exc: object) -> int | None:
    for close_frame in (getattr(exc, "rcvd", None), getattr(exc, "sent", None)):
        code = getattr(close_frame, "code", None)
        if isinstance(code, int):
            return code
    return None


def _websocket_upgrade_status(exc: object) -> int | None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


class OpenHandsRelayError(RuntimeError):
    """The fixed OpenHands relay boundary is malformed or incompatible."""


class OpenHandsRelayClientError(OpenHandsRelayError):
    """The fixed OpenHands relay client boundary is malformed."""


class RelayMode(str, Enum):
    RUNNING = "running"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True, slots=True, repr=False)
class RelayClientEndpoint:
    """One redacted generation endpoint for HTTP and WebSocket UDS clients."""

    socket_path: Path
    conversation_id: uuid.UUID
    relay_capability: str
    session_api_key: str
    mode: RelayMode
    logical_host: str = LOGICAL_RELAY_HOST

    def __post_init__(self) -> None:
        if not isinstance(self.socket_path, Path) or not self.socket_path.is_absolute():
            raise OpenHandsRelayClientError("relay socket path must be absolute")
        rendered_path = str(self.socket_path)
        if "\0" in rendered_path or self.socket_path.name != "agent.sock":
            raise OpenHandsRelayClientError("relay socket path must end in agent.sock")
        if len(rendered_path.encode("utf-8")) > 100:
            raise OpenHandsRelayClientError(
                "relay socket path exceeds the UDS length budget"
            )
        if not isinstance(self.conversation_id, uuid.UUID):
            raise OpenHandsRelayClientError("relay conversation id must be a UUID")
        if not isinstance(self.relay_capability, str) or not _TOKEN.fullmatch(
            self.relay_capability
        ):
            raise OpenHandsRelayClientError("invalid relay capability")
        if not isinstance(self.session_api_key, str) or not _TOKEN.fullmatch(
            self.session_api_key
        ):
            raise OpenHandsRelayClientError("invalid OpenHands session API key")
        if not isinstance(self.mode, RelayMode):
            raise OpenHandsRelayClientError("invalid relay mode")
        if self.logical_host != LOGICAL_RELAY_HOST:
            raise OpenHandsRelayClientError(
                "relay logical host is fixed by the profile"
            )

    def __repr__(self) -> str:
        return (
            "RelayClientEndpoint("
            f"socket_path={str(self.socket_path)!r}, "
            f"conversation_id={str(self.conversation_id)!r}, "
            "relay_capability=<redacted>, session_api_key=<redacted>, "
            f"mode={self.mode.value!r}, logical_host={self.logical_host!r})"
        )

    @property
    def websocket_uri(self) -> str:
        return f"ws://openhands-relay.invalid/sockets/events/{self.conversation_id}"


def _http_timeout(read_timeout: float) -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=read_timeout, write=10.0, pool=10.0)


@lru_cache(maxsize=1)
def _relay_workspace_class():
    from pydantic import Field, PrivateAttr
    from openhands.sdk.workspace import RemoteWorkspace

    class _RelayRemoteWorkspace(RemoteWorkspace):
        api_key: str | None = Field(default=None, exclude=True, repr=False)
        _relay_endpoint: RelayClientEndpoint = PrivateAttr()

        def __init__(
            self,
            *,
            endpoint: RelayClientEndpoint,
            working_dir: str = "/workspace",
            read_timeout: float = 600.0,
            max_connections: int | None = 16,
        ) -> None:
            super().__init__(
                host=endpoint.logical_host,
                working_dir=working_dir,
                api_key=endpoint.session_api_key,
                read_timeout=read_timeout,
                max_connections=max_connections,
            )
            self._relay_endpoint = endpoint

        @property
        def client(self) -> httpx.Client:
            client = self._client
            if client is None:
                headers = dict(self._headers)
                headers[RELAY_CAPABILITY_HEADER] = self._relay_endpoint.relay_capability
                client = httpx.Client(
                    base_url=self.host,
                    transport=httpx.HTTPTransport(
                        uds=str(self._relay_endpoint.socket_path)
                    ),
                    timeout=_http_timeout(self.read_timeout),
                    headers=headers,
                    limits=httpx.Limits(max_connections=self.max_connections),
                )
                self._client = client
            return client

        def stream_git_delta(
            self,
            sink: BinaryIO,
            *,
            base_ref: str,
        ) -> tuple[str, int]:
            if not base_ref or base_ref.startswith("-"):
                raise OpenHandsRelayClientError("invalid git-delta base ref")
            written = 0
            with self.client.stream(
                "GET",
                "/api/file/archive",
                params={
                    "path": self.working_dir,
                    "format": "git-delta",
                    "base_ref": base_ref,
                },
            ) as response:
                response.raise_for_status()
                base_commit = response.headers.get("X-Archive-Base-Commit", "")
                if not _GIT_OBJECT_ID.fullmatch(base_commit):
                    raise OpenHandsRelayClientError(
                        "OpenHands archive returned an invalid base commit"
                    )
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    result = sink.write(chunk)
                    if result is not None and result != len(chunk):
                        raise OpenHandsRelayClientError(
                            "workspace artifact sink performed a short write"
                        )
                    written += len(chunk)
            return base_commit, written

    _RelayRemoteWorkspace.__name__ = "RelayRemoteWorkspace"
    _RelayRemoteWorkspace.__qualname__ = "RelayRemoteWorkspace"
    return _RelayRemoteWorkspace


def create_relay_workspace(
    endpoint: RelayClientEndpoint,
    *,
    working_dir: str = "/workspace",
    read_timeout: float = 600.0,
    max_connections: int | None = 16,
):
    """Construct the actual pinned RemoteWorkspace subclass lazily."""
    if not isinstance(endpoint, RelayClientEndpoint):
        raise OpenHandsRelayClientError("invalid relay client endpoint")
    return _relay_workspace_class()(
        endpoint=endpoint,
        working_dir=working_dir,
        read_timeout=read_timeout,
        max_connections=max_connections,
    )


def relay_websocket_callback_client_factory(endpoint: RelayClientEndpoint):
    """Return a pinned callback-client factory bound to exactly one endpoint."""
    if not isinstance(endpoint, RelayClientEndpoint):
        raise OpenHandsRelayClientError("invalid relay client endpoint")

    from openhands.sdk.conversation.impl.remote_conversation import (
        WebSocketCallbackClient,
    )

    class RelayWebSocketCallbackClient(WebSocketCallbackClient):
        async def _client_loop(self) -> None:
            import websockets
            from openhands.sdk.event.base import Event
            from openhands.sdk.event.conversation_state import (
                ConversationStateUpdateEvent,
            )

            if self.host != endpoint.logical_host:
                raise OpenHandsRelayClientError(
                    "WebSocket logical host does not match relay"
                )
            if self.conversation_id != str(endpoint.conversation_id):
                raise OpenHandsRelayClientError(
                    "WebSocket conversation id does not match relay"
                )
            if self.api_key != endpoint.session_api_key:
                raise OpenHandsRelayClientError(
                    "WebSocket session key does not match relay"
                )

            delay = 1.0
            has_connected = False
            while not self._stop.is_set():
                try:
                    async with websockets.unix_connect(
                        path=str(endpoint.socket_path),
                        uri=endpoint.websocket_uri,
                        additional_headers={
                            RELAY_CAPABILITY_HEADER: endpoint.relay_capability
                        },
                        open_timeout=10.0,
                        close_timeout=5.0,
                        max_size=4 * 1024 * 1024,
                    ) as websocket:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "auth",
                                    "session_api_key": endpoint.session_api_key,
                                },
                                separators=(",", ":"),
                            )
                        )
                        delay = 1.0
                        connection_ready = False
                        async for message in websocket:
                            if self._stop.is_set():
                                break
                            try:
                                event = Event.model_validate(json.loads(message))
                                if (
                                    isinstance(event, ConversationStateUpdateEvent)
                                    and not connection_ready
                                ):
                                    connection_ready = True
                                    if has_connected:
                                        await self._handle_reconnect()
                                    else:
                                        has_connected = True
                                        self._ready.set()
                                self.callback(event)
                            except Exception:
                                logger.exception(
                                    "relay_ws_event_processing_error",
                                    stack_info=True,
                                )
                except websockets.exceptions.ConnectionClosed as exc:
                    if _websocket_close_code(exc) in _FATAL_WEBSOCKET_CLOSE_CODES:
                        logger.debug(
                            "relay_ws_connection_closed_fatal",
                            exc_info=True,
                        )
                        self._stop.set()
                        break
                    logger.debug("relay_ws_connection_closed_retry", exc_info=True)
                    await self._sleep_before_retry(delay)
                    delay = min(delay * 2, 30.0)
                except websockets.exceptions.InvalidStatus as exc:
                    if (
                        _websocket_upgrade_status(exc)
                        in _FATAL_WEBSOCKET_UPGRADE_STATUSES
                    ):
                        logger.debug(
                            "relay_ws_upgrade_rejected_fatal",
                            exc_info=True,
                        )
                        self._stop.set()
                        break
                    logger.debug("relay_ws_upgrade_rejected_retry", exc_info=True)
                    await self._sleep_before_retry(delay)
                    delay = min(delay * 2, 30.0)
                except Exception:
                    if self._stop.is_set():
                        break
                    logger.debug("relay_ws_connect_retry", exc_info=True)
                    await self._sleep_before_retry(delay)
                    delay = min(delay * 2, 30.0)

    return RelayWebSocketCallbackClient


__all__ = [
    "AGENT_SESSION_HEADER",
    "LOGICAL_RELAY_HOST",
    "RELAY_CAPABILITY_HEADER",
    "OpenHandsRelayError",
    "OpenHandsRelayClientError",
    "RelayClientEndpoint",
    "RelayMode",
    "create_relay_workspace",
    "relay_websocket_callback_client_factory",
]
