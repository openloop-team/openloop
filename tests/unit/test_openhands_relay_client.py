from __future__ import annotations

import asyncio
import inspect
import io
import json
import uuid
from pathlib import Path

import httpx
import pytest

from openloop.tools.openhands_relay_client import (
    AGENT_SESSION_HEADER,
    LOGICAL_RELAY_HOST,
    RELAY_CAPABILITY_HEADER,
    OpenHandsRelayClientError,
    RelayClientEndpoint,
    RelayMode,
    create_relay_workspace,
    relay_websocket_callback_client_factory,
)
from openloop.tools.openhands_relay_profile import compile_openhands_relay


JOB_ID = uuid.UUID("fc04973b-dc6b-4472-8903-e0981fbbd38e")
CONVERSATION_ID = uuid.UUID("9a1db585-06ba-47cd-952d-cd60c2d0d5d1")
CAPABILITY = "r" * 43
SESSION_KEY = "s" * 43


def _endpoint() -> RelayClientEndpoint:
    return compile_openhands_relay(
        job_id=JOB_ID,
        generation=7,
        conversation_id=CONVERSATION_ID,
        relay_capability=CAPABILITY,
        session_api_key=SESSION_KEY,
        mode=RelayMode.RUNNING,
    ).endpoint


def test_endpoint_is_redacted_and_has_fixed_logical_hosts() -> None:
    endpoint = _endpoint()
    rendered = repr(endpoint)

    assert CAPABILITY not in rendered
    assert SESSION_KEY not in rendered
    assert rendered.count("<redacted>") == 2
    assert endpoint.logical_host == LOGICAL_RELAY_HOST
    assert endpoint.websocket_uri == (
        f"ws://openhands-relay.invalid/sockets/events/{CONVERSATION_ID}"
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"socket_path": Path("relative/agent.sock")}, "absolute"),
        ({"socket_path": Path("/tmp/not-the-socket")}, "agent.sock"),
        ({"socket_path": Path("/" + "x" * 101 + "/agent.sock")}, "length"),
        ({"conversation_id": str(CONVERSATION_ID)}, "UUID"),
        ({"relay_capability": "short"}, "capability"),
        ({"session_api_key": "short"}, "session API key"),
        ({"mode": "running"}, "mode"),
        ({"logical_host": "http://caller.example"}, "logical host"),
    ],
)
def test_endpoint_rejects_malformed_or_selected_values(overrides, match) -> None:
    values = {
        "socket_path": Path(f"/run/openloop/jobs/{JOB_ID}/7/agent.sock"),
        "conversation_id": CONVERSATION_ID,
        "relay_capability": CAPABILITY,
        "session_api_key": SESSION_KEY,
        "mode": RelayMode.RUNNING,
    }
    values.update(overrides)
    with pytest.raises(OpenHandsRelayClientError, match=match):
        RelayClientEndpoint(**values)


def test_workspace_uses_uds_and_both_http_credentials(monkeypatch) -> None:
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["request"] = request
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)

    def fake_transport(**kwargs):
        observed["transport_kwargs"] = kwargs
        return transport

    monkeypatch.setattr(httpx, "HTTPTransport", fake_transport)
    workspace = create_relay_workspace(_endpoint())
    try:
        response = workspace.client.get("/health")
        assert response.status_code == 200
    finally:
        workspace.reset_client()

    assert observed["transport_kwargs"] == {
        "uds": f"/run/openloop/jobs/{JOB_ID}/7/agent.sock"
    }
    request = observed["request"]
    assert request.url == f"{LOGICAL_RELAY_HOST}/health"
    assert request.headers[RELAY_CAPABILITY_HEADER] == CAPABILITY
    assert request.headers[AGENT_SESSION_HEADER] == SESSION_KEY


class _Chunks(httpx.SyncByteStream):
    def __iter__(self):
        yield b"first"
        yield b"-second"


def _archive_workspace(response_headers: dict[str, str] | None = None):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers=response_headers or {"X-Archive-Base-Commit": "a" * 40},
            stream=_Chunks(),
        )

    workspace = create_relay_workspace(_endpoint())
    workspace._client = httpx.Client(
        base_url=LOGICAL_RELAY_HOST,
        transport=httpx.MockTransport(handler),
    )
    return workspace, requests


def test_archive_streams_chunks_and_validates_base_commit() -> None:
    workspace, requests = _archive_workspace()
    sink = io.BytesIO()
    try:
        base_commit, written = workspace.stream_git_delta(sink, base_ref="a" * 40)
    finally:
        workspace.reset_client()

    assert base_commit == "a" * 40
    assert written == len(b"first-second")
    assert sink.getvalue() == b"first-second"
    assert requests[0].url.params["path"] == "/workspace"
    assert requests[0].url.params["format"] == "git-delta"
    assert requests[0].url.params["base_ref"] == "a" * 40


def test_archive_rejects_invalid_base_commit_header() -> None:
    workspace, _ = _archive_workspace({"X-Archive-Base-Commit": "invalid"})
    try:
        with pytest.raises(OpenHandsRelayClientError, match="invalid base commit"):
            workspace.stream_git_delta(io.BytesIO(), base_ref="main")
    finally:
        workspace.reset_client()


class _ShortSink:
    def write(self, chunk: bytes) -> int:
        return len(chunk) - 1


def test_archive_rejects_short_sink_and_option_shaped_ref() -> None:
    workspace, _ = _archive_workspace()
    try:
        with pytest.raises(OpenHandsRelayClientError, match="short write"):
            workspace.stream_git_delta(_ShortSink(), base_ref="main")
        with pytest.raises(OpenHandsRelayClientError, match="base ref"):
            workspace.stream_git_delta(io.BytesIO(), base_ref="--output=/secret")
    finally:
        workspace.reset_client()


class _FakeWebSocket:
    def __init__(self, callback_client) -> None:
        self.callback_client = callback_client
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        self.callback_client._stop.set()

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def test_websocket_factory_uses_uds_header_and_first_message_auth(
    monkeypatch,
) -> None:
    endpoint = _endpoint()
    factory = relay_websocket_callback_client_factory(endpoint)
    callback_client = factory(
        host=LOGICAL_RELAY_HOST,
        conversation_id=str(CONVERSATION_ID),
        callback=lambda _event: None,
        api_key=SESSION_KEY,
        on_reconnect=None,
    )
    observed = {}

    def fake_unix_connect(**kwargs):
        observed["kwargs"] = kwargs
        socket = _FakeWebSocket(callback_client)
        observed["socket"] = socket
        return socket

    import websockets

    monkeypatch.setattr(websockets, "unix_connect", fake_unix_connect)
    asyncio.run(callback_client._client_loop())

    kwargs = observed["kwargs"]
    assert kwargs["path"] == str(endpoint.socket_path)
    assert kwargs["uri"] == endpoint.websocket_uri
    assert SESSION_KEY not in kwargs["uri"]
    assert kwargs["additional_headers"] == {RELAY_CAPABILITY_HEADER: CAPABILITY}
    assert json.loads(observed["socket"].sent[0]) == {
        "type": "auth",
        "session_api_key": SESSION_KEY,
    }


def test_websocket_factory_matches_sdk_keyword_contract_and_mutates_no_global() -> None:
    from openhands.sdk.conversation.impl import remote_conversation

    original = remote_conversation.WebSocketCallbackClient
    factory = relay_websocket_callback_client_factory(_endpoint())

    assert tuple(inspect.signature(factory).parameters) == (
        "host",
        "conversation_id",
        "callback",
        "api_key",
        "on_reconnect",
    )
    assert remote_conversation.WebSocketCallbackClient is original


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"host": "http://wrong.invalid"}, "logical host"),
        ({"conversation_id": str(uuid.uuid4())}, "conversation id"),
        ({"api_key": "x" * 43}, "session key"),
    ],
)
def test_websocket_refuses_endpoint_mismatch(overrides, match) -> None:
    values = {
        "host": LOGICAL_RELAY_HOST,
        "conversation_id": str(CONVERSATION_ID),
        "callback": lambda _event: None,
        "api_key": SESSION_KEY,
        "on_reconnect": None,
    }
    values.update(overrides)
    callback_client = relay_websocket_callback_client_factory(_endpoint())(**values)
    with pytest.raises(OpenHandsRelayClientError, match=match):
        asyncio.run(callback_client._client_loop())


def test_websocket_retries_initial_connection_with_bounded_backoff(
    monkeypatch,
) -> None:
    endpoint = _endpoint()
    callback_client = relay_websocket_callback_client_factory(endpoint)(
        host=LOGICAL_RELAY_HOST,
        conversation_id=str(CONVERSATION_ID),
        callback=lambda _event: None,
        api_key=SESSION_KEY,
        on_reconnect=None,
    )
    attempts = 0
    delays = []

    def flaky_unix_connect(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("not ready")
        return _FakeWebSocket(callback_client)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    import websockets

    monkeypatch.setattr(websockets, "unix_connect", flaky_unix_connect)
    monkeypatch.setattr(callback_client, "_sleep_before_retry", fake_sleep)
    asyncio.run(callback_client._client_loop())

    assert attempts == 2
    assert delays == [1.0]


def _state_update_payload(event_id: str) -> str:
    return json.dumps(
        {
            "kind": "ConversationStateUpdateEvent",
            "id": event_id,
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "environment",
            "key": "execution_status",
            "value": "running",
        }
    )


def _connection_closed(code: int):
    import websockets
    import websockets.frames

    return websockets.exceptions.ConnectionClosed(
        rcvd=websockets.frames.Close(code, "test"),
        sent=websockets.frames.Close(code, "test"),
        rcvd_then_sent=False,
    )


class _ReconnectWebSocket:
    def __init__(self, messages: list[str], close_code: int | None = None) -> None:
        self.messages = list(messages)
        self.close_code = close_code
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        if self.close_code is not None:
            code = self.close_code
            self.close_code = None
            raise _connection_closed(code)
        raise StopAsyncIteration


def test_websocket_reconnects_reauthenticates_and_reconciles_in_order(
    monkeypatch,
) -> None:
    endpoint = _endpoint()
    order: list[str] = []
    callback_client = None

    def callback(event) -> None:
        order.append(event.id)
        if event.id == "state-2":
            callback_client._stop.set()

    def reconcile() -> None:
        order.append("reconcile")

    callback_client = relay_websocket_callback_client_factory(endpoint)(
        host=LOGICAL_RELAY_HOST,
        conversation_id=str(CONVERSATION_ID),
        callback=callback,
        api_key=SESSION_KEY,
        on_reconnect=reconcile,
    )
    connections = [
        _ReconnectWebSocket([_state_update_payload("state-1")], close_code=1000),
        _ReconnectWebSocket([_state_update_payload("state-2")]),
    ]
    pending_connections = list(connections)
    delays: list[float] = []

    def fake_unix_connect(**_kwargs):
        return pending_connections.pop(0)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    import websockets

    monkeypatch.setattr(websockets, "unix_connect", fake_unix_connect)
    monkeypatch.setattr(callback_client, "_sleep_before_retry", fake_sleep)
    asyncio.run(callback_client._client_loop())

    assert order == ["state-1", "reconcile", "state-2"]
    assert delays == [1.0]
    assert callback_client._ready.is_set()
    assert len(pending_connections) == 0
    expected_auth = {"type": "auth", "session_api_key": SESSION_KEY}
    for socket in connections:
        assert json.loads(socket.sent[0]) == expected_auth


@pytest.mark.parametrize("close_code", [4001, 4004])
def test_websocket_stops_after_fatal_close(monkeypatch, close_code: int) -> None:
    callback_client = relay_websocket_callback_client_factory(_endpoint())(
        host=LOGICAL_RELAY_HOST,
        conversation_id=str(CONVERSATION_ID),
        callback=lambda _event: None,
        api_key=SESSION_KEY,
        on_reconnect=None,
    )
    attempts = 0

    def fatal_connect(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise _connection_closed(close_code)

    import websockets

    monkeypatch.setattr(websockets, "unix_connect", fatal_connect)
    asyncio.run(callback_client._client_loop())

    assert attempts == 1
    assert callback_client._stop.is_set()


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_websocket_stops_after_fatal_upgrade_status(
    monkeypatch,
    status_code: int,
) -> None:
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    callback_client = relay_websocket_callback_client_factory(_endpoint())(
        host=LOGICAL_RELAY_HOST,
        conversation_id=str(CONVERSATION_ID),
        callback=lambda _event: None,
        api_key=SESSION_KEY,
        on_reconnect=None,
    )
    attempts = 0

    def fatal_connect(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise InvalidStatus(Response(status_code, "denied", Headers(), body=b""))

    async def unexpected_retry(_delay: float) -> None:
        raise AssertionError("fatal upgrade response was retried")

    import websockets

    monkeypatch.setattr(websockets, "unix_connect", fatal_connect)
    monkeypatch.setattr(callback_client, "_sleep_before_retry", unexpected_retry)
    asyncio.run(callback_client._client_loop())

    assert attempts == 1
    assert callback_client._stop.is_set()


def test_websocket_stop_interrupts_capped_backoff() -> None:
    callback_client = relay_websocket_callback_client_factory(_endpoint())(
        host=LOGICAL_RELAY_HOST,
        conversation_id=str(CONVERSATION_ID),
        callback=lambda _event: None,
        api_key=SESSION_KEY,
        on_reconnect=None,
    )

    async def stop_during_backoff() -> None:
        task = asyncio.create_task(callback_client._sleep_before_retry(30.0))
        await asyncio.sleep(0)
        callback_client._stop.set()
        await asyncio.wait_for(task, timeout=0.5)

    asyncio.run(stop_during_backoff())
