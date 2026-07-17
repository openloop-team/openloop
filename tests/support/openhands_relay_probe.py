"""Linux-side client used by the opt-in OpenHands relay integration test.

One JSON request is read from stdin and one JSON result is written to stdout.
Credentials therefore never appear in Docker command arguments or query strings.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.metadata
import io
import json
import sys
import threading
import uuid
from pathlib import Path

import httpx

from openloop.tools.openhands_relay import (
    AGENT_SESSION_HEADER,
    LOGICAL_RELAY_HOST,
    RELAY_CAPABILITY_HEADER,
    RelayClientEndpoint,
    RelayMode,
    create_relay_workspace,
    probe_relay_compatibility,
    relay_websocket_callback_client_factory,
)


_OPENHANDS_DISTRIBUTIONS = (
    "openhands-agent-server",
    "openhands-sdk",
    "openhands-tools",
    "openhands-workspace",
)


def _endpoint(request: dict) -> RelayClientEndpoint:
    return RelayClientEndpoint(
        socket_path=Path(request["socket_path"]),
        conversation_id=uuid.UUID(request["conversation_id"]),
        relay_capability=request["relay_capability"],
        session_api_key=request["session_api_key"],
        mode=RelayMode(request["mode"]),
    )


def _raw_client(endpoint: RelayClientEndpoint, request: dict) -> httpx.Client:
    headers = {}
    capability = request.get("request_relay_capability")
    if capability is not None:
        headers[RELAY_CAPABILITY_HEADER] = capability
    session_key = request.get("request_session_api_key")
    if session_key is not None:
        headers[AGENT_SESSION_HEADER] = session_key
    return httpx.Client(
        base_url=LOGICAL_RELAY_HOST,
        transport=httpx.HTTPTransport(uds=str(endpoint.socket_path)),
        headers=headers,
        timeout=10.0,
    )


def _http(endpoint: RelayClientEndpoint, request: dict) -> dict:
    with _raw_client(endpoint, request) as client:
        response = client.request(
            request.get("method", "GET"),
            request.get("path", "/health"),
            json=request.get("json"),
        )
        result: dict[str, object] = {"status_code": response.status_code}
        if request.get("include_json"):
            result["json"] = response.json()
        return result


def _conversation(endpoint: RelayClientEndpoint) -> dict:
    from openhands.sdk import Agent, LLM, Tool
    from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    workspace = create_relay_workspace(endpoint)
    callback_factory = relay_websocket_callback_client_factory(endpoint)
    conversation = None
    try:
        conversation = RemoteConversation(
            agent=Agent(
                llm=LLM(model="openai/gpt-4o-mini", api_key="proof-only"),
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=FileEditorTool.name),
                ],
            ),
            workspace=workspace,
            conversation_id=endpoint.conversation_id,
            visualizer=None,
            delete_on_close=False,
            websocket_client_factory=callback_factory,
        )
        ready = conversation._ws_client is not None and (
            conversation._ws_client.wait_until_ready(timeout=1)
        )
        return {"conversation_id": str(conversation.id), "ready": ready}
    finally:
        if conversation is not None:
            conversation.close()
        workspace.reset_client()


def _probe_marker(endpoint: RelayClientEndpoint, request: dict, name: str) -> Path:
    marker = Path(request[name])
    if marker.parent != endpoint.socket_path.parent or not marker.name.startswith(
        ".probe-"
    ):
        raise ValueError(f"invalid {name}")
    return marker


def _conversation_reconnect(endpoint: RelayClientEndpoint, request: dict) -> dict:
    from openhands.sdk import Agent, LLM, Tool
    from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    workspace = create_relay_workspace(endpoint)
    callback_factory = relay_websocket_callback_client_factory(endpoint)
    ready_path = _probe_marker(endpoint, request, "ready_path")
    reconnected = threading.Event()
    delivered = threading.Event()
    order: list[str] = []

    def observable_factory(*, host, conversation_id, callback, api_key, on_reconnect):
        def observed_reconnect() -> None:
            if on_reconnect is not None:
                on_reconnect()
            order.append("reconcile")
            reconnected.set()

        return callback_factory(
            host=host,
            conversation_id=conversation_id,
            callback=callback,
            api_key=api_key,
            on_reconnect=observed_reconnect,
        )

    def observe_reconnected_event(_event) -> None:
        if reconnected.is_set() and not delivered.is_set():
            order.append("event")
            delivered.set()

    conversation = None
    try:
        conversation = RemoteConversation(
            agent=Agent(
                llm=LLM(model="openai/gpt-4o-mini", api_key="proof-only"),
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=FileEditorTool.name),
                ],
            ),
            workspace=workspace,
            conversation_id=endpoint.conversation_id,
            callbacks=[observe_reconnected_event],
            visualizer=None,
            delete_on_close=False,
            websocket_client_factory=observable_factory,
        )
        ready_path.write_text("ready\n", encoding="ascii")
        if not delivered.wait(timeout=float(request.get("timeout", 60.0))):
            raise TimeoutError("replacement relay subscription did not reconcile")
        return {
            "conversation_id": str(conversation.id),
            "order": order,
            "rest_status_code": workspace.client.get("/health").status_code,
        }
    finally:
        ready_path.unlink(missing_ok=True)
        if conversation is not None:
            conversation.close()
        workspace.reset_client()


def _archive(endpoint: RelayClientEndpoint, request: dict) -> dict:
    workspace = create_relay_workspace(endpoint)
    archive = io.BytesIO()
    try:
        base_commit, written = workspace.stream_git_delta(
            archive, base_ref=request["base_ref"]
        )
    finally:
        workspace.reset_client()
    return {
        "base_commit": base_commit,
        "written": written,
        "archive_base64": base64.b64encode(archive.getvalue()).decode("ascii"),
    }


async def _websocket_status(endpoint: RelayClientEndpoint) -> int:
    import websockets

    try:
        async with websockets.unix_connect(
            path=str(endpoint.socket_path),
            uri=endpoint.websocket_uri,
            additional_headers={RELAY_CAPABILITY_HEADER: endpoint.relay_capability},
            open_timeout=5,
        ):
            return 101
    except websockets.exceptions.InvalidStatus as exc:
        return exc.response.status_code


def main() -> None:
    request = json.load(sys.stdin)
    endpoint = _endpoint(request)
    action = request["action"]
    if action == "http":
        result = _http(endpoint, request)
    elif action == "conversation":
        result = _conversation(endpoint)
    elif action == "conversation_reconnect":
        result = _conversation_reconnect(endpoint, request)
    elif action == "compatibility":
        probe_relay_compatibility()
        result = {"compatible": True}
    elif action == "versions":
        result = {
            "versions": {
                distribution: importlib.metadata.version(distribution)
                for distribution in _OPENHANDS_DISTRIBUTIONS
            }
        }
    elif action == "archive":
        result = _archive(endpoint, request)
    elif action == "websocket_status":
        result = {"status_code": asyncio.run(_websocket_status(endpoint))}
    else:
        raise ValueError(f"unsupported relay probe action: {action!r}")
    json.dump(result, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
