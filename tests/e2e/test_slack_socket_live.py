"""Gated live smoke against REAL Slack — proves the wire, kept off the PR gate.

This is the only test that exercises the actual Slack transport end to end:
Socket Mode connects an outbound WebSocket, we post a mention via the Web API,
Slack delivers an ``app_mention`` event back over the socket, the handler runs
the runtime, and the bot's reply lands in-thread. The deterministic glue
(mention → Task → reply shape) is covered offline in
``tests/integration/test_slack_mention.py``; this verifies the parts only Slack
can: auth, event subscription, and message delivery.

Made non-flaky on purpose:

* **Stubbed model** — a ScriptedGateway returns a fixed reply, so content is
  deterministic and the LLM's nondeterminism stays out of this assertion.
* **Per-run nonce** — the reply text carries a uuid, so we match *our* reply
  and ignore any other traffic in the channel.
* **Thread correlation** — the bot replies under the message we posted, so we
  poll exactly that thread.
* **Bounded poll + honest failure** — wait up to a budget for the nonce reply,
  then fail with diagnostics rather than hanging.
* **Cleanup in finally** — posted messages are deleted so runs don't accumulate.

What it is NOT: hermetic. It needs Slack reachable, so it's gated and skips
cleanly in the normal suite.

The triggering mention must be posted by a *human* user token (``xoxp-…``), not
the bot token: Slack suppresses ``app_mention`` for messages an app posts as
itself (loop prevention), so a bot mentioning itself never fires the event. The
bot token still owns the socket and the reply.

  E2E_LIVE=1
  SLACK_BOT_TOKEN=xoxb-…       (chat:write, app_mentions:read, channels:history)
  SLACK_APP_TOKEN=xapp-…       (Socket Mode enabled on the app)
  E2E_SLACK_USER_TOKEN=xoxp-…  (user-token scope chat:write — posts the mention)
  E2E_SLACK_CHANNEL=C…         (a channel both the bot and that user are in)
"""

from pathlib import Path
import asyncio
import os
import uuid

import pytest

from openloop.agents import load_agent
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime
from openloop.sessions import InMemorySurfaceSessionStore
from openloop.surfaces.slack import build_slack_app
from openloop.testing import in_memory_workflow_engine

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"

REPLY_TIMEOUT_S = 30.0
POLL_INTERVAL_S = 1.0
CONNECT_TIMEOUT_S = 15.0


def _missing() -> str | None:
    if os.environ.get("E2E_LIVE") != "1":
        return "set E2E_LIVE=1 to run the live Slack smoke"
    for var in (
        "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "E2E_SLACK_USER_TOKEN",
        "E2E_SLACK_CHANNEL",
    ):
        if not os.environ.get(var):
            return f"{var} not set"
    return None


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.live,
    pytest.mark.skipif(_missing() is not None, reason=_missing() or ""),
]


class _FixedGateway:
    """Returns one canned reply regardless of input — deterministic content."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def complete(self, model, messages, **kwargs) -> ModelResponse:
        return ModelResponse(text=self._reply, model="scripted")


async def _await_connected(handler) -> None:
    """Block until the Socket Mode WebSocket is live.

    ``connect_async()`` returns before the connection is established, and Socket
    Mode does **not** redeliver events that fired before a connection was live —
    so posting the mention too early loses it forever. Wait for the socket to be
    genuinely connected first.
    """
    deadline = asyncio.get_event_loop().time() + CONNECT_TIMEOUT_S
    while asyncio.get_event_loop().time() < deadline:
        if await handler.client.is_connected():
            # Small settle margin: connected session vs. event-delivery readiness.
            await asyncio.sleep(1.0)
            return
        await asyncio.sleep(0.25)
    raise AssertionError(
        f"Socket Mode did not connect within {CONNECT_TIMEOUT_S}s — check "
        f"SLACK_APP_TOKEN and that Socket Mode is enabled on the app"
    )


async def _wait_for_reply(web, channel: str, parent_ts: str, nonce: str) -> dict:
    """Poll the message's thread for the bot reply carrying our nonce."""
    deadline = asyncio.get_event_loop().time() + REPLY_TIMEOUT_S
    seen: list[str] = []
    while asyncio.get_event_loop().time() < deadline:
        resp = await web.conversations_replies(channel=channel, ts=parent_ts)
        for msg in resp.get("messages", []):
            if msg.get("ts") == parent_ts:
                continue  # the parent mention we posted
            text = msg.get("text", "")
            seen.append(text)
            if nonce in text:
                return msg
        await asyncio.sleep(POLL_INTERVAL_S)
    raise AssertionError(
        f"no reply containing nonce {nonce!r} within {REPLY_TIMEOUT_S}s; "
        f"saw thread messages: {seen!r}"
    )


async def test_live_slack_mention_round_trip():
    from slack_bolt.adapter.socket_mode.async_handler import (
        AsyncSocketModeHandler,
    )
    from slack_sdk.web.async_client import AsyncWebClient

    channel = os.environ["E2E_SLACK_CHANNEL"]
    bot_token = os.environ["SLACK_BOT_TOKEN"]
    app_token = os.environ["SLACK_APP_TOKEN"]
    user_token = os.environ["E2E_SLACK_USER_TOKEN"]

    nonce = uuid.uuid4().hex[:12]
    runtime = Runtime(
        load_agent(AGENT_YAML),
        gateway=_FixedGateway(f"e2e-ack {nonce}"),
        engine=in_memory_workflow_engine(),
    )
    slack_app = build_slack_app(
        runtime, InMemorySurfaceSessionStore(), bot_token=bot_token
    )

    # Diagnostics: separate "event never reached the socket" from "handler ran
    # but failed to reply". Middleware records every inbound request type; the
    # error handler captures anything the listener raises (e.g. a missing scope
    # on `say`).
    received: list[str] = []
    errors: list[str] = []

    @slack_app.use
    async def _record(body, next):  # type: ignore[no-untyped-def]
        received.append((body or {}).get("event", {}).get("type") or (body or {}).get("type", "?"))
        await next()

    @slack_app.error
    async def _on_error(error, body):  # type: ignore[no-untyped-def]
        errors.append(repr(error))

    handler = AsyncSocketModeHandler(slack_app, app_token)

    # Raw tap below Bolt: every WebSocket frame type the socket sees, incl.
    # `hello`. Attached before connect so the opening `hello` is captured.
    raw_types: list[str] = []

    async def _raw(client, message, raw):  # type: ignore[no-untyped-def]
        if isinstance(message, dict):
            raw_types.append(message.get("type", "?"))

    handler.client.message_listeners.append(_raw)

    web = AsyncWebClient(token=bot_token)  # bot: reads replies, identifies bot
    user_web = AsyncWebClient(token=user_token)  # human: posts the mention
    bot_user_id = (await web.auth_test())["user_id"]

    posted_ts: str | None = None
    try:
        await handler.connect_async()  # open the socket, don't block
        await _await_connected(handler)  # ...but don't post until it's live

        # Post as a *user* so Slack fires app_mention — a bot mentioning itself
        # is suppressed. Slack then delivers the event over the bot's socket.
        posted = await user_web.chat_postMessage(
            channel=channel, text=f"<@{bot_user_id}> e2e {nonce}"
        )
        posted_ts = posted["ts"]

        try:
            reply = await _wait_for_reply(web, channel, posted_ts, nonce)
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n  raw socket frame types: {raw_types!r}\n"
                f"  bolt-dispatched envelopes: {received!r}\n"
                f"  handler errors: {errors!r}\n"
                f"  bot_user_id={bot_user_id} channel={channel}"
            ) from None
        assert reply["text"] == f"e2e-ack {nonce}"
        assert reply.get("thread_ts") == posted_ts  # replied in-thread
        assert "app_mention" in received  # the event actually came over the socket
    finally:
        if posted_ts:  # delete as the author (the user that posted it)
            await user_web.chat_delete(channel=channel, ts=posted_ts)
        await handler.close_async()
