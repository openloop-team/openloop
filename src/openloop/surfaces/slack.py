"""Slack surface.

Handles `app_mention` events: turns a mention into a :class:`Task`, runs it
through the agent runtime, and posts the reply back in-thread. Built on
slack-bolt's async app, exposed to FastAPI via the request handler.
"""

from __future__ import annotations

import logging
import re

from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

from openloop.runtime import Runtime, Task
from openloop.surfaces.approvals import (
    APPROVE_ACTION,
    DENY_ACTION,
    approval_blocks,
    resolve_from_action,
)

logger = logging.getLogger(__name__)

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def _strip_mentions(text: str) -> str:
    return _MENTION_RE.sub("", text or "").strip()


def _approver_handle(user: dict) -> str:
    """Best-effort Slack identity → approver handle (e.g. '@priya')."""
    name = user.get("username") or user.get("name") or user.get("id", "")
    return f"@{name}" if name else "unknown"


async def handle_mention(runtime: Runtime, event: dict, say) -> None:  # type: ignore[no-untyped-def]
    """Core ``app_mention`` logic: mention → Task → runtime → in-thread reply.

    Kept module-level (rather than a closure in :func:`build_slack_app`) so it
    can be driven directly with a synthetic event and a fake ``say`` — the full
    mention path without a live Slack connection.
    """
    text = _strip_mentions(event.get("text", ""))
    thread_ts = event.get("thread_ts") or event.get("ts")
    if not text:
        await say(text="Hi — mention me with a request.", thread_ts=thread_ts)
        return

    task = Task(
        text=text,
        surface="slack",
        channel=event.get("channel"),
        user=event.get("user"),
    )
    try:
        response = await runtime.handle(task)
    except Exception:
        logger.exception("runtime failed handling Slack mention")
        await say(
            text="⚠️ Something went wrong handling that. Check the runtime logs.",
            thread_ts=thread_ts,
        )
        return

    if response.approval_ids and runtime.tools is not None:
        requests = [
            req
            for rid in response.approval_ids
            if (req := await runtime.tools.approvals.get(rid)) is not None
        ]
        await say(
            text=response.text or "Approval required.",
            blocks=approval_blocks(requests),
            thread_ts=thread_ts,
        )
    else:
        await say(text=response.text or "(no response)", thread_ts=thread_ts)


def build_slack_app(
    runtime: Runtime,
    *,
    bot_token: str,
    signing_secret: str | None = None,
) -> AsyncApp:
    """Build the Bolt app (mention + approval handlers) bound to a runtime.

    Shared by both transports: the FastAPI HTTP handler and Socket Mode. With
    no signing secret (Socket-Mode-only), request verification is disabled.
    """
    if signing_secret:
        app = AsyncApp(token=bot_token, signing_secret=signing_secret)
    else:
        app = AsyncApp(token=bot_token, request_verification_enabled=False)

    @app.event("app_mention")
    async def on_mention(event, say):  # type: ignore[no-untyped-def]
        await handle_mention(runtime, event, say)

    async def _on_decision(ack, body, action, respond, approve):  # type: ignore[no-untyped-def]
        await ack()
        if runtime.tools is None:
            return
        approver = _approver_handle(body.get("user", {}))
        message = await resolve_from_action(
            runtime.tools, action["value"], approver, approve=approve
        )
        await respond(text=message, replace_original=False)

    @app.action(APPROVE_ACTION)
    async def on_approve(ack, body, action, respond):  # type: ignore[no-untyped-def]
        await _on_decision(ack, body, action, respond, approve=True)

    @app.action(DENY_ACTION)
    async def on_deny(ack, body, action, respond):  # type: ignore[no-untyped-def]
        await _on_decision(ack, body, action, respond, approve=False)

    return app


def build_slack_handler(
    runtime: Runtime,
    *,
    bot_token: str,
    signing_secret: str,
) -> AsyncSlackRequestHandler:
    """Wrap the Bolt app in a FastAPI request handler (HTTP events transport)."""
    app = build_slack_app(
        runtime, bot_token=bot_token, signing_secret=signing_secret
    )
    return AsyncSlackRequestHandler(app)
