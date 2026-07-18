"""Slack surface.

Handles `app_mention` events with Phase D's async-delivery contract: a mention
creates a persisted :class:`~openloop.sessions.store.SurfaceSession`, the handler
sets a transient in-thread progress indicator and returns fast, and the
:class:`~openloop.sessions.runner.SessionRunner` works the turn in the background,
posting the final answer (or an approval card) back to the thread later. Built on
slack-bolt's async app, exposed to FastAPI via the request handler.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

from openloop.analysis.uploads import UploadRecord
from openloop.runtime import Runtime, Task
from openloop.sessions import (
    SessionRunner,
    SlackSurfaceDelivery,
    SurfaceSessionStore,
    SurfaceTarget,
    ThreadRecordStore,
)
from openloop.sessions.threads import thread_scope_key
from openloop.surfaces.approvals import (
    APPROVE_ACTION,
    DENY_ACTION,
    OPENHANDS_ACCEPT_ACTION,
    OPENHANDS_REJECT_ACTION,
)

if TYPE_CHECKING:
    from openloop.analysis import ArtifactStore, UploadStore

logger = logging.getLogger(__name__)

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def _strip_mentions(text: str) -> str:
    return _MENTION_RE.sub("", text or "").strip()


def _strip_bot_mention(text: str, bot_user_id: str | None) -> str:
    """Strip only the bot's own mention, preserving mentions of other users.

    A thread reply like "ask <@U123> about it" is *about* that user, so we must
    keep their mention in the task text — only the bot's own @ is noise.
    """
    if bot_user_id:
        text = (text or "").replace(f"<@{bot_user_id}>", " ")
    return " ".join((text or "").split())


def _approver_handle(user: dict) -> str:
    """Best-effort Slack identity → approver handle (e.g. '@maciag.artur')."""
    name = user.get("username") or user.get("name") or user.get("id", "")
    return f"@{name}" if name else "unknown"


def _target_from_event(runtime: Runtime, event: dict, thread_ts: str | None) -> SurfaceTarget:
    agent = runtime.agent
    return SurfaceTarget(
        surface="slack",
        workspace=agent.metadata.workspace,
        agent=agent.metadata.name,
        channel=event.get("channel"),
        thread=thread_ts,
        # The event ts is the idempotency key — Slack re-delivers the same event
        # on a delivery timeout, and the runner dedupes on it.
        event_id=event.get("event_ts") or event.get("ts"),
    )


async def record_shared_files(runner: SessionRunner, event: dict) -> None:
    """Record any files riding this event as thread-scoped upload metadata.

    Phase 4 lazy staging: metadata only — no bytes are copied at arrival;
    nothing a user posts is retained unless an approved analysis later asks
    for it. The scope is the full thread-ownership tuple key, so an upload is
    provisionable only from the exact thread it was shared in. Best-effort:
    a recording failure must never break mention handling.
    """
    uploads = getattr(runner, "uploads", None)
    if uploads is None or event.get("bot_id"):
        return
    files = event.get("files") or []
    if not files:
        return
    thread_ts = event.get("thread_ts") or event.get("ts")
    scope = thread_scope_key(_target_from_event(runner.runtime, event, thread_ts))
    for file in files:
        file_id = file.get("id") if isinstance(file, dict) else None
        if not file_id:
            continue
        try:
            await uploads.record(
                UploadRecord(
                    upload_ref=file_id,
                    scope_key=scope,
                    name=file.get("name") or file_id,
                    size=int(file.get("size") or 0),
                    user=event.get("user"),
                )
            )
        except Exception:  # noqa: BLE001 — metadata is an inventory, not the turn
            logger.warning("failed to record shared file %s", file_id, exc_info=True)


async def handle_mention(runner: SessionRunner, event: dict, say) -> None:  # type: ignore[no-untyped-def]
    """Core ``app_mention`` logic: mention → session → background runner.

    Kept module-level (rather than a closure in :func:`build_slack_app`) so it
    can be driven directly with a synthetic event and a fake delivery — the full
    mention path without a live Slack connection. The runner owns progress/final
    delivery; only the empty-mention help reply uses ``say`` directly.
    """
    await record_shared_files(runner, event)
    text = _strip_mentions(event.get("text", ""))
    thread_ts = event.get("thread_ts") or event.get("ts")
    if not text:
        await say(text="Hi — mention me with a request.", thread_ts=thread_ts)
        return

    target = _target_from_event(runner.runtime, event, thread_ts)
    task = Task(
        text=text,
        surface="slack",
        channel=event.get("channel"),
        user=event.get("user"),
    )
    await runner.run_threaded(task, target)


async def handle_message(
    runner: SessionRunner, event: dict, say, *, bot_user_id: str | None = None
) -> None:  # type: ignore[no-untyped-def]
    """A non-mention reply in a thread the bot already owns → continue it.

    Slack fires ``message`` for every channel post, so this is deliberately
    narrow: it only acts on a **thread reply** (has ``thread_ts``) whose thread
    already has a session — so the bot continues its own conversations but stays
    silent on unrelated chatter. Bot/self and edited/deleted messages are skipped.

    Only a mention **of the bot** is left to :func:`handle_mention`
    (``app_mention`` fires solely for bot mentions); a reply mentioning some other
    user is a perfectly valid follow-up and is handled here. The reply runs as a
    fresh turn in the same thread; ``run`` dedupes on the message ts.
    """
    # A file share arrives as a `message` event with subtype "file_share" —
    # record its metadata before the subtype filter below skips the message.
    await record_shared_files(runner, event)
    if event.get("bot_id") or event.get("subtype"):
        return  # bot message, or an edit/delete/join subtype — not a user reply
    thread_ts = event.get("thread_ts")
    if not thread_ts:
        return  # only thread replies continue a session
    raw = event.get("text", "")
    if bot_user_id and f"<@{bot_user_id}>" in raw:
        return  # the bot itself was @mentioned — app_mention owns that message
    # Preserve mentions of other users — the reply may be *about* them.
    text = _strip_bot_mention(raw, bot_user_id)
    if not text:
        return

    target = _target_from_event(runner.runtime, event, thread_ts)
    if await runner.sessions.get_by_thread(target) is None:
        return  # the bot isn't part of this thread — stay silent

    task = Task(
        text=text, surface="slack", channel=event.get("channel"),
        user=event.get("user"),
    )
    await runner.run_threaded(task, target)


def build_slack_app(
    runtime: Runtime,
    sessions: SurfaceSessionStore,
    *,
    bot_token: str,
    signing_secret: str | None = None,
    threads: ThreadRecordStore | None = None,
    artifacts: "ArtifactStore | None" = None,
    uploads: "UploadStore | None" = None,
) -> AsyncApp:
    """Build the Bolt app (mention + approval handlers) bound to a runtime.

    Shared by both transports: the FastAPI HTTP handler and Socket Mode. With
    no signing secret (Socket-Mode-only), request verification is disabled. A
    :class:`SessionRunner` over a :class:`SlackSurfaceDelivery` (bound to the
    app's web client) handles the async delivery.
    """
    if signing_secret:
        app = AsyncApp(token=bot_token, signing_secret=signing_secret)
    else:
        app = AsyncApp(token=bot_token, request_verification_enabled=False)

    runner = SessionRunner(
        runtime, sessions, SlackSurfaceDelivery(app.client), threads=threads,
        artifacts=artifacts, uploads=uploads,
    )
    # Exposed so the composition root and approval handler can reach the runner.
    app._session_runner = runner  # type: ignore[attr-defined]

    @app.event("app_mention")
    async def on_mention(event, say):  # type: ignore[no-untyped-def]
        # Return fast: the whole turn runs in the background so Slack's event
        # request isn't held open for the agent's (possibly long) work.
        asyncio.create_task(_run_mention(runner, event, say))

    @app.event("message")
    async def on_message(event, say, context):  # type: ignore[no-untyped-def]
        # Thread replies that continue one of the bot's sessions (handle_message
        # filters out everything else). `context.bot_user_id` lets it tell a
        # mention of the bot (app_mention's job) from one of another user.
        asyncio.create_task(
            _run_message(runner, event, say, context.get("bot_user_id"))
        )

    async def _on_decision(ack, body, action, approve):  # type: ignore[no-untyped-def]
        await ack()
        if runtime.tools is None:
            return
        approver = _approver_handle(body.get("user", {}))
        # The runner resolves the approval *and* continues the owning session:
        # it collapses the approval card in place to the resolution line and posts
        # the eventual answer back in the original thread. `ack()` already satisfies
        # Slack's interaction ack, so we don't also `respond()` — that only added an
        # ephemeral duplicate of the collapsed card.
        await runner.resolve_approval(action["value"], approver, approve=approve)

    @app.action(APPROVE_ACTION)
    async def on_approve(ack, body, action):  # type: ignore[no-untyped-def]
        await _on_decision(ack, body, action, approve=True)

    @app.action(DENY_ACTION)
    async def on_deny(ack, body, action):  # type: ignore[no-untyped-def]
        await _on_decision(ack, body, action, approve=False)

    async def _on_openhands(ack, body, action, kind):  # type: ignore[no-untyped-def]
        await ack()
        try:
            job_id, decision_id = action["value"].split("|", 1)
        except (KeyError, ValueError):
            logger.warning("invalid OpenHands Slack action payload")
            return
        actor = _approver_handle(body.get("user", {})).lstrip("@")
        event_id = (
            action.get("action_ts")
            or (body.get("container") or {}).get("message_ts")
            or body.get("trigger_id")
        )
        if not actor or not event_id:
            logger.warning("incomplete OpenHands Slack decision identity")
            return
        await runner.resolve_openhands_decision(
            job_id,
            decision_id,
            kind=kind,
            actor_id=actor,
            event_id=str(event_id),
        )

    @app.action(OPENHANDS_ACCEPT_ACTION)
    async def on_openhands_accept(ack, body, action):  # type: ignore[no-untyped-def]
        await _on_openhands(ack, body, action, "accept")

    @app.action(OPENHANDS_REJECT_ACTION)
    async def on_openhands_reject(ack, body, action):  # type: ignore[no-untyped-def]
        await _on_openhands(ack, body, action, "reject")

    return app


async def _run_mention(runner: SessionRunner, event: dict, say) -> None:  # type: ignore[no-untyped-def]
    """Background wrapper around :func:`handle_mention` that swallows errors.

    The runner already records + delivers its own failures *once a session
    exists*. This guard covers the earlier handoff steps whose failure would
    otherwise leave the user staring at a mention that silently went nowhere — so
    on any escape it posts a best-effort error notice in-thread.
    """
    try:
        await handle_mention(runner, event, say)
    except Exception:
        logger.exception("Slack mention handling failed for event %s", event.get("ts"))
        thread_ts = event.get("thread_ts") or event.get("ts")
        try:
            await say(
                text="⚠️ Something went wrong starting that. Please try again.",
                thread_ts=thread_ts,
            )
        except Exception:
            logger.exception("failed to post mention-handoff error to the thread")


async def _run_message(
    runner: SessionRunner, event: dict, say, bot_user_id: str | None
) -> None:  # type: ignore[no-untyped-def]
    """Background wrapper around :func:`handle_message`.

    Thread replies are ambient (most are filtered out before any work), so an
    unexpected failure is logged rather than announced in-thread — unlike a direct
    mention, where the user is explicitly waiting on a reply.
    """
    try:
        await handle_message(runner, event, say, bot_user_id=bot_user_id)
    except Exception:
        logger.exception("Slack message handling failed for event %s", event.get("ts"))


def build_slack_handler(
    runtime: Runtime,
    sessions: SurfaceSessionStore,
    *,
    bot_token: str,
    signing_secret: str,
) -> AsyncSlackRequestHandler:
    """Wrap the Bolt app in a FastAPI request handler (HTTP events transport)."""
    app = build_slack_app(
        runtime, sessions, bot_token=bot_token, signing_secret=signing_secret
    )
    return AsyncSlackRequestHandler(app)
