"""Surface delivery — posting status, approvals, and final answers to a surface.

Phase D decouples the *answer* from the inbound request lifecycle. A
:class:`SurfaceDelivery` is the surface-agnostic seam the session runner uses to
set transient progress status, post approval cards when human input is needed,
and post the final answer (or an error) later — possibly long after the original
HTTP/Bolt request returned.

Producers hand this seam a surface-neutral :class:`~openloop.deliverable.Deliverable`
(plain strings are coerced to :class:`~openloop.deliverable.Prose`); each
implementation is the *renderer* that maps it onto what the surface does best.
Slack renders prose via ``markdown_text`` (server-side standard-Markdown
rendering), artifacts as hosted snippets with a summary message, and approval
cards as Block Kit — producers never see any of those dialects.

Delivery must be **idempotent**: the runner persists the ids returned here on the
session, so a crash-and-resume or a duplicate inbound event reuses the existing
approval/final message instead of posting a second one. The protocol returns the
provider message id from each durable post; ``update_approval`` takes one back.

That persisted id is the primary guard, but it leaves one window open: between a
provider accepting a post and the runner recording the returned id, a crash means
the id is lost and a retry can't tell the post already landed — at-least-once. To
close it, each post carries a deterministic ``key``: the post is *tagged* with the
key so a later attempt can find it, and when ``recover`` is set the implementation
first looks for an already-posted message with that key and returns its id instead
of posting a duplicate. Tagging is free (no extra call); the lookup runs only on
the recovery path, so the happy path is unaffected. Surfaces with no native dedup
(Slack) realize this best-effort and degrade to at-least-once if the lookup can't
run — the persisted id + startup reconciler remain the primary mechanism.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from openloop.deliverable import Artifact, Deliverable, Prose, coerce
from openloop.sessions.store import SurfaceTarget
from openloop.surfaces import slack_format

if TYPE_CHECKING:
    from openloop.approvals.store import ApprovalRequest

logger = logging.getLogger(__name__)


@runtime_checkable
class SurfaceDelivery(Protocol):
    async def set_progress_status(self, target: SurfaceTarget, text: str) -> None:
        """Set a transient, surface-native progress indicator.

        This is best-effort UI polish: implementations should not let missing
        provider support or transient provider errors block the actual answer.
        """
        ...

    async def update_approval(
        self,
        target: SurfaceTarget,
        message_id: str,
        text: str,
        requests: "list[ApprovalRequest]",
    ) -> None:
        """Update an existing approval card with buttons or resolution text.

        Editing the approval card in place keeps button handling tidy; the final
        answer is delivered separately so a cosmetic update failure cannot lose
        the outcome.
        """
        ...

    async def post_approval(
        self,
        target: SurfaceTarget,
        text: str,
        requests: "list[ApprovalRequest]",
        *,
        key: str | None = None,
        recover: bool = False,
    ) -> str:
        """Post a new approval card with buttons; return its provider id.

        ``key`` tags the message for idempotent recovery; ``recover`` asks the
        implementation to first return an already-posted message with this key
        (closing the post-succeeded-but-id-lost window) rather than duplicating.
        """
        ...

    async def post_openhands_decision(
        self,
        target: SurfaceTarget,
        job_id: str,
        decision_id: str,
        summary: str,
        *,
        key: str | None = None,
        recover: bool = False,
    ) -> str: ...

    async def update_openhands_decision(
        self,
        target: SurfaceTarget,
        message_id: str,
        job_id: str,
        decision_id: str,
        summary: str,
    ) -> None: ...

    async def post_final(
        self, target: SurfaceTarget, result: "Deliverable | str", *,
        key: str | None = None, recover: bool = False,
    ) -> str:
        """Render and post the final answer; return its provider id.

        ``result`` is surface-neutral (a plain string is treated as Markdown
        prose); the implementation owns the rendering. See :meth:`post_approval`
        for ``key`` / ``recover``.
        """
        ...

    async def post_error(
        self, target: SurfaceTarget, text: str, *, key: str | None = None,
        recover: bool = False,
    ) -> str:
        """Post an error/interrupted notice; return its provider id.

        See :meth:`post_approval` for ``key`` / ``recover``.
        """
        ...


# Slack has no native idempotency key on chat.postMessage, so we tag each posted
# message with the delivery key in Slack `metadata` and, on recovery, scan the
# thread for a message already bearing it. Bounded scan: only the most recent page
# is checked — enough for the crash-retry window (our message is among the latest),
# best-effort for very long threads. Needs the bot's `*:history` read scope; if it
# lacks it (or the call fails) the lookup degrades to None and we post fresh.
_DELIVERY_EVENT_TYPE = "openloop_delivery"
_LOOKUP_LIMIT = 200


class SlackSurfaceDelivery:
    """The Slack renderer behind :class:`SurfaceDelivery`, via an ``AsyncWebClient``.

    Rendering policy (text transforms live in :mod:`openloop.surfaces.slack_format`):

    - :class:`Prose` → ``chat.postMessage(markdown_text=...)`` — Slack converts
      the agent's standard Markdown server-side (plain ``text`` would render
      ``**bold**`` literally). Broadcast pings (``<!channel>`` etc.) in
      model-authored text are neutralized first; user mentions are preserved.
    - :class:`Artifact` (and prose past the ``markdown_text`` cap) → a hosted
      snippet via ``files.upload`` v2, shared into the thread at upload time,
      plus a short keyed summary message (prose only, no permalink, unfurling
      off) that captions the share and anchors delivery. Needs the
      ``files:write`` scope; on upload failure the content degrades to an
      inline plain-text post rather than dropping the answer.
    - Approval cards → Block Kit, with ``text`` as the notification fallback.

    Uses the stored channel + thread and message timestamps Slack returns so the
    runner can dedupe and update. Threading is best-effort: if a target has no
    thread it posts at the channel root.
    """

    def __init__(self, client) -> None:  # AsyncWebClient
        self.client = client

    @staticmethod
    def _metadata(key: str | None) -> dict | None:
        """Slack message metadata that tags a post with its delivery key."""
        if not key:
            return None
        return {"event_type": _DELIVERY_EVENT_TYPE, "event_payload": {"key": key}}

    async def _find_by_key(self, target: SurfaceTarget, key: str) -> str | None:
        """Return the ts of an already-posted message tagged with ``key``, if any.

        Scans the thread (or channel root) for a message carrying our delivery
        metadata. Defensive: any failure (missing history scope, transient error)
        degrades to ``None`` so the caller posts fresh rather than crashing.
        """
        try:
            if target.thread:
                resp = await self.client.conversations_replies(
                    channel=target.channel,
                    ts=target.thread,
                    include_all_metadata=True,
                    limit=_LOOKUP_LIMIT,
                )
            else:
                resp = await self.client.conversations_history(
                    channel=target.channel,
                    include_all_metadata=True, limit=_LOOKUP_LIMIT,
                )
        except Exception:  # noqa: BLE001 — best-effort dedup; fall back to posting
            logger.warning(
                "delivery idempotency lookup failed for key %s; posting fresh",
                key, exc_info=True,
            )
            return None
        for msg in resp.get("messages", []) or []:
            md = msg.get("metadata") or {}
            payload = md.get("event_payload") or {}
            if (
                md.get("event_type") == _DELIVERY_EVENT_TYPE
                and payload.get("key") == key
            ):
                return msg.get("ts")
        return None

    async def set_progress_status(self, target: SurfaceTarget, text: str) -> None:
        """Set Slack's assistant-thread status, e.g. "<App> is thinking..."."""
        if not target.channel or not target.thread:
            return
        try:
            setter = getattr(self.client, "assistant_threads_setStatus", None)
            if setter is not None:
                await setter(
                    channel_id=target.channel,
                    thread_ts=target.thread,
                    status=text,
                    loading_messages=[text],
                )
            else:
                await self.client.api_call(
                    "assistant.threads.setStatus",
                    json={
                        "channel_id": target.channel,
                        "thread_ts": target.thread,
                        "status": text,
                        "loading_messages": [text],
                    },
                )
        except Exception:  # noqa: BLE001 — status must never block delivery
            logger.warning(
                "failed to set Slack assistant status for thread %s",
                target.thread,
                exc_info=True,
            )

    async def _post(
        self,
        target: SurfaceTarget,
        *,
        text: str | None = None,
        markdown_text: str | None = None,
        blocks: list[dict] | None = None,
        key: str | None,
        recover: bool,
        unfurl_links: bool | None = None,
        unfurl_media: bool | None = None,
    ) -> str:
        """Tagged, idempotent message post: recover an existing keyed message or
        post one. The SDK drops ``None`` params, so unused fields never reach
        the API (``markdown_text`` is mutually exclusive with ``text``/``blocks``).

        ``unfurl_links``/``unfurl_media`` default to ``None`` (Slack's normal
        behavior); the artifact summary sets them ``False`` so the file permalink
        it carries stays a plain locator link instead of unfurling into a second
        preview of the file already shared into the thread.
        """
        if key and recover:
            existing = await self._find_by_key(target, key)
            if existing is not None:
                return existing
        resp = await self.client.chat_postMessage(
            channel=target.channel,
            thread_ts=target.thread,
            text=text,
            markdown_text=markdown_text,
            blocks=blocks,
            metadata=self._metadata(key),
            unfurl_links=unfurl_links,
            unfurl_media=unfurl_media,
        )
        return resp["ts"]

    async def _post_prose(
        self, target: SurfaceTarget, text: str, *, key: str | None, recover: bool
    ) -> str:
        """Sanitized standard-Markdown post; oversize prose becomes a snippet."""
        text = slack_format.sanitize(text)
        if len(text) <= slack_format.MARKDOWN_TEXT_LIMIT:
            return await self._post(
                target, markdown_text=text, key=key, recover=recover
            )
        return await self._post_artifact(
            target, slack_format.oversize_to_artifact(text), key=key, recover=recover
        )

    async def _post_artifact(
        self, target: SurfaceTarget, artifact: Artifact, *,
        key: str | None, recover: bool,
    ) -> str:
        """Upload the content as a hosted snippet, then post the keyed summary.

        The snippet is shared into the thread at upload time (explicit
        ``channel``/``thread_ts``) rather than relying on permalink-unfurl
        sharing, so the file is always visible to thread members. That share is
        the report; the keyed summary that follows is a short prose caption and
        the delivery *anchor* (the share itself cannot carry delivery metadata,
        so it is unkeyed — the summary is what recovery looks up). The summary
        deliberately carries **no** file permalink and posts with unfurling OFF:
        the file is already rendered by the share above, so a permalink would
        only unfurl into a duplicate preview (the reported triple-render was
        share + summary-with-link + that link's unfurl).

        Recovery looks the keyed summary up before any upload, so a crash-retry
        never duplicates the answer — the accepted cost is that a retry in the
        upload-to-post window can leave one duplicate file-share message.
        """
        if key and recover:
            existing = await self._find_by_key(target, key)
            if existing is not None:
                return existing
        summary = slack_format.sanitize(artifact.summary)
        uploaded_ok = False
        try:
            await self.client.files_upload_v2(
                content=artifact.content,
                filename=artifact.filename,
                title=artifact.title,
                snippet_type=artifact.snippet_type,
                channel=target.channel,
                thread_ts=target.thread,
            )
            uploaded_ok = True
        except Exception:  # noqa: BLE001 — the answer must land even if hosting fails
            logger.warning(
                "snippet upload failed for %s; posting content inline",
                artifact.filename, exc_info=True,
            )
        if uploaded_ok:
            # The share already rendered the file; this keyed caption is just
            # prose (no permalink) with unfurling off so nothing re-renders it.
            return await self._post(
                target,
                markdown_text=summary,
                key=key,
                recover=False,  # recovery already checked above
                unfurl_links=False,
                unfurl_media=False,
            )
        # Inline fallback renders as mrkdwn where `<!channel>` pings, so the
        # content is sanitized here — and only here: uploaded snippet bytes
        # must stay exact (files cannot ping; diffs must not be rewritten).
        body = f"{summary}\n\n{slack_format.sanitize(artifact.content)}"
        if len(body) > slack_format.FALLBACK_TEXT_LIMIT:
            body = body[: slack_format.FALLBACK_TEXT_LIMIT] + "\n… (truncated)"
        return await self._post(target, text=body, key=key, recover=False)

    async def update_approval(self, target, message_id, text, requests) -> None:
        # Local import keeps the Block Kit helper out of the surface-agnostic core.
        from openloop.surfaces.approvals import approval_blocks

        await self.client.chat_update(
            channel=target.channel,
            ts=message_id,
            text=text,
            blocks=approval_blocks(requests),
        )

    async def post_approval(
        self, target, text, requests, *, key=None, recover=False
    ) -> str:
        # Local import keeps the Block Kit helper out of the surface-agnostic core.
        from openloop.surfaces.approvals import approval_blocks

        return await self._post(
            target,
            text=text,
            blocks=approval_blocks(requests),
            key=key,
            recover=recover,
        )

    async def post_openhands_decision(
        self,
        target,
        job_id,
        decision_id,
        summary,
        *,
        key=None,
        recover=False,
    ) -> str:
        from openloop.surfaces.approvals import openhands_decision_blocks

        text = f"Confirmation needed: {summary}"
        return await self._post(
            target,
            text=text,
            blocks=openhands_decision_blocks(job_id, decision_id, summary),
            key=key,
            recover=recover,
        )

    async def update_openhands_decision(
        self, target, message_id, job_id, decision_id, summary
    ) -> None:
        from openloop.surfaces.approvals import openhands_decision_blocks

        await self.client.chat_update(
            channel=target.channel,
            ts=message_id,
            text=f"Confirmation needed: {summary}",
            blocks=openhands_decision_blocks(job_id, decision_id, summary),
        )

    async def post_final(
        self,
        target: SurfaceTarget,
        result: "Deliverable | str",
        *,
        key: str | None = None,
        recover: bool = False,
    ) -> str:
        deliverable = coerce(result)
        if isinstance(deliverable, Prose):
            return await self._post_prose(
                target, deliverable.text, key=key, recover=recover
            )
        return await self._post_artifact(
            target, deliverable, key=key, recover=recover
        )

    async def post_error(
        self,
        target: SurfaceTarget,
        text: str,
        *,
        key: str | None = None,
        recover: bool = False,
    ) -> str:
        # Errors are prose; the shared path also covers a pathological oversize
        # exception string via the snippet fallback.
        return await self._post_prose(target, text, key=key, recover=recover)
