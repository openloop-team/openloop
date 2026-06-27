"""Surface delivery — posting progress and final answers back to a surface.

Phase D decouples the *answer* from the inbound request lifecycle. A
:class:`SurfaceDelivery` is the surface-agnostic seam the session runner uses to
post a short progress message, update it as the agent works, and post the final
answer (or an error) later — possibly long after the original HTTP/Bolt request
returned.

Delivery must be **idempotent**: the runner persists the ids returned here on the
session, so a crash-and-resume or a duplicate inbound event reuses the existing
progress/final message instead of posting a second one. The protocol returns the
provider message id from each post; ``update_progress`` takes one back.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from openloop.sessions.store import SurfaceTarget

logger = logging.getLogger(__name__)


@runtime_checkable
class SurfaceDelivery(Protocol):
    async def post_progress(self, target: SurfaceTarget, text: str) -> str:
        """Post a new progress message; return its provider id."""
        ...

    async def update_progress(
        self, target: SurfaceTarget, message_id: str, text: str
    ) -> None:
        """Edit an existing progress message in place."""
        ...

    async def post_final(
        self, target: SurfaceTarget, text: str, *, blocks: list[dict] | None = None
    ) -> str:
        """Post the final answer; return its provider id."""
        ...

    async def post_error(self, target: SurfaceTarget, text: str) -> str:
        """Post an error/interrupted notice; return its provider id."""
        ...


class SlackSurfaceDelivery:
    """Delivers to Slack via a Bolt/`AsyncWebClient` ``client``.

    Uses the stored channel + thread and message timestamps Slack returns so the
    runner can dedupe and update. Threading is best-effort: if a target has no
    thread it posts at the channel root.
    """

    def __init__(self, client) -> None:  # AsyncWebClient
        self.client = client

    async def post_progress(self, target: SurfaceTarget, text: str) -> str:
        resp = await self.client.chat_postMessage(
            channel=target.channel, thread_ts=target.thread, text=text
        )
        return resp["ts"]

    async def update_progress(
        self, target: SurfaceTarget, message_id: str, text: str
    ) -> None:
        await self.client.chat_update(
            channel=target.channel, ts=message_id, text=text
        )

    async def post_final(
        self, target: SurfaceTarget, text: str, *, blocks: list[dict] | None = None
    ) -> str:
        resp = await self.client.chat_postMessage(
            channel=target.channel,
            thread_ts=target.thread,
            text=text,
            blocks=blocks,
        )
        return resp["ts"]

    async def post_error(self, target: SurfaceTarget, text: str) -> str:
        return await self.post_final(target, text)
