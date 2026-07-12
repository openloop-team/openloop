"""Metadata for surface file uploads usable as sealed-analysis inputs.

Staging is deliberately **lazy** (Phase 4): when a user shares a file in a
thread, the surface records only metadata here — nothing a user posts is
retained unless an approved analysis later asks for it, at which point the
provisioner fetches the bytes from the surface's file API.

Every record is bound to the thread it was shared in via ``scope_key`` — the
full thread-ownership tuple key ``(surface, workspace, agent, channel,
thread)`` from :func:`openloop.sessions.threads.thread_scope_key`, NOT a bare
thread id (which would collide across channels and workspaces). An upload is
provisionable only by a request whose gateway-stamped scope matches; requests
from scopeless paths (the direct tools API) carry no scope and are refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class UploadRecord:
    """One shared file's trusted metadata — never its bytes."""

    upload_ref: str  # the surface's file id (e.g. a Slack file id)
    scope_key: str  # thread-ownership tuple key the file was shared under
    name: str
    size: int
    user: str | None = None
    shared_at: datetime = field(default_factory=_now)


@runtime_checkable
class UploadStore(Protocol):
    async def record(self, upload: UploadRecord) -> None: ...

    async def get(self, upload_ref: str) -> UploadRecord | None: ...

    async def for_scope(
        self, scope_key: str, *, limit: int = 20
    ) -> list[UploadRecord]: ...


class InMemoryUploadStore:
    """Process-local upload metadata — good for dev and tests."""

    def __init__(self) -> None:
        self._by_ref: dict[str, UploadRecord] = {}

    async def record(self, upload: UploadRecord) -> None:
        # First write wins: a surface re-delivering the same file event must
        # not move an already-recorded upload into a different scope.
        self._by_ref.setdefault(upload.upload_ref, upload)

    async def get(self, upload_ref: str) -> UploadRecord | None:
        return self._by_ref.get(upload_ref)

    async def for_scope(
        self, scope_key: str, *, limit: int = 20
    ) -> list[UploadRecord]:
        matches = [
            u for u in self._by_ref.values() if u.scope_key == scope_key
        ]
        matches.sort(key=lambda u: u.shared_at)
        return matches[-limit:] if limit else matches
