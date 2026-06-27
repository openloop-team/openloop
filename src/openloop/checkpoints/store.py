"""Worker checkpoints — make a coding-worker job resumable across a restart.

Phase B's durability is scoped to the worker. After each step the connector
persists a :class:`WorkerCheckpoint` keyed by ``job_id`` (the stable identity
minted before approval). On a mid-execution crash the job resumes from the last
checkpoint instead of restarting, and the idempotency keys on
:class:`~openloop.tools.coding_worker.WorkerState` keep the durable side effects
(branch push, PR open) from being duplicated.

The record stores **both** ``state_json`` (the full serialized worker state)
**and** ``completed_steps`` as its own column — not just an enum ``status``.
Failures happen *halfway through* a named phase, so a coarse status alone would
lie about partial progress. Phase C generalizes this into a workflow store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

# Coarse lifecycle, for querying/observability. The fine-grained truth is
# completed_steps + state_json.
#   running       — worker in flight (branch not yet pushed)
#   pushed        — branch pushed; PR not opened yet
#   opened        — draft PR created (terminal success)
#   failed        — worker failed before pushing
#   open_pr_failed— branch pushed but opening the PR failed (resumable)
TERMINAL_OK = "opened"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class WorkerCheckpoint:
    """A persisted snapshot of one coding-worker job."""

    job_id: str
    repo: str
    instruction: str
    base: str
    branch: str
    status: str
    completed_steps: list[str] = field(default_factory=list)
    state_json: dict = field(default_factory=dict)
    title: str | None = None
    body: str | None = None
    pr_number: int | None = None
    pr_url: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@runtime_checkable
class CheckpointStore(Protocol):
    async def get(self, job_id: str) -> WorkerCheckpoint | None: ...

    async def upsert(self, checkpoint: WorkerCheckpoint) -> None: ...

    async def recent(self, limit: int = 50) -> list[WorkerCheckpoint]: ...


class InMemoryCheckpointStore:
    """Process-local checkpoints — good for dev and tests (not crash-durable)."""

    def __init__(self) -> None:
        self._by_id: dict[str, WorkerCheckpoint] = {}

    async def get(self, job_id: str) -> WorkerCheckpoint | None:
        return self._by_id.get(job_id)

    async def upsert(self, checkpoint: WorkerCheckpoint) -> None:
        existing = self._by_id.get(checkpoint.job_id)
        if existing is not None:
            checkpoint.created_at = existing.created_at
        checkpoint.updated_at = _now()
        self._by_id[checkpoint.job_id] = checkpoint

    async def recent(self, limit: int = 50) -> list[WorkerCheckpoint]:
        ordered = sorted(
            self._by_id.values(), key=lambda c: c.updated_at, reverse=True
        )
        return ordered[:limit]
