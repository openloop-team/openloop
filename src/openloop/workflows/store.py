"""Durable workflow instances — the general form of Phase B's worker checkpoint.

A :class:`WorkflowInstance` is one running workflow: its position (which steps are
done, whether it is parked on a wait node), a JSON ``state`` snapshot, and a
terminal ``result`` / ``error``. The store persists it after every step so a
crash resumes from the last completed step, exactly like the worker checkpoint —
but for any workflow, not just the coding worker. Phase C generalizes the
checkpoint store into this; the worker becomes one workflow on top of it.

Every mutation is an atomic claim, fence, or eviction — there is deliberately
no unfenced write operation. ``drive_gen`` and ``leased_until`` are
**store-owned**: no operation writes them from the instance payload. The gen
bumps only on ownership transitions (claim, release, evict), and only
``claim_drive`` / ``renew_lease`` set the lease, so a stale driver's snapshot
can neither carry an old gen back into the row nor roll back a renewed lease.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

# Terminal statuses never resume; "running" resumes on startup (crashed mid-step);
# "waiting" stays parked until an event wakes it. "abandoned" is a crashed run of
# a non-resumable workflow (e.g. a chat turn — we never replay paid model calls).
TERMINAL = ("completed", "failed", "cancelled", "abandoned")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class WorkflowInstance:
    """A persisted snapshot of one workflow run."""

    id: str
    workflow: str
    status: str = "running"  # running | waiting | completed | failed
    completed_steps: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    waiting_on: str | None = None  # name of the wait node it is parked at
    result: dict | None = None
    error: str | None = None
    leased_until: datetime | None = None
    drive_gen: int = 0  # ownership fence — bumped on claim/release/evict only
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@runtime_checkable
class WorkflowStore(Protocol):
    async def get(self, instance_id: str) -> WorkflowInstance | None: ...

    async def create(self, instance: WorkflowInstance) -> bool:
        """Insert a new instance; ``False`` (and no write) if the id exists."""
        ...

    async def claim_drive(
        self, instance_id: str, *, lease_seconds: float
    ) -> WorkflowInstance | None:
        """Atomically take drive ownership of a ``running`` instance.

        Wins only when the lease is absent or expired (store-clock ``now``);
        bumps ``drive_gen`` and returns the claimed row carrying the new gen.
        ``None`` means another driver holds a live lease, or the instance is
        parked, terminal, or missing.
        """
        ...

    async def fenced_write(
        self, instance: WorkflowInstance, gen: int, *, release: bool = False
    ) -> bool:
        """Write ``instance``'s fields iff the row still carries ``gen``.

        ``release=True`` (park/terminal writes) also bumps the gen and clears
        the lease, invalidating the writer's own ownership. Ordinary writes
        leave ``drive_gen`` and ``leased_until`` untouched — the payload's
        values for either are ignored.
        """
        ...

    async def renew_lease(
        self, instance_id: str, gen: int, *, lease_seconds: float
    ) -> bool:
        """Extend the lease iff the row still carries ``gen``."""
        ...

    async def cancel_instance(
        self, instance_id: str, reason: str
    ) -> WorkflowInstance | None:
        """Atomically cancel a non-terminal instance, evicting any live driver.

        Bumps ``drive_gen`` so the driver's next fenced write fails. ``None``
        means the instance was already terminal or missing — the caller lost
        the cancel race (or there was nothing to cancel).
        """
        ...

    async def claim_event(
        self, instance_id: str, event: str, payload: dict
    ) -> WorkflowInstance | None:
        """Atomically consume one exact wait event; leaves the lease unset.

        Event consumption and drive ownership are separate claims: the woken
        instance is ``running`` with no lease, and whoever reaches
        :meth:`claim_drive` first drives it.
        """
        ...

    async def recent(self, limit: int = 100) -> list[WorkflowInstance]: ...


class InMemoryWorkflowStore:
    """Process-local instances — good for dev and tests (not crash-durable).

    Snapshot-isolated like the Postgres rows it stands in for: every boundary
    deep-copies, so mutating an instance you were handed never reaches the
    store except through a store operation. Without this the fence would be
    bypassable by aliasing, and replica tests sharing one store would prove
    nothing.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, WorkflowInstance] = {}

    async def get(self, instance_id: str) -> WorkflowInstance | None:
        stored = self._by_id.get(instance_id)
        return copy.deepcopy(stored) if stored is not None else None

    async def create(self, instance: WorkflowInstance) -> bool:
        if instance.id in self._by_id:
            return False
        self._by_id[instance.id] = copy.deepcopy(instance)
        return True

    async def claim_drive(
        self, instance_id: str, *, lease_seconds: float
    ) -> WorkflowInstance | None:
        # No await between check and mutation: atomic within one event loop.
        stored = self._by_id.get(instance_id)
        now = _now()
        if stored is None or stored.status != "running":
            return None
        if stored.leased_until is not None and stored.leased_until > now:
            return None
        stored.drive_gen += 1
        stored.leased_until = now + timedelta(seconds=lease_seconds)
        stored.updated_at = now
        return copy.deepcopy(stored)

    async def fenced_write(
        self, instance: WorkflowInstance, gen: int, *, release: bool = False
    ) -> bool:
        stored = self._by_id.get(instance.id)
        if stored is None or stored.drive_gen != gen:
            return False
        snapshot = copy.deepcopy(instance)
        # Store-owned fields: never taken from the payload.
        snapshot.drive_gen = stored.drive_gen + (1 if release else 0)
        snapshot.leased_until = None if release else stored.leased_until
        snapshot.created_at = stored.created_at
        snapshot.updated_at = _now()
        self._by_id[instance.id] = snapshot
        return True

    async def renew_lease(
        self, instance_id: str, gen: int, *, lease_seconds: float
    ) -> bool:
        stored = self._by_id.get(instance_id)
        if stored is None or stored.drive_gen != gen:
            return False
        now = _now()
        stored.leased_until = now + timedelta(seconds=lease_seconds)
        stored.updated_at = now
        return True

    async def cancel_instance(
        self, instance_id: str, reason: str
    ) -> WorkflowInstance | None:
        stored = self._by_id.get(instance_id)
        if stored is None or stored.status in TERMINAL:
            return None
        stored.status = "cancelled"
        stored.error = reason or None
        stored.waiting_on = None
        stored.leased_until = None
        stored.drive_gen += 1
        stored.updated_at = _now()
        return copy.deepcopy(stored)

    async def claim_event(
        self, instance_id: str, event: str, payload: dict
    ) -> WorkflowInstance | None:
        stored = self._by_id.get(instance_id)
        if (
            stored is None
            or stored.status != "waiting"
            or stored.waiting_on != event
        ):
            return None
        stored.state.setdefault("events", {})[event] = copy.deepcopy(payload)
        stored.completed_steps.append(event)
        stored.status = "running"
        stored.waiting_on = None
        stored.leased_until = None
        stored.updated_at = _now()
        return copy.deepcopy(stored)

    async def recent(self, limit: int = 100) -> list[WorkflowInstance]:
        ordered = sorted(
            self._by_id.values(), key=lambda i: i.updated_at, reverse=True
        )
        return [copy.deepcopy(i) for i in ordered[:limit]]
