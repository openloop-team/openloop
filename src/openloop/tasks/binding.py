"""Thread↔task bindings — one durable thread, one continuable workspace task.

A :class:`~openloop.tasks.contract.WorkspaceTask` today lives only inside the
records of whatever ran it: a workflow instance's ``state``, a worker
checkpoint's ``state_json``. That makes its identity a property of one workflow
run, so a later reply in the same thread has nothing to continue — the model
re-delegates and a second task (a second branch, a second pull request, a second
approval) is born.

A :class:`ThreadTask` is that identity held **outside** any single run: the
durable row that says *this thread scope is executing this workspace task*, plus
the serialized task itself. It is the cold-reconstruction source — a replica
that has never seen the thread can read one row and rebuild the task — so no
continuation invariant depends on a live workflow, a warm checkout, or an
in-process session (ADR 0005).

Cardinality is one open binding per thread scope: a thread executes at most one
workspace task at a time, and that task outlives the turns that drive it. The
``state`` blob is the serialized ``WorkspaceTask``, refreshed as the task
progresses, so ``WorkspaceTask.from_dict(record.state)`` is always a faithful,
run-independent view of the task.

``status`` is a **claim**, not a lifecycle: ``open`` (nothing is driving it),
``busy`` (a turn has claimed it), ``closed`` (never continue again — its
authorization was denied, or the task was retired). Whether work is actually
in flight is derived from the durable workflow instance, never from this column
alone; the claim exists so two replicas can't drive one task concurrently, and
a claim orphaned by a crash is repaired from the instance at startup.

Like the other stores it is a Protocol with an in-memory default and a Postgres
implementation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

# A binding nothing is driving: an eligible reply may claim and continue it.
OPEN = "open"
# A turn holds the binding. A concurrent reply must wait, never race it.
BUSY = "busy"
# Retired: denied at its start gate, or explicitly closed. Never continuable.
CLOSED = "closed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ThreadTask:
    """One durable thread scope bound to one reconstructable workspace task.

    ``scope_key`` is the thread's durable scope key
    (:func:`~openloop.sessions.threads.thread_scope_key`) — the same string the
    warm-workspace pool is keyed on, so the binding addresses exactly the
    conversation the work was requested in.

    Identity fields (``agent``, ``agent_id``, ``approval_id``, ``requested_by``)
    are the task's **authorization envelope**, captured when the task was
    created and never re-derived from a later turn: spend attributes to the
    agent that was authorized, and the human who may continue the task is the
    one who started it. ``instance_id`` names the workflow instance driving the
    current (or most recent) turn — it changes every turn, which is precisely
    why task identity cannot live there.
    """

    task_id: str
    scope_key: str
    profile: str
    entry_action: str
    status: str = OPEN
    agent: str | None = None
    agent_id: str | None = None
    approval_id: str | None = None
    # Surface identity of the human who initiated the task (the mention author,
    # as the surface names them). Compared against a later reply's author to
    # decide continuation eligibility, so it must stay in the surface's own id
    # namespace — never a display handle.
    requested_by: str | None = None
    instance_id: str | None = None
    session_id: str | None = None
    # How many turns have driven this task (1 = the turn that created it).
    turns: int = 1
    # The serialized WorkspaceTask — the cold-reconstruction source of truth.
    state: dict = field(default_factory=dict)
    closed_reason: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@runtime_checkable
class ThreadTaskStore(Protocol):
    async def get(self, task_id: str) -> ThreadTask | None: ...

    async def bound(self, scope_key: str) -> ThreadTask | None:
        """The thread's continuable binding, if it has one (never ``closed``)."""
        ...

    async def bind(self, record: ThreadTask) -> ThreadTask:
        """Bind a thread to a task; idempotent on ``task_id``.

        A task_id already bound returns its stored row untouched — the birth
        record is written once, so a retried invoke can never re-open a closed
        task or re-point it at another thread.
        """
        ...

    async def claim(
        self, task_id: str, *, instance_id: str, session_id: str | None = None
    ) -> ThreadTask | None:
        """Atomically take the binding for one turn (``open`` → ``busy``).

        ``None`` means the row is missing, closed, or already claimed — the
        caller must not drive it.
        """
        ...

    async def release(
        self,
        task_id: str,
        *,
        state: dict | None = None,
        status: str = OPEN,
        reason: str | None = None,
    ) -> ThreadTask | None:
        """Release the claim, optionally refreshing the serialized task.

        Idempotent: releasing an unclaimed row still records ``state``, so a
        turn delivered twice (inline plus the terminal callback) is harmless.
        """
        ...

    async def retire(self, task_id: str, reason: str) -> ThreadTask | None:
        """Retire a binding for good. A closed task never continues.

        Deliberately not named ``close``: a Postgres-backed store already owes
        that name to its pool lifecycle, and shadowing it would leave the
        borrowed pool attached at shutdown.
        """
        ...

    async def claimed(self, limit: int = 200) -> list[ThreadTask]:
        """Every ``busy`` row — the startup sweep's input for orphan repair."""
        ...


class InMemoryThreadTaskStore:
    """Process-local bindings — good for dev and tests (not crash-durable).

    Snapshot-isolated like the Postgres rows it stands in for: every boundary
    copies the record *including its ``state`` blob*, so mutating a record you
    were handed never reaches the store except through a store operation.
    Without this the two backends diverge exactly where it matters — Postgres
    deserializes fresh JSONB per read, while a shared dict here would let a
    caller advance the durable task before the turn describing it has run.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, ThreadTask] = {}

    @staticmethod
    def _snapshot(record: ThreadTask) -> ThreadTask:
        return replace(record, state=copy.deepcopy(record.state))

    async def get(self, task_id: str) -> ThreadTask | None:
        record = self._by_id.get(task_id)
        return self._snapshot(record) if record is not None else None

    async def bound(self, scope_key: str) -> ThreadTask | None:
        if not scope_key:
            return None
        matches = [
            r
            for r in self._by_id.values()
            if r.scope_key == scope_key and r.status != CLOSED
        ]
        if not matches:
            return None
        # Newest wins: a thread that (legitimately) started a second task after
        # the first was closed continues the one it is actually working on.
        matches.sort(key=lambda r: r.created_at)
        return self._snapshot(matches[-1])

    async def bind(self, record: ThreadTask) -> ThreadTask:
        existing = self._by_id.get(record.task_id)
        if existing is not None:
            return self._snapshot(existing)
        stored = self._snapshot(record)
        stored.created_at = stored.updated_at = _now()
        self._by_id[record.task_id] = stored
        return self._snapshot(stored)

    async def claim(
        self, task_id: str, *, instance_id: str, session_id: str | None = None
    ) -> ThreadTask | None:
        # No await between the check and the mutation: atomic within one loop.
        record = self._by_id.get(task_id)
        if record is None or record.status != OPEN:
            return None
        record.status = BUSY
        record.instance_id = instance_id
        record.session_id = session_id
        record.turns += 1
        record.updated_at = _now()
        return self._snapshot(record)

    async def release(
        self,
        task_id: str,
        *,
        state: dict | None = None,
        status: str = OPEN,
        reason: str | None = None,
    ) -> ThreadTask | None:
        record = self._by_id.get(task_id)
        if record is None:
            return None
        if record.status != CLOSED:
            record.status = status
            record.closed_reason = reason if status == CLOSED else None
        if state is not None:
            record.state = copy.deepcopy(state)
        record.updated_at = _now()
        return self._snapshot(record)

    async def retire(self, task_id: str, reason: str) -> ThreadTask | None:
        return await self.release(task_id, status=CLOSED, reason=reason)

    async def claimed(self, limit: int = 200) -> list[ThreadTask]:
        rows = [r for r in self._by_id.values() if r.status == BUSY]
        rows.sort(key=lambda r: r.updated_at)
        return [self._snapshot(r) for r in rows[:limit]]
