"""Thread records — the durable, thread-scoped half of a persistent session.

A :class:`SurfaceSession` is the *per-turn* delivery record (one inbound event,
one visible task, one ``final_message_id``). A :class:`ThreadRecord` is the
*per-thread* aggregate: one row per thread scope, holding the delivered
conversation transcript and — in later phases — an ordered message inbox, an
active-turn claim, and a warm-context handle. The two are separated on purpose:
they have different cardinality (``N`` sessions per thread) and different
invariants (scheduler state vs delivery state), and stuffing thread-scoped state
onto a per-turn row would either duplicate it or force a "primary session" hack.

This module ships the **Phase A slice**: the *delivered transcript* lane only.
A completed, delivered turn contributes a :class:`TranscriptFragment`
(request → answer); a later turn in the same thread reads them back
oldest-first to seed its model context with the real conversation, not a summary.
The transcript is written *after* delivery is confirmed and the append is
idempotent on the turn id, so everything stored here is replayable by
construction — no pending/committed visibility flag. The internal (in-flight,
undelivered) turn log deliberately does **not** live here.

Like the session store it is a Protocol with an in-memory default and a Postgres
implementation, sharing ``SurfaceTarget`` for scope and the same pool as
``surface_sessions`` so a delivered turn can write both in one transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from sqlalchemy import exists, literal, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from openloop.db import BorrowedEngineStore
from openloop.sessions.schema import (
    surface_thread_inbox,
    surface_thread_transcript,
    surface_threads,
)
from openloop.sessions.store import SurfaceTarget


def _now() -> datetime:
    return datetime.now(UTC)


def _scope_key(target: SurfaceTarget) -> str:
    """Deterministic key for a thread scope (surface+workspace+agent+channel+thread).

    ``event_id`` is intentionally excluded — it identifies a single inbound event,
    not the thread. ``\\x1f`` (unit separator) can't appear in the ids, so the join
    is unambiguous; ``None`` channel/thread become empty segments.
    """
    return "\x1f".join(
        (
            target.surface,
            target.workspace,
            target.agent,
            target.channel or "",
            target.thread or "",
        )
    )


def thread_scope_key(target: SurfaceTarget) -> str:
    """Public alias for a thread's durable scope key (see :func:`_scope_key`).

    Doubles as the **warm-context key** (Phase B): the same string keys the
    thread's ``context_ref`` here *and* the process-local warm-workspace pool the
    coding worker draws from, so a follow-up turn in the thread reuses the
    checkout the prior turn warmed instead of cloning cold.
    """
    return _scope_key(target)


@dataclass(slots=True)
class TranscriptFragment:
    """One delivered exchange in a thread: the user's request and the answer.

    ``turn_id`` is the owning :class:`SurfaceSession` id — the idempotency key, so a
    redelivery or reconcile of the same turn never appends the fragment twice.
    """

    turn_id: str
    request: str
    answer: str
    created_at: datetime = field(default_factory=_now)


@dataclass(slots=True)
class InboxItem:
    """One queued inbound reply awaiting its turn on the thread.

    ``event_id`` is the dedup key (a re-delivered event never enqueues twice);
    ``payload`` is opaque to the store — the runner stashes what it needs to
    reconstruct the task + delivery target.
    """

    event_id: str
    payload: dict
    seq: int = 0


@dataclass(slots=True)
class ThreadRecord:
    """The per-thread aggregate: scope identity, plus (Phase B) an optional
    ``context_ref`` — a durable handle to a warm execution context (a kept git
    checkout, later a process/container) that a follow-up turn can reuse.

    The handle is only ever a *cache pointer*: the process-local pool is the
    authoritative liveness check, and a replica that finds no live context (a
    restart, another replica) simply reconstructs cold. So a stale or missing
    ``context_ref`` is always safe."""

    scope: SurfaceTarget
    context_ref: str | None = None
    created_at: datetime = field(default_factory=_now)


@runtime_checkable
class ThreadRecordStore(Protocol):
    async def get_or_create(self, scope: SurfaceTarget) -> ThreadRecord: ...

    async def append_delivered_fragment(
        self, scope: SurfaceTarget, fragment: TranscriptFragment
    ) -> None: ...

    async def replayable_transcript(
        self,
        scope: SurfaceTarget,
        *,
        exclude_turn_id: str | None = None,
        limit: int = 20,
    ) -> list[TranscriptFragment]: ...

    # --- inbox + active-turn claim (Phase C) ---

    async def append_inbox(
        self, scope: SurfaceTarget, event_id: str, payload: dict
    ) -> bool: ...

    async def try_begin_turn(self, scope: SurfaceTarget) -> bool: ...

    async def next_inbox(self, scope: SurfaceTarget) -> InboxItem | None: ...

    async def end_turn(self, scope: SurfaceTarget) -> None: ...

    async def reset_active_claims(self) -> int: ...

    # --- warm-context handle (Phase B) ---

    async def set_context_ref(
        self, scope_key: str, context_ref: str | None
    ) -> None: ...

    async def get_context_ref(self, scope_key: str) -> str | None: ...


class InMemoryThreadRecordStore:
    """Process-local thread records — good for dev and tests (not crash-durable)."""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadRecord] = {}
        # scope_key -> {turn_id: fragment}, insertion-ordered by first append.
        self._transcript: dict[str, dict[str, TranscriptFragment]] = {}
        # scope_key -> ordered pending inbox items; scope_key -> is-a-turn-active.
        self._inbox: dict[str, list[InboxItem]] = {}
        self._active: dict[str, bool] = {}
        # scope_key -> serialized warm-context handle (Phase B).
        self._context_ref: dict[str, str] = {}
        self._seq = 0

    async def get_or_create(self, scope: SurfaceTarget) -> ThreadRecord:
        key = _scope_key(scope)
        record = self._threads.get(key)
        if record is None:
            record = ThreadRecord(scope=scope)
            self._threads[key] = record
        record.context_ref = self._context_ref.get(key)
        return record

    async def append_delivered_fragment(
        self, scope: SurfaceTarget, fragment: TranscriptFragment
    ) -> None:
        await self.get_or_create(scope)
        fragments = self._transcript.setdefault(_scope_key(scope), {})
        # Idempotent on turn_id: first (delivered) write wins; a redelivery is a
        # no-op rather than a duplicate transcript entry.
        fragments.setdefault(fragment.turn_id, fragment)

    async def replayable_transcript(
        self,
        scope: SurfaceTarget,
        *,
        exclude_turn_id: str | None = None,
        limit: int = 20,
    ) -> list[TranscriptFragment]:
        fragments = list(self._transcript.get(_scope_key(scope), {}).values())
        fragments = [f for f in fragments if f.turn_id != exclude_turn_id]
        fragments.sort(key=lambda f: f.created_at)  # oldest-first
        return fragments[-limit:] if limit else fragments

    async def append_inbox(
        self, scope: SurfaceTarget, event_id: str, payload: dict
    ) -> bool:
        await self.get_or_create(scope)
        items = self._inbox.setdefault(_scope_key(scope), [])
        if any(it.event_id == event_id for it in items):
            return False  # dedup: this event is already pending
        self._seq += 1
        items.append(InboxItem(event_id=event_id, payload=payload, seq=self._seq))
        return True

    async def try_begin_turn(self, scope: SurfaceTarget) -> bool:
        # Claim the thread iff it is free AND there is pending work. The
        # "has pending work" condition is what lets the runner's drain loop
        # re-claim after releasing (to catch a reply that arrived mid-drain)
        # without spinning on an empty inbox.
        key = _scope_key(scope)
        if self._active.get(key):
            return False
        if not self._inbox.get(key):
            return False
        self._active[key] = True
        return True

    async def next_inbox(self, scope: SurfaceTarget) -> InboxItem | None:
        items = self._inbox.get(_scope_key(scope))
        if not items:
            return None
        return items.pop(0)  # oldest-first

    async def end_turn(self, scope: SurfaceTarget) -> None:
        self._active[_scope_key(scope)] = False

    async def reset_active_claims(self) -> int:
        held = sum(1 for v in self._active.values() if v)
        self._active.clear()
        return held

    async def set_context_ref(self, scope_key: str, context_ref: str | None) -> None:
        if context_ref is None:
            self._context_ref.pop(scope_key, None)
        else:
            self._context_ref[scope_key] = context_ref
        record = self._threads.get(scope_key)
        if record is not None:
            record.context_ref = context_ref

    async def get_context_ref(self, scope_key: str) -> str | None:
        return self._context_ref.get(scope_key)


class PostgresThreadRecordStore(BorrowedEngineStore):
    """Postgres-backed thread records — the durable delivered-transcript lane.

    ``surface_threads`` is one row per thread scope (active-turn and
    ``context_ref`` columns land in later phases); ``surface_thread_transcript``
    holds the delivered fragments as **rows** (not JSONB), keyed on
    ``(scope_key, turn_id)`` so the append is an idempotent UPSERT and ordered
    reads fall out of a serial ``seq``.
    """

    async def setup(self, engine: AsyncEngine) -> None:
        # sql-text: schema evolution moves to a migration tool; DDL is not
        # restated as metadata.
        async with self._setup_connection(engine) as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS surface_threads (
                        scope_key      TEXT PRIMARY KEY,
                        surface        TEXT NOT NULL,
                        workspace      TEXT NOT NULL,
                        agent          TEXT NOT NULL,
                        channel        TEXT,
                        thread         TEXT,
                        active_turn_id TEXT,
                        context_ref    TEXT,
                        created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            # Migration for thread rows created before the active-turn claim (C).
            await conn.execute(
                text(
                    "ALTER TABLE surface_threads "
                    "ADD COLUMN IF NOT EXISTS active_turn_id TEXT"
                )
            )
            # Migration for the Phase B warm-context handle.
            await conn.execute(
                text(
                    "ALTER TABLE surface_threads "
                    "ADD COLUMN IF NOT EXISTS context_ref TEXT"
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS surface_thread_transcript (
                        scope_key   TEXT NOT NULL,
                        turn_id     TEXT NOT NULL,
                        seq         BIGSERIAL,
                        request     TEXT NOT NULL,
                        answer      TEXT NOT NULL,
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (scope_key, turn_id)
                    )
                    """
                )
            )
            # Drives the ordered "most-recent N, oldest-first" transcript read.
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS surface_thread_transcript_seq_idx "
                    "ON surface_thread_transcript (scope_key, seq)"
                )
            )
            # Ordered inbox of pending replies (Phase C). `id` (serial) both orders
            # the drain and is the dedup-friendly key; UNIQUE(scope, event_id) makes
            # a re-delivered event a no-op INSERT while it is still pending.
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS surface_thread_inbox (
                        id          BIGSERIAL PRIMARY KEY,
                        scope_key   TEXT NOT NULL,
                        event_id    TEXT NOT NULL,
                        payload     JSONB NOT NULL,
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE (scope_key, event_id)
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS surface_thread_inbox_scope_idx "
                    "ON surface_thread_inbox (scope_key, id)"
                )
            )

    async def get_or_create(self, scope: SurfaceTarget) -> ThreadRecord:
        engine = self._require_engine()
        async with engine.begin() as conn:
            await conn.execute(_ensure_thread_row(scope))
        return ThreadRecord(scope=scope)

    async def append_delivered_fragment(
        self, scope: SurfaceTarget, fragment: TranscriptFragment
    ) -> None:
        engine = self._require_engine()
        key = _scope_key(scope)
        # Idempotent on (scope, turn): the first delivered write wins, so a
        # redelivery/reconcile of the same turn never double-appends.
        append = (
            insert(surface_thread_transcript)
            .values(
                scope_key=key,
                turn_id=fragment.turn_id,
                request=fragment.request,
                answer=fragment.answer,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    surface_thread_transcript.c.scope_key,
                    surface_thread_transcript.c.turn_id,
                ]
            )
        )
        # One transaction: the thread row and its fragment land together.
        async with engine.begin() as conn:
            await conn.execute(_ensure_thread_row(scope))
            await conn.execute(append)

    async def replayable_transcript(
        self,
        scope: SurfaceTarget,
        *,
        exclude_turn_id: str | None = None,
        limit: int = 20,
    ) -> list[TranscriptFragment]:
        engine = self._require_engine()
        # Most-recent `limit` fragments, then flipped to ascending so the caller
        # replays them oldest-first (mirrors surface_sessions.thread_history).
        inner = (
            select(
                surface_thread_transcript.c.turn_id,
                surface_thread_transcript.c.request,
                surface_thread_transcript.c.answer,
                surface_thread_transcript.c.created_at,
            )
            .where(surface_thread_transcript.c.scope_key == _scope_key(scope))
            .order_by(surface_thread_transcript.c.seq.desc())
            .limit(limit)
        )
        if exclude_turn_id is not None:
            inner = inner.where(surface_thread_transcript.c.turn_id != exclude_turn_id)
        recent = inner.subquery("recent")
        statement = select(recent).order_by(recent.c.created_at.asc())
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [
            TranscriptFragment(
                turn_id=r["turn_id"],
                request=r["request"],
                answer=r["answer"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def append_inbox(
        self, scope: SurfaceTarget, event_id: str, payload: dict
    ) -> bool:
        engine = self._require_engine()
        # Dedup on (scope, event_id) while the event is still pending.
        enqueue = (
            insert(surface_thread_inbox)
            .values(scope_key=_scope_key(scope), event_id=event_id, payload=payload)
            .on_conflict_do_nothing(
                index_elements=[
                    surface_thread_inbox.c.scope_key,
                    surface_thread_inbox.c.event_id,
                ]
            )
            .returning(surface_thread_inbox.c.id)
        )
        async with engine.begin() as conn:
            await conn.execute(_ensure_thread_row(scope))
            row = (await conn.execute(enqueue)).first()
        return row is not None

    async def try_begin_turn(self, scope: SurfaceTarget) -> bool:
        # Atomic CAS: claim the thread iff it is free AND has pending work. The
        # EXISTS clause means the runner's drain loop can re-claim after releasing
        # (to catch a reply that arrived mid-drain) without spinning on an empty
        # inbox — a free thread with nothing queued is simply not claimed.
        key = _scope_key(scope)
        statement = (
            surface_threads.update()
            .where(
                surface_threads.c.scope_key == key,
                surface_threads.c.active_turn_id.is_(None),
                exists(
                    select(literal(1)).where(surface_thread_inbox.c.scope_key == key)
                ),
            )
            .values(active_turn_id="held")
            .returning(surface_threads.c.scope_key)
        )
        engine = self._require_engine()
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).first()
        return row is not None

    async def next_inbox(self, scope: SurfaceTarget) -> InboxItem | None:
        oldest = (
            select(surface_thread_inbox.c.id)
            .where(surface_thread_inbox.c.scope_key == _scope_key(scope))
            .order_by(surface_thread_inbox.c.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        statement = (
            surface_thread_inbox.delete()
            .where(surface_thread_inbox.c.id == oldest)
            .returning(
                surface_thread_inbox.c.id,
                surface_thread_inbox.c.event_id,
                surface_thread_inbox.c.payload,
            )
        )
        engine = self._require_engine()
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).mappings().first()
        if row is None:
            return None
        return InboxItem(
            event_id=row["event_id"],
            # JSONB arrives decoded — the dialect registers a jsonb codec.
            payload=row["payload"],
            seq=row["id"],
        )

    async def end_turn(self, scope: SurfaceTarget) -> None:
        engine = self._require_engine()
        statement = (
            surface_threads.update()
            .where(surface_threads.c.scope_key == _scope_key(scope))
            .values(active_turn_id=None)
        )
        async with engine.begin() as conn:
            await conn.execute(statement)

    async def reset_active_claims(self) -> int:
        """Clear every active-turn claim. Called once at startup: a crashed drain
        leader would otherwise leave ``active_turn_id`` set forever, wedging the
        thread. Single-replica-correct (a restart means nothing is draining); the
        multi-replica version is a leased claim, not a blanket reset."""
        statement = (
            surface_threads.update()
            .where(surface_threads.c.active_turn_id.is_not(None))
            .values(active_turn_id=None)
        )
        engine = self._require_engine()
        async with engine.begin() as conn:
            result = await conn.execute(statement)
        # The driver's "UPDATE 3" status string is gone; rowcount says the same.
        return result.rowcount

    async def set_context_ref(self, scope_key: str, context_ref: str | None) -> None:
        """Persist (or clear) the thread's warm-context handle.

        Keyed by the raw ``scope_key`` because the caller is the warm-workspace
        pool, which holds only that string — not the full :class:`SurfaceTarget`.
        An UPDATE (never an INSERT): the thread row is created by the inbox/
        transcript path before any turn — and hence any warm context — exists, so
        a missing row means the thread was never seen and the (best-effort, cache)
        handle is simply dropped.
        """
        engine = self._require_engine()
        statement = (
            surface_threads.update()
            .where(surface_threads.c.scope_key == scope_key)
            .values(context_ref=context_ref)
        )
        async with engine.begin() as conn:
            await conn.execute(statement)

    async def get_context_ref(self, scope_key: str) -> str | None:
        engine = self._require_engine()
        statement = select(surface_threads.c.context_ref).where(
            surface_threads.c.scope_key == scope_key
        )
        async with engine.connect() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return row["context_ref"] if row is not None else None


def _ensure_thread_row(scope: SurfaceTarget):
    """Insert the thread's row if this scope has never been seen before."""
    return (
        insert(surface_threads)
        .values(
            scope_key=_scope_key(scope),
            surface=scope.surface,
            workspace=scope.workspace,
            agent=scope.agent,
            channel=scope.channel,
            thread=scope.thread,
        )
        .on_conflict_do_nothing(index_elements=[surface_threads.c.scope_key])
    )
