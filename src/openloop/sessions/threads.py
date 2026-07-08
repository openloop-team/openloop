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

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from openloop.sessions.store import SurfaceTarget


def _now() -> datetime:
    return datetime.now(timezone.utc)


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

    async def set_context_ref(
        self, scope_key: str, context_ref: str | None
    ) -> None:
        if context_ref is None:
            self._context_ref.pop(scope_key, None)
        else:
            self._context_ref[scope_key] = context_ref
        record = self._threads.get(scope_key)
        if record is not None:
            record.context_ref = context_ref

    async def get_context_ref(self, scope_key: str) -> str | None:
        return self._context_ref.get(scope_key)


class PostgresThreadRecordStore:
    """Postgres-backed thread records — the durable delivered-transcript lane.

    ``surface_threads`` is one row per thread scope (active-turn and
    ``context_ref`` columns land in later phases); ``surface_thread_transcript``
    holds the delivered fragments as **rows** (not JSONB), keyed on
    ``(scope_key, turn_id)`` so the append is an idempotent UPSERT and ordered
    reads fall out of a serial ``seq``.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None  # asyncpg.Pool, created in setup()

    async def setup(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(
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
            # Migration for thread rows created before the active-turn claim (C).
            await conn.execute(
                "ALTER TABLE surface_threads "
                "ADD COLUMN IF NOT EXISTS active_turn_id TEXT"
            )
            # Migration for the Phase B warm-context handle.
            await conn.execute(
                "ALTER TABLE surface_threads "
                "ADD COLUMN IF NOT EXISTS context_ref TEXT"
            )
            await conn.execute(
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
            # Drives the ordered "most-recent N, oldest-first" transcript read.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS surface_thread_transcript_seq_idx "
                "ON surface_thread_transcript (scope_key, seq)"
            )
            # Ordered inbox of pending replies (Phase C). `id` (serial) both orders
            # the drain and is the dedup-friendly key; UNIQUE(scope, event_id) makes
            # a re-delivered event a no-op INSERT while it is still pending.
            await conn.execute(
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
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS surface_thread_inbox_scope_idx "
                "ON surface_thread_inbox (scope_key, id)"
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError(
                "PostgresThreadRecordStore.setup() must be called first"
            )
        return self._pool

    async def get_or_create(self, scope: SurfaceTarget) -> ThreadRecord:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO surface_threads
                    (scope_key, surface, workspace, agent, channel, thread)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (scope_key) DO NOTHING
                """,
                _scope_key(scope),
                scope.surface,
                scope.workspace,
                scope.agent,
                scope.channel,
                scope.thread,
            )
        return ThreadRecord(scope=scope)

    async def append_delivered_fragment(
        self, scope: SurfaceTarget, fragment: TranscriptFragment
    ) -> None:
        pool = self._require_pool()
        key = _scope_key(scope)
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO surface_threads
                        (scope_key, surface, workspace, agent, channel, thread)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (scope_key) DO NOTHING
                    """,
                    key, scope.surface, scope.workspace, scope.agent,
                    scope.channel, scope.thread,
                )
                # Idempotent on (scope, turn): the first delivered write wins, so a
                # redelivery/reconcile of the same turn never double-appends.
                await conn.execute(
                    """
                    INSERT INTO surface_thread_transcript
                        (scope_key, turn_id, request, answer)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (scope_key, turn_id) DO NOTHING
                    """,
                    key, fragment.turn_id, fragment.request, fragment.answer,
                )

    async def replayable_transcript(
        self,
        scope: SurfaceTarget,
        *,
        exclude_turn_id: str | None = None,
        limit: int = 20,
    ) -> list[TranscriptFragment]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Most-recent `limit` fragments, then flipped to ascending so the caller
            # replays them oldest-first (mirrors surface_sessions.thread_history).
            rows = await conn.fetch(
                """
                SELECT * FROM (
                    SELECT turn_id, request, answer, created_at
                    FROM surface_thread_transcript
                    WHERE scope_key = $1
                      AND ($2::text IS NULL OR turn_id <> $2)
                    ORDER BY seq DESC
                    LIMIT $3
                ) recent
                ORDER BY created_at ASC
                """,
                _scope_key(scope),
                exclude_turn_id,
                limit,
            )
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
        pool = self._require_pool()
        key = _scope_key(scope)
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO surface_threads
                        (scope_key, surface, workspace, agent, channel, thread)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (scope_key) DO NOTHING
                    """,
                    key, scope.surface, scope.workspace, scope.agent,
                    scope.channel, scope.thread,
                )
                # Dedup on (scope, event_id) while the event is still pending.
                row = await conn.fetchrow(
                    """
                    INSERT INTO surface_thread_inbox (scope_key, event_id, payload)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (scope_key, event_id) DO NOTHING
                    RETURNING id
                    """,
                    key, event_id, json.dumps(payload),
                )
        return row is not None

    async def try_begin_turn(self, scope: SurfaceTarget) -> bool:
        # Atomic CAS: claim the thread iff it is free AND has pending work. The
        # EXISTS clause means the runner's drain loop can re-claim after releasing
        # (to catch a reply that arrived mid-drain) without spinning on an empty
        # inbox — a free thread with nothing queued is simply not claimed.
        pool = self._require_pool()
        key = _scope_key(scope)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE surface_threads SET active_turn_id = 'held'
                WHERE scope_key = $1 AND active_turn_id IS NULL
                  AND EXISTS (
                    SELECT 1 FROM surface_thread_inbox WHERE scope_key = $1
                  )
                RETURNING scope_key
                """,
                key,
            )
        return row is not None

    async def next_inbox(self, scope: SurfaceTarget) -> InboxItem | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                DELETE FROM surface_thread_inbox
                WHERE id = (
                    SELECT id FROM surface_thread_inbox
                    WHERE scope_key = $1 ORDER BY id LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, event_id, payload
                """,
                _scope_key(scope),
            )
        if row is None:
            return None
        return InboxItem(
            event_id=row["event_id"],
            payload=json.loads(row["payload"]),
            seq=row["id"],
        )

    async def end_turn(self, scope: SurfaceTarget) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE surface_threads SET active_turn_id = NULL WHERE scope_key = $1",
                _scope_key(scope),
            )

    async def reset_active_claims(self) -> int:
        """Clear every active-turn claim. Called once at startup: a crashed drain
        leader would otherwise leave ``active_turn_id`` set forever, wedging the
        thread. Single-replica-correct (a restart means nothing is draining); the
        multi-replica version is a leased claim, not a blanket reset."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE surface_threads SET active_turn_id = NULL "
                "WHERE active_turn_id IS NOT NULL"
            )
        # asyncpg returns e.g. "UPDATE 3"; parse the count defensively.
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def set_context_ref(
        self, scope_key: str, context_ref: str | None
    ) -> None:
        """Persist (or clear) the thread's warm-context handle.

        Keyed by the raw ``scope_key`` because the caller is the warm-workspace
        pool, which holds only that string — not the full :class:`SurfaceTarget`.
        An UPDATE (never an INSERT): the thread row is created by the inbox/
        transcript path before any turn — and hence any warm context — exists, so
        a missing row means the thread was never seen and the (best-effort, cache)
        handle is simply dropped.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE surface_threads SET context_ref = $2 WHERE scope_key = $1",
                scope_key,
                context_ref,
            )

    async def get_context_ref(self, scope_key: str) -> str | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT context_ref FROM surface_threads WHERE scope_key = $1",
                scope_key,
            )
        return row["context_ref"] if row is not None else None
