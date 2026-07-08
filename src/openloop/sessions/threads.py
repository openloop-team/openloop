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
class ThreadRecord:
    """The per-thread aggregate. Phase A carries only its scope identity; the inbox,
    active-turn claim, and ``context_ref`` land in later phases."""

    scope: SurfaceTarget
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


class InMemoryThreadRecordStore:
    """Process-local thread records — good for dev and tests (not crash-durable)."""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadRecord] = {}
        # scope_key -> {turn_id: fragment}, insertion-ordered by first append.
        self._transcript: dict[str, dict[str, TranscriptFragment]] = {}

    async def get_or_create(self, scope: SurfaceTarget) -> ThreadRecord:
        key = _scope_key(scope)
        record = self._threads.get(key)
        if record is None:
            record = ThreadRecord(scope=scope)
            self._threads[key] = record
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
                    scope_key   TEXT PRIMARY KEY,
                    surface     TEXT NOT NULL,
                    workspace   TEXT NOT NULL,
                    agent       TEXT NOT NULL,
                    channel     TEXT,
                    thread      TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
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
