"""Postgres-backed surface sessions — async tasks survive a process restart.

Mirrors :class:`InMemorySurfaceSessionStore` against a ``surface_sessions``
table, following the approvals/usage/checkpoint/workflow store pattern. The
surface target is flattened into columns so the startup reconciler can query by
status without deserializing JSON.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from openloop.db import BorrowedEngineStore
from openloop.sessions.schema import surface_sessions
from openloop.sessions.store import SurfaceSession, SurfaceTarget


class PostgresSurfaceSessionStore(BorrowedEngineStore):
    async def setup(self, engine: AsyncEngine) -> None:
        # sql-text: schema evolution moves to a migration tool; DDL is not
        # restated as metadata.
        async with self._setup_connection(engine) as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS surface_sessions (
                        id                   TEXT PRIMARY KEY,
                        surface              TEXT NOT NULL,
                        workspace            TEXT NOT NULL,
                        agent                TEXT NOT NULL,
                        channel              TEXT,
                        thread               TEXT,
                        event_id             TEXT,
                        status               TEXT NOT NULL,
                        workflow_instance_id TEXT,
                        progress_message_id  TEXT,
                        final_message_id     TEXT,
                        approval_ids         JSONB NOT NULL DEFAULT '[]',
                        request_text         TEXT,
                        result_summary       TEXT,
                        result_artifact_ref  TEXT,
                        error                TEXT,
                        created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            # Migration for tables created before conversation-history threading.
            await conn.execute(
                text(
                    "ALTER TABLE surface_sessions "
                    "ADD COLUMN IF NOT EXISTS request_text TEXT"
                )
            )
            # Migration for tables created before generic artifact references.
            await conn.execute(
                text(
                    "ALTER TABLE surface_sessions "
                    "ADD COLUMN IF NOT EXISTS result_artifact_ref TEXT"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS surface_sessions_status_idx "
                    "ON surface_sessions (status, updated_at DESC)"
                )
            )
            # event_id is the idempotency key for an inbound surface event: a
            # partial unique index makes a second, concurrent delivery of the same
            # event fail the insert (the runner catches it and defers to the
            # winner) rather than silently creating a duplicate session.
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS surface_sessions_event_uniq "
                    "ON surface_sessions (event_id) WHERE event_id IS NOT NULL"
                )
            )
            # GIN index backs the containment lookup (button → session).
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS surface_sessions_approval_idx "
                    "ON surface_sessions USING GIN (approval_ids)"
                )
            )
            # Backs the thread-reply lookup (is the bot part of this thread?).
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS surface_sessions_thread_idx "
                    "ON surface_sessions (channel, thread)"
                )
            )

    async def get(self, session_id: str) -> SurfaceSession | None:
        engine = self._require_engine()
        statement = select(surface_sessions).where(surface_sessions.c.id == session_id)
        async with engine.connect() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_session(row) if row else None

    async def get_by_event(self, event_id: str) -> SurfaceSession | None:
        if not event_id:
            return None
        engine = self._require_engine()
        statement = (
            select(surface_sessions)
            .where(surface_sessions.c.event_id == event_id)
            .order_by(surface_sessions.c.created_at.desc())
            .limit(1)
        )
        async with engine.connect() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_session(row) if row else None

    async def get_by_approval(self, approval_id: str) -> SurfaceSession | None:
        if not approval_id:
            return None
        engine = self._require_engine()
        # `@>` (JSONB containment) tests that the array holds this id. Unlike
        # the `?` operator it takes a normal bound parameter and is
        # GIN-indexable; `.contains` is how the expression language spells it.
        statement = (
            select(surface_sessions)
            .where(surface_sessions.c.approval_ids.contains([approval_id]))
            .order_by(surface_sessions.c.updated_at.desc())
            .limit(1)
        )
        async with engine.connect() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_session(row) if row else None

    async def get_by_thread(self, target: SurfaceTarget) -> SurfaceSession | None:
        if not target.thread:
            return None
        engine = self._require_engine()
        # Scope on the full target (not just channel/thread) so a shared store
        # never crosses agents/workspaces. (channel, thread) index drives it.
        statement = (
            select(surface_sessions)
            .where(*_target_predicates(target))
            .order_by(surface_sessions.c.updated_at.desc())
            .limit(1)
        )
        async with engine.connect() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_session(row) if row else None

    async def thread_history(
        self,
        target: SurfaceTarget,
        *,
        exclude_id: str | None = None,
        limit: int = 20,
    ) -> list[SurfaceSession]:
        if not target.thread:
            return []
        engine = self._require_engine()
        # Most-recent `limit` *delivered* exchanges in the thread, then flipped
        # to ascending so the caller replays them oldest-first. Scoped on the
        # full target (the (channel, thread) index drives the inner scan).
        # The status/final_message_id/text predicates mirror
        # store._is_replayable_turn: only turns the user actually saw, filtered
        # BEFORE the limit so a burst of failed/pending replies can't crowd
        # valid older exchanges out of the window. `exclude_id` drops the
        # in-flight session.
        inner = (
            select(surface_sessions)
            .where(
                *_target_predicates(target),
                surface_sessions.c.status == "completed",
                surface_sessions.c.final_message_id.is_not(None),
                surface_sessions.c.request_text.is_not(None),
                surface_sessions.c.result_summary.is_not(None),
            )
            .order_by(surface_sessions.c.created_at.desc())
            .limit(limit)
        )
        if exclude_id is not None:
            inner = inner.where(surface_sessions.c.id != exclude_id)
        recent = inner.subquery("recent")
        statement = select(recent).order_by(recent.c.created_at.asc())
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [_row_to_session(r) for r in rows]

    async def upsert(self, session: SurfaceSession) -> None:
        engine = self._require_engine()
        t = session.target
        # created_at is set once; updated_at always bumped to now().
        values = {
            "id": session.id,
            "surface": t.surface,
            "workspace": t.workspace,
            "agent": t.agent,
            "channel": t.channel,
            "thread": t.thread,
            "event_id": t.event_id,
            "status": session.status,
            "workflow_instance_id": session.workflow_instance_id,
            "progress_message_id": session.progress_message_id,
            "final_message_id": session.final_message_id,
            "approval_ids": session.approval_ids,
            "request_text": session.request_text,
            "result_summary": session.result_summary,
            "result_artifact_ref": session.result_artifact_ref,
            "error": session.error,
            "updated_at": func.now(),
        }
        statement = insert(surface_sessions).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[surface_sessions.c.id],
            set_={
                name: statement.excluded[name]
                for name in values
                if name not in ("id", "updated_at")
            }
            | {"updated_at": func.now()},
        )
        async with engine.begin() as conn:
            await conn.execute(statement)

    async def recent(self, limit: int = 100) -> list[SurfaceSession]:
        engine = self._require_engine()
        statement = (
            select(surface_sessions)
            .order_by(surface_sessions.c.updated_at.desc())
            .limit(limit)
        )
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [_row_to_session(r) for r in rows]


def _target_predicates(target: SurfaceTarget) -> tuple:
    """The full-target scope every thread lookup shares."""
    return (
        surface_sessions.c.surface == target.surface,
        surface_sessions.c.workspace == target.workspace,
        surface_sessions.c.agent == target.agent,
        surface_sessions.c.channel == target.channel,
        surface_sessions.c.thread == target.thread,
    )


def _row_to_session(row) -> SurfaceSession:
    now = datetime.now(UTC)
    return SurfaceSession(
        id=row["id"],
        target=SurfaceTarget(
            surface=row["surface"],
            workspace=row["workspace"],
            agent=row["agent"],
            channel=row["channel"],
            thread=row["thread"],
            event_id=row["event_id"],
        ),
        status=row["status"],
        workflow_instance_id=row["workflow_instance_id"],
        progress_message_id=row["progress_message_id"],
        final_message_id=row["final_message_id"],
        # JSONB arrives decoded — the dialect registers a jsonb codec.
        approval_ids=row["approval_ids"] or [],
        request_text=row["request_text"],
        result_summary=row["result_summary"],
        result_artifact_ref=row["result_artifact_ref"],
        error=row["error"],
        created_at=row["created_at"] or now,
        updated_at=row["updated_at"] or now,
    )
