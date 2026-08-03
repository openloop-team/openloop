"""Postgres-backed thread↔task bindings — a task outlives the process.

Mirrors :class:`~openloop.tasks.binding.InMemoryThreadTaskStore` against a
``workspace_thread_tasks`` table, following the sessions/threads/workflow store
pattern. The claim transitions are single-statement conditional UPDATEs so two
replicas can never drive one task at once, and the serialized task rides in a
JSONB column so a replica that has never seen the thread can reconstruct it
cold.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openloop.postgres import BorrowedPostgresStore
from openloop.tasks.binding import BUSY, CLOSED, OPEN, ThreadTask


class PostgresThreadTaskStore(BorrowedPostgresStore):
    async def setup(self, pool) -> None:
        async with self._setup_connection(pool) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_thread_tasks (
                    task_id       TEXT PRIMARY KEY,
                    scope_key     TEXT NOT NULL,
                    profile       TEXT NOT NULL,
                    entry_action  TEXT NOT NULL,
                    status        TEXT NOT NULL,
                    agent         TEXT,
                    agent_id      TEXT,
                    approval_id   TEXT,
                    requested_by  TEXT,
                    instance_id   TEXT,
                    session_id    TEXT,
                    turns         INTEGER NOT NULL DEFAULT 1,
                    state         JSONB NOT NULL DEFAULT '{}',
                    closed_reason TEXT,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # Drives the thread → open-binding lookup on every inbound reply.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS workspace_thread_tasks_scope_idx "
                "ON workspace_thread_tasks (scope_key, created_at DESC)"
            )
            # Drives the startup orphan-claim sweep.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS workspace_thread_tasks_status_idx "
                "ON workspace_thread_tasks (status, updated_at)"
            )

    async def get(self, task_id: str) -> ThreadTask | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workspace_thread_tasks WHERE task_id = $1", task_id
            )
        return _row_to_record(row) if row else None

    async def bound(self, scope_key: str) -> ThreadTask | None:
        if not scope_key:
            return None
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workspace_thread_tasks WHERE scope_key = $1 "
                "AND status <> $2 ORDER BY created_at DESC LIMIT 1",
                scope_key,
                CLOSED,
            )
        return _row_to_record(row) if row else None

    async def bind(self, record: ThreadTask) -> ThreadTask:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Write-once birth record: a re-invoke of the same task never
            # re-opens a closed row or re-points it at another thread.
            await conn.execute(
                """
                INSERT INTO workspace_thread_tasks (
                    task_id, scope_key, profile, entry_action, status, agent,
                    agent_id, approval_id, requested_by, instance_id,
                    session_id, turns, state, closed_reason
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                ON CONFLICT (task_id) DO NOTHING
                """,
                record.task_id,
                record.scope_key,
                record.profile,
                record.entry_action,
                record.status,
                record.agent,
                record.agent_id,
                record.approval_id,
                record.requested_by,
                record.instance_id,
                record.session_id,
                record.turns,
                json.dumps(record.state or {}),
                record.closed_reason,
            )
            row = await conn.fetchrow(
                "SELECT * FROM workspace_thread_tasks WHERE task_id = $1",
                record.task_id,
            )
        return _row_to_record(row) if row else record

    async def claim(
        self, task_id: str, *, instance_id: str, session_id: str | None = None
    ) -> ThreadTask | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Atomic CAS: only an `open` row can be claimed, so a second
            # replica reaching here for the same reply loses cleanly.
            row = await conn.fetchrow(
                """
                UPDATE workspace_thread_tasks
                SET status = $2, instance_id = $3, session_id = $4,
                    turns = turns + 1, updated_at = now()
                WHERE task_id = $1 AND status = $5
                RETURNING *
                """,
                task_id,
                BUSY,
                instance_id,
                session_id,
                OPEN,
            )
        return _row_to_record(row) if row else None

    async def release(
        self,
        task_id: str,
        *,
        state: dict | None = None,
        status: str = OPEN,
        reason: str | None = None,
    ) -> ThreadTask | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # A closed row never re-opens; the state refresh still lands so the
            # durable task stays a faithful record of what actually ran.
            row = await conn.fetchrow(
                """
                UPDATE workspace_thread_tasks
                SET status = CASE WHEN status = $4 THEN status ELSE $2 END,
                    closed_reason = CASE
                        WHEN status = $4 THEN closed_reason
                        WHEN $2 = $4 THEN $3
                        ELSE NULL
                    END,
                    state = COALESCE($5::jsonb, state),
                    updated_at = now()
                WHERE task_id = $1
                RETURNING *
                """,
                task_id,
                status,
                reason,
                CLOSED,
                json.dumps(state) if state is not None else None,
            )
        return _row_to_record(row) if row else None

    async def retire(self, task_id: str, reason: str) -> ThreadTask | None:
        # NOT ``close``: BorrowedPostgresStore.close() detaches the pool.
        return await self.release(task_id, status=CLOSED, reason=reason)

    async def claimed(self, limit: int = 200) -> list[ThreadTask]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM workspace_thread_tasks WHERE status = $1 "
                "ORDER BY updated_at LIMIT $2",
                BUSY,
                limit,
            )
        return [_row_to_record(r) for r in rows]


def _row_to_record(row) -> ThreadTask:
    now = datetime.now(timezone.utc)
    raw_state = row["state"]
    return ThreadTask(
        task_id=row["task_id"],
        scope_key=row["scope_key"],
        profile=row["profile"],
        entry_action=row["entry_action"],
        status=row["status"],
        agent=row["agent"],
        agent_id=row["agent_id"],
        approval_id=row["approval_id"],
        requested_by=row["requested_by"],
        instance_id=row["instance_id"],
        session_id=row["session_id"],
        turns=row["turns"] or 1,
        state=json.loads(raw_state) if isinstance(raw_state, str) else (raw_state or {}),
        closed_reason=row["closed_reason"],
        created_at=row["created_at"] or now,
        updated_at=row["updated_at"] or now,
    )
