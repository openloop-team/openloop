"""Postgres-backed workflow instances — workflows survive a process restart.

Mirrors :class:`InMemoryWorkflowStore` against a ``workflow_instances`` table,
following the approvals/usage/checkpoint store pattern.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openloop.workflows.store import WorkflowInstance


class PostgresWorkflowStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None  # asyncpg.Pool, created in setup()

    async def setup(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_instances (
                    id              TEXT PRIMARY KEY,
                    workflow        TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    completed_steps JSONB NOT NULL DEFAULT '[]',
                    state           JSONB NOT NULL DEFAULT '{}',
                    waiting_on      TEXT,
                    result          JSONB,
                    error           TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS workflow_instances_status_idx "
                "ON workflow_instances (status, updated_at DESC)"
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError("PostgresWorkflowStore.setup() must be called first")
        return self._pool

    async def get(self, instance_id: str) -> WorkflowInstance | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workflow_instances WHERE id = $1", instance_id
            )
        return _row_to_instance(row) if row else None

    async def upsert(self, instance: WorkflowInstance) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO workflow_instances (
                    id, workflow, status, completed_steps, state, waiting_on,
                    result, error, updated_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8, now())
                ON CONFLICT (id) DO UPDATE SET
                    workflow = EXCLUDED.workflow,
                    status = EXCLUDED.status,
                    completed_steps = EXCLUDED.completed_steps,
                    state = EXCLUDED.state,
                    waiting_on = EXCLUDED.waiting_on,
                    result = EXCLUDED.result,
                    error = EXCLUDED.error,
                    updated_at = now()
                """,
                instance.id,
                instance.workflow,
                instance.status,
                json.dumps(instance.completed_steps),
                json.dumps(instance.state),
                instance.waiting_on,
                json.dumps(instance.result) if instance.result is not None else None,
                instance.error,
            )

    async def recent(self, limit: int = 100) -> list[WorkflowInstance]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM workflow_instances ORDER BY updated_at DESC LIMIT $1",
                limit,
            )
        return [_row_to_instance(r) for r in rows]


def _row_to_instance(row) -> WorkflowInstance:
    now = datetime.now(timezone.utc)
    return WorkflowInstance(
        id=row["id"],
        workflow=row["workflow"],
        status=row["status"],
        completed_steps=json.loads(row["completed_steps"]) if row["completed_steps"] else [],
        state=json.loads(row["state"]) if row["state"] else {},
        waiting_on=row["waiting_on"],
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        created_at=row["created_at"] or now,
        updated_at=row["updated_at"] or now,
    )
