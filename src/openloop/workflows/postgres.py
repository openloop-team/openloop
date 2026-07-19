"""Postgres-backed workflow instances — workflows survive a process restart.

Mirrors :class:`InMemoryWorkflowStore` against a ``workflow_instances`` table,
following the approvals/usage/checkpoint store pattern. All arbitration
predicates run server-side (``now()``), so replicas need no clock agreement;
``drive_gen`` and ``leased_until`` are store-owned and never written from the
instance payload.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openloop.postgres import BorrowedPostgresStore
from openloop.workflows.store import TERMINAL, WorkflowInstance


class PostgresWorkflowStore(BorrowedPostgresStore):
    async def setup(self, pool) -> None:
        async with self._setup_connection(pool) as conn:
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
                    leased_until    TIMESTAMPTZ,
                    drive_gen       INTEGER NOT NULL DEFAULT 0,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "ALTER TABLE workflow_instances "
                "ADD COLUMN IF NOT EXISTS leased_until TIMESTAMPTZ"
            )
            await conn.execute(
                "ALTER TABLE workflow_instances "
                "ADD COLUMN IF NOT EXISTS drive_gen INTEGER NOT NULL DEFAULT 0"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS workflow_instances_status_idx "
                "ON workflow_instances (status, updated_at DESC)"
            )

    async def get(self, instance_id: str) -> WorkflowInstance | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workflow_instances WHERE id = $1", instance_id
            )
        return _row_to_instance(row) if row else None

    async def create(self, instance: WorkflowInstance) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO workflow_instances (
                    id, workflow, status, completed_steps, state, waiting_on,
                    result, error, leased_until, drive_gen
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                instance.id,
                instance.workflow,
                instance.status,
                json.dumps(instance.completed_steps),
                json.dumps(instance.state),
                instance.waiting_on,
                json.dumps(instance.result) if instance.result is not None else None,
                instance.error,
                instance.leased_until,
                instance.drive_gen,
            )
        return row is not None

    async def claim_drive(
        self, instance_id: str, *, lease_seconds: float
    ) -> WorkflowInstance | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE workflow_instances
                SET drive_gen = drive_gen + 1,
                    leased_until = now() + make_interval(secs => $2),
                    updated_at = now()
                WHERE id = $1
                  AND status = 'running'
                  AND (leased_until IS NULL OR leased_until < now())
                RETURNING *
                """,
                instance_id,
                lease_seconds,
            )
        return _row_to_instance(row) if row else None

    async def fenced_write(
        self, instance: WorkflowInstance, gen: int, *, release: bool = False
    ) -> bool:
        pool = self._require_pool()
        # drive_gen and leased_until are store-owned: the ordinary form leaves
        # both untouched (a payload write would roll back the ticker's renewed
        # lease); release bumps the gen and clears the lease.
        ownership = (
            "drive_gen = drive_gen + 1, leased_until = NULL,"
            if release
            else ""
        )
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE workflow_instances
                SET workflow = $3,
                    status = $4,
                    completed_steps = $5,
                    state = $6,
                    waiting_on = $7,
                    result = $8,
                    error = $9,
                    {ownership}
                    updated_at = now()
                WHERE id = $1 AND drive_gen = $2
                RETURNING id
                """,
                instance.id,
                gen,
                instance.workflow,
                instance.status,
                json.dumps(instance.completed_steps),
                json.dumps(instance.state),
                instance.waiting_on,
                json.dumps(instance.result) if instance.result is not None else None,
                instance.error,
            )
        return row is not None

    async def renew_lease(
        self, instance_id: str, gen: int, *, lease_seconds: float
    ) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE workflow_instances
                SET leased_until = now() + make_interval(secs => $3),
                    updated_at = now()
                WHERE id = $1 AND drive_gen = $2
                RETURNING id
                """,
                instance_id,
                gen,
                lease_seconds,
            )
        return row is not None

    async def cancel_instance(
        self, instance_id: str, reason: str
    ) -> WorkflowInstance | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE workflow_instances
                SET status = 'cancelled',
                    error = NULLIF($2, ''),
                    waiting_on = NULL,
                    leased_until = NULL,
                    drive_gen = drive_gen + 1,
                    updated_at = now()
                WHERE id = $1 AND status != ALL($3::text[])
                RETURNING *
                """,
                instance_id,
                reason,
                list(TERMINAL),
            )
        return _row_to_instance(row) if row else None

    async def claim_event(
        self, instance_id: str, event: str, payload: dict
    ) -> WorkflowInstance | None:
        """Atomically consume one exact wait event across all replicas."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE workflow_instances
                SET state = state || jsonb_build_object(
                        'events',
                        COALESCE(state->'events', '{}'::jsonb)
                        || jsonb_build_object($2::text, $3::jsonb)
                    ),
                    completed_steps = completed_steps || jsonb_build_array($2::text),
                    status = 'running',
                    waiting_on = NULL,
                    leased_until = NULL,
                    updated_at = now()
                WHERE id = $1
                  AND status = 'waiting'
                  AND waiting_on = $2
                RETURNING *
                """,
                instance_id,
                event,
                json.dumps(payload),
            )
        return _row_to_instance(row) if row else None

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
        leased_until=row["leased_until"],
        drive_gen=row["drive_gen"],
        created_at=row["created_at"] or now,
        updated_at=row["updated_at"] or now,
    )
