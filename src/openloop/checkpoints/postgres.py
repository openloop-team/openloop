"""Postgres-backed worker checkpoints — jobs survive a process restart.

Mirrors :class:`InMemoryCheckpointStore` against a ``worker_checkpoints`` table,
following the same pattern as the approvals/usage stores. Plain Postgres is
enough; no pgvector needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openloop.checkpoints.store import WorkerCheckpoint
from openloop.postgres import BorrowedPostgresStore


class PostgresCheckpointStore(BorrowedPostgresStore):
    async def setup(self, pool) -> None:
        async with self._setup_connection(pool) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_checkpoints (
                    job_id          TEXT PRIMARY KEY,
                    repo            TEXT NOT NULL,
                    instruction     TEXT NOT NULL,
                    base            TEXT NOT NULL,
                    branch          TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    completed_steps JSONB NOT NULL DEFAULT '[]',
                    state_json      JSONB NOT NULL DEFAULT '{}',
                    title           TEXT,
                    body            TEXT,
                    pr_number       INTEGER,
                    pr_url          TEXT,
                    error           TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS worker_checkpoints_updated_idx "
                "ON worker_checkpoints (updated_at DESC)"
            )

    async def get(self, job_id: str) -> WorkerCheckpoint | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM worker_checkpoints WHERE job_id = $1", job_id
            )
        return _row_to_checkpoint(row) if row else None

    async def upsert(self, checkpoint: WorkerCheckpoint) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # created_at is set once; updated_at always bumped to now().
            await conn.execute(
                """
                INSERT INTO worker_checkpoints (
                    job_id, repo, instruction, base, branch, status,
                    completed_steps, state_json, title, body, pr_number,
                    pr_url, error, updated_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13, now())
                ON CONFLICT (job_id) DO UPDATE SET
                    repo = EXCLUDED.repo,
                    instruction = EXCLUDED.instruction,
                    base = EXCLUDED.base,
                    branch = EXCLUDED.branch,
                    status = EXCLUDED.status,
                    completed_steps = EXCLUDED.completed_steps,
                    state_json = EXCLUDED.state_json,
                    title = EXCLUDED.title,
                    body = EXCLUDED.body,
                    pr_number = EXCLUDED.pr_number,
                    pr_url = EXCLUDED.pr_url,
                    error = EXCLUDED.error,
                    updated_at = now()
                """,
                checkpoint.job_id,
                checkpoint.repo,
                checkpoint.instruction,
                checkpoint.base,
                checkpoint.branch,
                checkpoint.status,
                json.dumps(checkpoint.completed_steps),
                json.dumps(checkpoint.state_json),
                checkpoint.title,
                checkpoint.body,
                checkpoint.pr_number,
                checkpoint.pr_url,
                checkpoint.error,
            )

    async def recent(self, limit: int = 50) -> list[WorkerCheckpoint]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM worker_checkpoints ORDER BY updated_at DESC LIMIT $1",
                limit,
            )
        return [_row_to_checkpoint(r) for r in rows]


def _row_to_checkpoint(row) -> WorkerCheckpoint:
    now = datetime.now(timezone.utc)
    return WorkerCheckpoint(
        job_id=row["job_id"],
        repo=row["repo"],
        instruction=row["instruction"],
        base=row["base"],
        branch=row["branch"],
        status=row["status"],
        completed_steps=json.loads(row["completed_steps"]) if row["completed_steps"] else [],
        state_json=json.loads(row["state_json"]) if row["state_json"] else {},
        title=row["title"],
        body=row["body"],
        pr_number=row["pr_number"],
        pr_url=row["pr_url"],
        error=row["error"],
        created_at=row["created_at"] or now,
        updated_at=row["updated_at"] or now,
    )
