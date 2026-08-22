"""Postgres-backed worker checkpoints — jobs survive a process restart.

Mirrors :class:`InMemoryCheckpointStore` against a ``worker_checkpoints`` table,
following the same pattern as the approvals/usage stores. Plain Postgres is
enough; no pgvector needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from openloop.checkpoints.schema import worker_checkpoints
from openloop.checkpoints.store import WorkerCheckpoint
from openloop.db import BorrowedEngineStore


class PostgresCheckpointStore(BorrowedEngineStore):
    async def setup(self, engine: AsyncEngine) -> None:
        async with self._setup_connection(engine) as conn:
            # sql-text: schema evolution moves to a migration tool; DDL is not
            # restated as metadata.
            await conn.execute(
                text(
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
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS worker_checkpoints_updated_idx "
                    "ON worker_checkpoints (updated_at DESC)"
                )
            )

    async def get(self, job_id: str) -> WorkerCheckpoint | None:
        engine = self._require_engine()
        statement = select(worker_checkpoints).where(
            worker_checkpoints.c.job_id == job_id
        )
        async with engine.connect() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_checkpoint(row) if row else None

    async def upsert(self, checkpoint: WorkerCheckpoint) -> None:
        engine = self._require_engine()
        # created_at is set once; updated_at always bumped to now().
        values = {
            "job_id": checkpoint.job_id,
            "repo": checkpoint.repo,
            "instruction": checkpoint.instruction,
            "base": checkpoint.base,
            "branch": checkpoint.branch,
            "status": checkpoint.status,
            "completed_steps": checkpoint.completed_steps,
            "state_json": checkpoint.state_json,
            "title": checkpoint.title,
            "body": checkpoint.body,
            "pr_number": checkpoint.pr_number,
            "pr_url": checkpoint.pr_url,
            "error": checkpoint.error,
            "updated_at": func.now(),
        }
        statement = insert(worker_checkpoints).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[worker_checkpoints.c.job_id],
            set_={
                name: statement.excluded[name]
                for name in values
                if name not in ("job_id", "updated_at")
            }
            | {"updated_at": func.now()},
        )
        async with engine.begin() as conn:
            await conn.execute(statement)

    async def recent(self, limit: int = 50) -> list[WorkerCheckpoint]:
        engine = self._require_engine()
        statement = (
            select(worker_checkpoints)
            .order_by(worker_checkpoints.c.updated_at.desc())
            .limit(limit)
        )
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [_row_to_checkpoint(r) for r in rows]


def _row_to_checkpoint(row) -> WorkerCheckpoint:
    now = datetime.now(UTC)
    return WorkerCheckpoint(
        job_id=row["job_id"],
        repo=row["repo"],
        instruction=row["instruction"],
        base=row["base"],
        branch=row["branch"],
        status=row["status"],
        # JSONB arrives decoded — the dialect registers a jsonb codec.
        completed_steps=row["completed_steps"] or [],
        state_json=row["state_json"] or {},
        title=row["title"],
        body=row["body"],
        pr_number=row["pr_number"],
        pr_url=row["pr_url"],
        error=row["error"],
        created_at=row["created_at"] or now,
        updated_at=row["updated_at"] or now,
    )
