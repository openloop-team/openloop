"""Postgres usage backend — the persistent audit trail and budget source."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from openloop.db import BorrowedEngineStore
from openloop.usage.schema import usage
from openloop.usage.store import UsageRecord


class PostgresUsageStore(BorrowedEngineStore):
    """Persists usage to a `usage` table; totals drive budget enforcement."""

    async def setup(self, engine: AsyncEngine) -> None:
        # sql-text: schema evolution moves to Alembic (ADR 0009); DDL is not
        # restated as metadata. The ALTER loop interpolates a column name and a
        # SQL type, never a value.
        async with self._setup_connection(engine) as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS usage (
                        id                BIGSERIAL PRIMARY KEY,
                        scope_key         TEXT NOT NULL,
                        workspace         TEXT NOT NULL,
                        agent             TEXT NOT NULL,
                        channel           TEXT,
                        surface           TEXT,
                        "user"            TEXT,
                        task_kind         TEXT,
                        idempotency_key   TEXT,
                        model             TEXT NOT NULL,
                        prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        cost_usd          DOUBLE PRECISION NOT NULL DEFAULT 0,
                        outcome           TEXT NOT NULL DEFAULT 'ok',
                        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
                        job_id            TEXT,
                        broker_job_id     TEXT,
                        broker_generation BIGINT,
                        approval_id       TEXT,
                        approver          TEXT,
                        session_id        TEXT
                    )
                    """
                )
            )
            await conn.execute(
                text("ALTER TABLE usage ADD COLUMN IF NOT EXISTS idempotency_key TEXT")
            )
            # Attribution envelope (finding 4): nullable, no backfill.
            for column, sql_type in (
                ("job_id", "TEXT"),
                ("broker_job_id", "TEXT"),
                ("broker_generation", "BIGINT"),
                ("approval_id", "TEXT"),
                ("approver", "TEXT"),
                ("session_id", "TEXT"),
            ):
                await conn.execute(
                    text(
                        f"ALTER TABLE usage ADD COLUMN IF NOT EXISTS "
                        f"{column} {sql_type}"
                    )
                )
            # broker_generation was briefly INTEGER in a pre-release dev build;
            # widen it to match the broker schema's BIGINT generations. Idempotent
            # (a no-op when the column is already BIGINT).
            await conn.execute(
                text("ALTER TABLE usage ALTER COLUMN broker_generation TYPE BIGINT")
            )
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS usage_idempotency_key_idx "
                    "ON usage (idempotency_key)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS usage_scope_time_idx "
                    "ON usage (scope_key, created_at DESC)"
                )
            )

    async def record(self, usage_record: UsageRecord) -> bool:
        engine = self._require_engine()
        statement = (
            insert(usage)
            .values(
                scope_key=usage_record.scope_key,
                workspace=usage_record.workspace,
                agent=usage_record.agent,
                channel=usage_record.channel,
                surface=usage_record.surface,
                user=usage_record.user,
                task_kind=usage_record.task_kind,
                idempotency_key=usage_record.idempotency_key,
                model=usage_record.model,
                prompt_tokens=usage_record.prompt_tokens,
                completion_tokens=usage_record.completion_tokens,
                cost_usd=usage_record.cost_usd,
                outcome=usage_record.outcome,
                created_at=usage_record.created_at,
                job_id=usage_record.job_id,
                broker_job_id=usage_record.broker_job_id,
                broker_generation=usage_record.broker_generation,
                approval_id=usage_record.approval_id,
                approver=usage_record.approver,
                session_id=usage_record.session_id,
            )
            .on_conflict_do_nothing(index_elements=[usage.c.idempotency_key])
        )
        async with engine.begin() as conn:
            result = await conn.execute(statement)
        # The driver's "INSERT 0 1" status string is gone; rowcount says the same.
        return result.rowcount == 1

    async def monthly_total(self, scope_key: str, now: datetime | None = None) -> float:
        engine = self._require_engine()
        # date_trunc keeps "current month" defined by the database clock.
        statement = select(func.coalesce(func.sum(usage.c.cost_usd), 0)).where(
            usage.c.scope_key == scope_key,
            usage.c.created_at >= func.date_trunc("month", func.now()),
        )
        async with engine.connect() as conn:
            value = await conn.scalar(statement)
        return float(value or 0.0)

    async def recent(self, limit: int = 50) -> list[UsageRecord]:
        engine = self._require_engine()
        statement = (
            select(*(c for c in usage.c if c.name != "id"))
            .order_by(usage.c.created_at.desc())
            .limit(limit)
        )
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [
            UsageRecord(
                scope_key=r["scope_key"],
                workspace=r["workspace"],
                agent=r["agent"],
                model=r["model"],
                channel=r["channel"],
                surface=r["surface"],
                user=r["user"],
                task_kind=r["task_kind"],
                idempotency_key=r["idempotency_key"],
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                cost_usd=r["cost_usd"],
                outcome=r["outcome"],
                created_at=r["created_at"],
                job_id=r["job_id"],
                broker_job_id=r["broker_job_id"],
                broker_generation=r["broker_generation"],
                approval_id=r["approval_id"],
                approver=r["approver"],
                session_id=r["session_id"],
            )
            for r in rows
        ]
