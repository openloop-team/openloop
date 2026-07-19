"""Postgres usage backend — the persistent audit trail and budget source."""

from __future__ import annotations

from datetime import datetime

from openloop.postgres import BorrowedPostgresStore
from openloop.usage.store import UsageRecord


class PostgresUsageStore(BorrowedPostgresStore):
    """Persists usage to a `usage` table; totals drive budget enforcement."""

    async def setup(self, pool) -> None:
        async with self._setup_connection(pool) as conn:
            await conn.execute(
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
            await conn.execute(
                "ALTER TABLE usage ADD COLUMN IF NOT EXISTS idempotency_key TEXT"
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
                    f"ALTER TABLE usage ADD COLUMN IF NOT EXISTS {column} {sql_type}"
                )
            # broker_generation was briefly INTEGER in a pre-release dev build;
            # widen it to match the broker schema's BIGINT generations. Idempotent
            # (a no-op when the column is already BIGINT).
            await conn.execute(
                "ALTER TABLE usage ALTER COLUMN broker_generation TYPE BIGINT"
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS usage_idempotency_key_idx "
                "ON usage (idempotency_key)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS usage_scope_time_idx "
                "ON usage (scope_key, created_at DESC)"
            )

    async def record(self, usage: UsageRecord) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            status = await conn.execute(
                """
                INSERT INTO usage (
                    scope_key, workspace, agent, channel, surface, "user",
                    task_kind, idempotency_key, model, prompt_tokens,
                    completion_tokens, cost_usd, outcome, created_at,
                    job_id, broker_job_id, broker_generation, approval_id,
                    approver, session_id
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                        $15,$16,$17,$18,$19,$20)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                usage.scope_key,
                usage.workspace,
                usage.agent,
                usage.channel,
                usage.surface,
                usage.user,
                usage.task_kind,
                usage.idempotency_key,
                usage.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cost_usd,
                usage.outcome,
                usage.created_at,
                usage.job_id,
                usage.broker_job_id,
                usage.broker_generation,
                usage.approval_id,
                usage.approver,
                usage.session_id,
            )
        return status.endswith("1")

    async def monthly_total(
        self, scope_key: str, now: datetime | None = None
    ) -> float:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # date_trunc keeps "current month" defined by the database clock.
            value = await conn.fetchval(
                """
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM usage
                WHERE scope_key = $1
                  AND created_at >= date_trunc('month', now())
                """,
                scope_key,
            )
        return float(value or 0.0)

    async def recent(self, limit: int = 50) -> list[UsageRecord]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT scope_key, workspace, agent, channel, surface, "user",
                       task_kind, idempotency_key, model, prompt_tokens, completion_tokens,
                       cost_usd, outcome, created_at,
                       job_id, broker_job_id, broker_generation, approval_id,
                       approver, session_id
                FROM usage
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
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
