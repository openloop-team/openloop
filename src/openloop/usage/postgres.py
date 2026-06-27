"""Postgres usage backend — the persistent audit trail and budget source."""

from __future__ import annotations

from datetime import datetime

from openloop.usage.store import UsageRecord


class PostgresUsageStore:
    """Persists usage to a `usage` table; totals drive budget enforcement."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None  # asyncpg.Pool, created in setup()

    async def setup(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn)
        async with self._pool.acquire() as conn:
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
                    model             TEXT NOT NULL,
                    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd          DOUBLE PRECISION NOT NULL DEFAULT 0,
                    outcome           TEXT NOT NULL DEFAULT 'ok',
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS usage_scope_time_idx "
                "ON usage (scope_key, created_at DESC)"
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError("PostgresUsageStore.setup() must be called first")
        return self._pool

    async def record(self, usage: UsageRecord) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO usage (
                    scope_key, workspace, agent, channel, surface, "user",
                    task_kind, model, prompt_tokens, completion_tokens,
                    cost_usd, outcome, created_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                """,
                usage.scope_key,
                usage.workspace,
                usage.agent,
                usage.channel,
                usage.surface,
                usage.user,
                usage.task_kind,
                usage.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cost_usd,
                usage.outcome,
                usage.created_at,
            )

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
                       task_kind, model, prompt_tokens, completion_tokens,
                       cost_usd, outcome, created_at
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
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                cost_usd=r["cost_usd"],
                outcome=r["outcome"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
