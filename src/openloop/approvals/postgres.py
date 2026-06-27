"""Postgres-backed approval store — held approvals survive restarts.

Human approvals are paced by people, not the runtime, so they must outlive a
process restart. This mirrors :class:`InMemoryApprovalStore` against an
`approvals` table.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openloop.approvals.store import ApprovalRequest


class PostgresApprovalStore:
    """pgvector image not required — plain Postgres is enough for approvals."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None  # asyncpg.Pool, created in setup()

    async def setup(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    id           TEXT PRIMARY KEY,
                    agent        TEXT NOT NULL,
                    action       TEXT NOT NULL,
                    tool         TEXT NOT NULL,
                    permission   TEXT NOT NULL,
                    args         JSONB NOT NULL DEFAULT '{}',
                    approvers    JSONB NOT NULL DEFAULT '[]',
                    summary      TEXT NOT NULL DEFAULT '',
                    requested_by TEXT,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    decided_by   TEXT,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS approvals_status_idx "
                "ON approvals (status, agent)"
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError("PostgresApprovalStore.setup() must be called first")
        return self._pool

    async def create(self, request: ApprovalRequest) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO approvals (
                    id, agent, action, tool, permission, args, approvers,
                    summary, requested_by, status, decided_by, created_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                """,
                request.id,
                request.agent,
                request.action,
                request.tool,
                request.permission,
                json.dumps(request.args),
                json.dumps(request.approvers),
                request.summary,
                request.requested_by,
                request.status,
                request.decided_by,
                request.created_at,
            )

    async def get(self, request_id: str) -> ApprovalRequest | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM approvals WHERE id = $1", request_id
            )
        return _row_to_request(row) if row else None

    async def pending(self, agent: str | None = None) -> list[ApprovalRequest]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if agent is None:
                rows = await conn.fetch(
                    "SELECT * FROM approvals WHERE status = 'pending' "
                    "ORDER BY created_at"
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM approvals WHERE status = 'pending' "
                    "AND agent = $1 ORDER BY created_at",
                    agent,
                )
        return [_row_to_request(r) for r in rows]

    async def update(self, request: ApprovalRequest) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE approvals SET status = $2, decided_by = $3 WHERE id = $1",
                request.id,
                request.status,
                request.decided_by,
            )


def _row_to_request(row) -> ApprovalRequest:
    return ApprovalRequest(
        id=row["id"],
        agent=row["agent"],
        action=row["action"],
        tool=row["tool"],
        permission=row["permission"],
        args=json.loads(row["args"]) if row["args"] else {},
        approvers=json.loads(row["approvers"]) if row["approvers"] else [],
        summary=row["summary"],
        requested_by=row["requested_by"],
        status=row["status"],
        decided_by=row["decided_by"],
        created_at=row["created_at"] or datetime.now(timezone.utc),
    )
