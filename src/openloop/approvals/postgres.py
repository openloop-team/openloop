"""Postgres-backed approval store — held approvals survive restarts.

Human approvals are paced by people, not the runtime, so they must outlive a
process restart. This mirrors :class:`InMemoryApprovalStore` against an
`approvals` table; ``claim_decision``'s pending guard and ``mark_reconciled``'s
``now()`` are enforced server-side so the row stays the single arbiter across
replicas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openloop.approvals.store import ApprovalRequest
from openloop.postgres import BorrowedPostgresStore


class PostgresApprovalStore(BorrowedPostgresStore):
    """pgvector image not required — plain Postgres is enough for approvals."""

    async def setup(self, pool) -> None:
        async with self._setup_connection(pool) as conn:
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
            # Migration for rows created before args-contract versioning
            # (same idiom as surface_threads' later columns). NULL doubles as
            # the pre-version sentinel version-checking consumers refuse.
            await conn.execute(
                "ALTER TABLE approvals "
                "ADD COLUMN IF NOT EXISTS args_schema INTEGER"
            )
            # Decide-once columns (same idiom). NULL workflow_backed marks a
            # legacy row the resolver classifies by registry; NULL effect_at
            # keeps a decided row in the reconciler's sweep.
            await conn.execute(
                "ALTER TABLE approvals "
                "ADD COLUMN IF NOT EXISTS workflow_backed BOOLEAN"
            )
            await conn.execute(
                "ALTER TABLE approvals "
                "ADD COLUMN IF NOT EXISTS workflow_instance_id TEXT"
            )
            await conn.execute(
                "ALTER TABLE approvals "
                "ADD COLUMN IF NOT EXISTS effect_at TIMESTAMPTZ"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS approvals_status_idx "
                "ON approvals (status, agent)"
            )
            # The decision reconciler's sweep (decided_unreconciled) filters
            # `status != 'pending' AND effect_at IS NULL` in (created_at, id)
            # order. A partial index on exactly that predicate keeps the sweep
            # O(unreconciled) instead of a full scan that grows with the whole
            # approvals history; the (created_at, id) key also serves the keyset
            # cursor and the ORDER BY.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS approvals_unreconciled_idx "
                "ON approvals (created_at, id) "
                "WHERE effect_at IS NULL AND status <> 'pending'"
            )

    async def create(self, request: ApprovalRequest) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO approvals (
                    id, agent, action, tool, permission, args, approvers,
                    summary, requested_by, status, decided_by, args_schema,
                    workflow_backed, workflow_instance_id, effect_at,
                    created_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
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
                request.args_schema,
                request.workflow_backed,
                request.workflow_instance_id,
                request.effect_at,
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

    async def claim_decision(
        self, request_id: str, approver: str, *, approve: bool
    ) -> ApprovalRequest | None:
        pool = self._require_pool()
        status = "approved" if approve else "denied"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE approvals SET status = $2, decided_by = $3 "
                "WHERE id = $1 AND status = 'pending' RETURNING *",
                request_id,
                status,
                approver,
            )
        return _row_to_request(row) if row else None

    async def decided_unreconciled(
        self,
        limit: int = 200,
        after: tuple[datetime, str] | None = None,
    ) -> list[ApprovalRequest]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if after is None:
                rows = await conn.fetch(
                    "SELECT * FROM approvals "
                    "WHERE status != 'pending' AND effect_at IS NULL "
                    "ORDER BY created_at ASC, id ASC LIMIT $1",
                    limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM approvals "
                    "WHERE status != 'pending' AND effect_at IS NULL "
                    "AND (created_at, id) > ($2, $3) "
                    "ORDER BY created_at ASC, id ASC LIMIT $1",
                    limit,
                    after[0],
                    after[1],
                )
        return [_row_to_request(r) for r in rows]

    async def mark_reconciled(self, request_id: str) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE approvals SET effect_at = now() "
                "WHERE id = $1 AND effect_at IS NULL",
                request_id,
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
        args_schema=row["args_schema"],
        workflow_backed=row["workflow_backed"],
        workflow_instance_id=row["workflow_instance_id"],
        effect_at=row["effect_at"],
        created_at=row["created_at"] or datetime.now(timezone.utc),
    )
