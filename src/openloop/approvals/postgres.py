"""Postgres-backed approval store — held approvals survive restarts.

Human approvals are paced by people, not the runtime, so they must outlive a
process restart. This mirrors :class:`InMemoryApprovalStore` against an
`approvals` table; ``claim_decision``'s pending guard and ``mark_reconciled``'s
``now()`` are enforced server-side so the row stays the single arbiter across
replicas.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, literal, select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncEngine

from openloop.approvals.schema import approvals
from openloop.approvals.store import ApprovalRequest
from openloop.db import BorrowedEngineStore


class PostgresApprovalStore(BorrowedEngineStore):
    """pgvector image not required — plain Postgres is enough for approvals."""

    async def setup(self, engine: AsyncEngine) -> None:
        # sql-text: schema evolution moves to Alembic (ADR 0009); DDL is not
        # restated as metadata.
        async with self._setup_connection(engine) as conn:
            await conn.execute(
                text(
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
            )
            # Migration for rows created before args-contract versioning
            # (same idiom as surface_threads' later columns). NULL doubles as
            # the pre-version sentinel version-checking consumers refuse.
            await conn.execute(
                text(
                    "ALTER TABLE approvals ADD COLUMN IF NOT EXISTS args_schema INTEGER"
                )
            )
            # Decide-once columns (same idiom). NULL workflow_backed marks a
            # legacy row the resolver classifies by registry; NULL effect_at
            # keeps a decided row in the reconciler's sweep.
            await conn.execute(
                text(
                    "ALTER TABLE approvals "
                    "ADD COLUMN IF NOT EXISTS workflow_backed BOOLEAN"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE approvals "
                    "ADD COLUMN IF NOT EXISTS workflow_instance_id TEXT"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE approvals ADD COLUMN IF NOT EXISTS "
                    "effect_at TIMESTAMPTZ"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS approvals_status_idx "
                    "ON approvals (status, agent)"
                )
            )
            # The decision reconciler's sweep (decided_unreconciled) filters
            # `status != 'pending' AND effect_at IS NULL` in (created_at, id)
            # order. A partial index on exactly that predicate keeps the sweep
            # O(unreconciled) instead of a full scan that grows with the whole
            # approvals history; the (created_at, id) key also serves the keyset
            # cursor and the ORDER BY.
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS approvals_unreconciled_idx "
                    "ON approvals (created_at, id) "
                    "WHERE effect_at IS NULL AND status <> 'pending'"
                )
            )

    async def create(self, request: ApprovalRequest) -> None:
        engine = self._require_engine()
        statement = approvals.insert().values(
            id=request.id,
            agent=request.agent,
            action=request.action,
            tool=request.tool,
            permission=request.permission,
            args=request.args,
            approvers=request.approvers,
            summary=request.summary,
            requested_by=request.requested_by,
            status=request.status,
            decided_by=request.decided_by,
            args_schema=request.args_schema,
            workflow_backed=request.workflow_backed,
            workflow_instance_id=request.workflow_instance_id,
            effect_at=request.effect_at,
            created_at=request.created_at,
        )
        async with engine.begin() as conn:
            await conn.execute(statement)

    async def get(self, request_id: str) -> ApprovalRequest | None:
        engine = self._require_engine()
        statement = select(approvals).where(approvals.c.id == request_id)
        async with engine.connect() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_request(row) if row else None

    async def pending(self, agent: str | None = None) -> list[ApprovalRequest]:
        engine = self._require_engine()
        statement = (
            select(approvals)
            .where(approvals.c.status == "pending")
            .order_by(approvals.c.created_at)
        )
        if agent is not None:
            statement = statement.where(approvals.c.agent == agent)
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [_row_to_request(r) for r in rows]

    async def claim_decision(
        self, request_id: str, approver: str, *, approve: bool
    ) -> ApprovalRequest | None:
        engine = self._require_engine()
        # One conditional UPDATE: the `status = 'pending'` predicate is what
        # makes the claim single-winner, and it stays in the writing statement.
        statement = (
            approvals.update()
            .where(approvals.c.id == request_id, approvals.c.status == "pending")
            .values(status="approved" if approve else "denied", decided_by=approver)
            .returning(*approvals.c)
        )
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_request(row) if row else None

    async def decided_unreconciled(
        self,
        limit: int = 200,
        after: tuple[datetime, str] | None = None,
    ) -> list[ApprovalRequest]:
        engine = self._require_engine()
        statement = (
            select(approvals)
            .where(approvals.c.status != "pending", approvals.c.effect_at.is_(None))
            .order_by(approvals.c.created_at.asc(), approvals.c.id.asc())
            .limit(limit)
        )
        if after is not None:
            # Row-value comparison — the keyset cursor the partial index serves.
            statement = statement.where(
                tuple_(approvals.c.created_at, approvals.c.id)
                > tuple_(literal(after[0]), literal(after[1]))
            )
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [_row_to_request(r) for r in rows]

    async def mark_reconciled(self, request_id: str) -> None:
        engine = self._require_engine()
        statement = (
            approvals.update()
            .where(approvals.c.id == request_id, approvals.c.effect_at.is_(None))
            .values(effect_at=func.now())
        )
        async with engine.begin() as conn:
            await conn.execute(statement)


def _row_to_request(row) -> ApprovalRequest:
    return ApprovalRequest(
        id=row["id"],
        agent=row["agent"],
        action=row["action"],
        tool=row["tool"],
        permission=row["permission"],
        # JSONB arrives decoded — the dialect registers a jsonb codec.
        args=row["args"] or {},
        approvers=row["approvers"] or [],
        summary=row["summary"],
        requested_by=row["requested_by"],
        status=row["status"],
        decided_by=row["decided_by"],
        args_schema=row["args_schema"],
        workflow_backed=row["workflow_backed"],
        workflow_instance_id=row["workflow_instance_id"],
        effect_at=row["effect_at"],
        created_at=row["created_at"] or datetime.now(UTC),
    )
