"""Postgres-backed workflow instances — workflows survive a process restart.

Mirrors :class:`InMemoryWorkflowStore` against a ``workflow_instances`` table,
following the approvals/usage/checkpoint store pattern. All arbitration
predicates run server-side (``now()``), so replicas need no clock agreement;
``drive_gen`` and ``leased_until`` are store-owned and never written from the
instance payload.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, literal, or_, select, text
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncEngine

from openloop.db import BorrowedEngineStore
from openloop.workflows.schema import workflow_instances
from openloop.workflows.store import TERMINAL, WorkflowInstance


class PostgresWorkflowStore(BorrowedEngineStore):
    async def setup(self, engine: AsyncEngine) -> None:
        # sql-text: schema evolution moves to a migration tool; DDL is not
        # restated as metadata.
        async with self._setup_connection(engine) as conn:
            await conn.execute(
                text(
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
            )
            await conn.execute(
                text(
                    "ALTER TABLE workflow_instances "
                    "ADD COLUMN IF NOT EXISTS leased_until TIMESTAMPTZ"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE workflow_instances "
                    "ADD COLUMN IF NOT EXISTS drive_gen INTEGER NOT NULL DEFAULT 0"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS workflow_instances_status_idx "
                    "ON workflow_instances (status, updated_at DESC)"
                )
            )

    async def get(self, instance_id: str) -> WorkflowInstance | None:
        engine = self._require_engine()
        statement = select(workflow_instances).where(
            workflow_instances.c.id == instance_id
        )
        async with engine.connect() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_instance(row) if row else None

    async def create(self, instance: WorkflowInstance) -> bool:
        engine = self._require_engine()
        statement = (
            insert(workflow_instances)
            .values(
                id=instance.id,
                workflow=instance.workflow,
                status=instance.status,
                completed_steps=instance.completed_steps,
                state=instance.state,
                waiting_on=instance.waiting_on,
                result=instance.result,
                error=instance.error,
                leased_until=instance.leased_until,
                drive_gen=instance.drive_gen,
            )
            .on_conflict_do_nothing(index_elements=[workflow_instances.c.id])
            .returning(workflow_instances.c.id)
        )
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).first()
        return row is not None

    async def claim_drive(
        self, instance_id: str, *, lease_seconds: float
    ) -> WorkflowInstance | None:
        engine = self._require_engine()
        # All arbitration predicates run server-side, so replicas need no clock
        # agreement — and the claim stays one conditional UPDATE.
        statement = (
            workflow_instances.update()
            .where(
                workflow_instances.c.id == instance_id,
                workflow_instances.c.status == "running",
                or_(
                    workflow_instances.c.leased_until.is_(None),
                    workflow_instances.c.leased_until < func.now(),
                ),
            )
            .values(
                drive_gen=workflow_instances.c.drive_gen + 1,
                leased_until=func.now() + _interval(lease_seconds),
                updated_at=func.now(),
            )
            .returning(*workflow_instances.c)
        )
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_instance(row) if row else None

    async def fenced_write(
        self, instance: WorkflowInstance, gen: int, *, release: bool = False
    ) -> bool:
        engine = self._require_engine()
        # drive_gen and leased_until are store-owned: the ordinary form leaves
        # both untouched (a payload write would roll back the ticker's renewed
        # lease); release bumps the gen and clears the lease. Built as a dict
        # rather than an interpolated SQL fragment.
        values: dict[str, Any] = {
            "workflow": instance.workflow,
            "status": instance.status,
            "completed_steps": instance.completed_steps,
            "state": instance.state,
            "waiting_on": instance.waiting_on,
            "result": instance.result,
            "error": instance.error,
            "updated_at": func.now(),
        }
        if release:
            values["drive_gen"] = workflow_instances.c.drive_gen + 1
            values["leased_until"] = None
        statement = (
            workflow_instances.update()
            .where(
                workflow_instances.c.id == instance.id,
                workflow_instances.c.drive_gen == gen,
            )
            .values(**values)
            .returning(workflow_instances.c.id)
        )
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).first()
        return row is not None

    async def renew_lease(
        self, instance_id: str, gen: int, *, lease_seconds: float
    ) -> bool:
        engine = self._require_engine()
        statement = (
            workflow_instances.update()
            .where(
                workflow_instances.c.id == instance_id,
                workflow_instances.c.drive_gen == gen,
            )
            .values(
                leased_until=func.now() + _interval(lease_seconds),
                updated_at=func.now(),
            )
            .returning(workflow_instances.c.id)
        )
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).first()
        return row is not None

    async def cancel_instance(
        self, instance_id: str, reason: str
    ) -> WorkflowInstance | None:
        engine = self._require_engine()
        statement = (
            workflow_instances.update()
            .where(
                workflow_instances.c.id == instance_id,
                # A terminal instance is never re-cancelled: this is a
                # correctness condition, not a filter.
                workflow_instances.c.status.notin_(TERMINAL),
            )
            .values(
                status="cancelled",
                error=func.nullif(literal(reason), ""),
                waiting_on=None,
                leased_until=None,
                drive_gen=workflow_instances.c.drive_gen + 1,
                updated_at=func.now(),
            )
            .returning(*workflow_instances.c)
        )
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_instance(row) if row else None

    async def claim_event(
        self, instance_id: str, event: str, payload: dict
    ) -> WorkflowInstance | None:
        """Atomically consume one exact wait event across all replicas."""
        engine = self._require_engine()
        # state.events gains one key, and the event name is appended to
        # completed_steps — both as jsonb concatenations so the whole claim
        # remains the single UPDATE that makes it exactly-once.
        events = func.coalesce(
            workflow_instances.c.state["events"], literal({}, JSONB)
        ).op("||", return_type=JSONB)(
            func.jsonb_build_object(literal(event), literal(payload, JSONB))
        )
        statement = (
            workflow_instances.update()
            .where(
                workflow_instances.c.id == instance_id,
                workflow_instances.c.status == "waiting",
                # The exact-event guard is what keeps the consume single-winner.
                workflow_instances.c.waiting_on == event,
            )
            .values(
                state=workflow_instances.c.state.op("||", return_type=JSONB)(
                    func.jsonb_build_object(literal("events"), events)
                ),
                completed_steps=workflow_instances.c.completed_steps.op(
                    "||", return_type=JSONB
                )(func.jsonb_build_array(literal(event))),
                status="running",
                waiting_on=None,
                leased_until=None,
                updated_at=func.now(),
            )
            .returning(*workflow_instances.c)
        )
        async with engine.begin() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return _row_to_instance(row) if row else None

    async def recent(self, limit: int = 100) -> list[WorkflowInstance]:
        engine = self._require_engine()
        statement = (
            select(workflow_instances)
            .order_by(workflow_instances.c.updated_at.desc())
            .limit(limit)
        )
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [_row_to_instance(r) for r in rows]


def _interval(seconds: float) -> Any:
    """`make_interval(secs => n)` — its arguments are positional in SQL.

    `func.make_interval(secs=...)` raises `TypeError: Function.__init__() got an
    unexpected keyword argument 'secs'`: SQLAlchemy reads keyword arguments as
    its own, never as SQL ones. The signature is
    (years, months, weeks, days, hours, mins, secs).
    """
    return func.make_interval(0, 0, 0, 0, 0, 0, seconds)


def _row_to_instance(row) -> WorkflowInstance:
    now = datetime.now(UTC)
    return WorkflowInstance(
        id=row["id"],
        workflow=row["workflow"],
        status=row["status"],
        # JSONB arrives decoded — the dialect registers a jsonb codec.
        completed_steps=row["completed_steps"] or [],
        state=row["state"] or {},
        waiting_on=row["waiting_on"],
        result=row["result"],
        error=row["error"],
        leased_until=row["leased_until"],
        drive_gen=row["drive_gen"],
        created_at=row["created_at"] or now,
        updated_at=row["updated_at"] or now,
    )
