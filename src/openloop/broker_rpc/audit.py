"""Secret-free append-only audit records for authenticated broker RPC."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from openloop.broker.models import (
    validate_positive_bigint,
    validate_timestamp,
    validate_uuid,
)
from openloop.broker_rpc.schema import broker_rpc_audit
from openloop.db import BorrowedEngineStore

from .identity import WorkloadIntent, WorkloadPrincipal


class AuditDecision(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    ERROR = "error"


class AuditReason(StrEnum):
    ALLOWED = "allowed"
    MISSING_INTENT = "missing_intent"
    NOT_FOUND_OR_UNAUTHORIZED = "not_found_or_unauthorized"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    STATE_CONFLICT = "state_conflict"
    INVALID_RECEIPT = "invalid_receipt"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    OVERLOADED = "overloaded"
    INTERNAL = "internal"


class RpcAuditProblem(Exception):
    """Safe failure raised when an authenticated RPC cannot be audited."""

    def __init__(self) -> None:
        super().__init__("broker RPC audit failed")


@dataclass(frozen=True, slots=True)
class PeerCredentials:
    pid: int
    uid: int
    gid: int

    def __post_init__(self) -> None:
        for name, value, minimum in (
            # Linux reports PID 0 when an AF_UNIX peer lives outside the
            # receiver's PID namespace. UID/GID remain kernel-authenticated.
            ("pid", self.pid, 0),
            ("uid", self.uid, 0),
            ("gid", self.gid, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"peer {name} must be an integer")
            if value < minimum:
                raise ValueError(f"peer {name} is invalid")


@dataclass(frozen=True, slots=True)
class RpcAuditRecord:
    request_id: UUID
    method: WorkloadIntent
    decision: AuditDecision
    reason: AuditReason
    peer: PeerCredentials
    principal: WorkloadPrincipal
    job_id: UUID | None = None
    operation_id: UUID | None = None

    def __post_init__(self) -> None:
        validate_uuid("request_id", self.request_id)
        if not isinstance(self.method, WorkloadIntent):
            raise TypeError("method must be WorkloadIntent")
        if not isinstance(self.decision, AuditDecision):
            raise TypeError("decision must be AuditDecision")
        if not isinstance(self.reason, AuditReason):
            raise TypeError("reason must be AuditReason")
        if not isinstance(self.peer, PeerCredentials):
            raise TypeError("peer must be PeerCredentials")
        if not isinstance(self.principal, WorkloadPrincipal):
            raise TypeError("principal must be WorkloadPrincipal")
        if self.job_id is not None:
            validate_uuid("job_id", self.job_id)
        if self.operation_id is not None:
            validate_uuid("operation_id", self.operation_id)


@dataclass(frozen=True, slots=True)
class StoredRpcAuditRecord:
    sequence: int
    created_at: datetime
    request: RpcAuditRecord

    def __post_init__(self) -> None:
        validate_positive_bigint("sequence", self.sequence)
        validate_timestamp("created_at", self.created_at)
        if not isinstance(self.request, RpcAuditRecord):
            raise TypeError("request must be RpcAuditRecord")

    def __getattr__(self, name: str):
        # Preserve a convenient immutable read model without copying secret-free
        # request fields into the durable envelope.
        try:
            return getattr(self.request, name)
        except AttributeError as error:
            raise AttributeError(name) from error


@runtime_checkable
class RpcAuditSink(Protocol):
    async def append(self, record: RpcAuditRecord) -> StoredRpcAuditRecord: ...


class InMemoryRpcAuditSink:
    def __init__(self, *, clock: Callable[[], datetime]) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._records: list[StoredRpcAuditRecord] = []
        self._lock = asyncio.Lock()

    async def append(self, record: RpcAuditRecord) -> StoredRpcAuditRecord:
        if not isinstance(record, RpcAuditRecord):
            raise TypeError("record must be RpcAuditRecord")
        async with self._lock:
            created_at = self._clock()
            validate_timestamp("audit clock", created_at)
            stored = StoredRpcAuditRecord(
                sequence=len(self._records) + 1,
                created_at=created_at,
                request=record,
            )
            self._records.append(stored)
            return stored

    async def records_for_test(self) -> tuple[StoredRpcAuditRecord, ...]:
        async with self._lock:
            return tuple(self._records)


class PostgresRpcAuditSink(BorrowedEngineStore):
    """Append authenticated RPC decisions to migration-owned durable storage."""

    async def setup(self, engine: AsyncEngine) -> None:
        async with self._setup_connection(engine) as connection:
            # sql-text: `to_regclass` is a presence probe against a table this
            # store never creates — there is nothing here to build from metadata.
            exists = await connection.scalar(
                text("SELECT to_regclass('broker_rpc_audit') IS NOT NULL")
            )
            if exists is not True:
                raise RpcAuditProblem()

    async def append(self, record: RpcAuditRecord) -> StoredRpcAuditRecord:
        if not isinstance(record, RpcAuditRecord):
            raise TypeError("record must be RpcAuditRecord")
        engine = self._require_engine()
        principal = record.principal
        statement = (
            broker_rpc_audit.insert()
            .values(
                request_id=record.request_id,
                method=record.method.value,
                decision=record.decision.value,
                reason_code=record.reason.value,
                peer_pid=record.peer.pid,
                peer_uid=record.peer.uid,
                peer_gid=record.peer.gid,
                tenant_id=principal.owner.tenant_id,
                workload_subject=principal.owner.workload_subject,
                worker_instance_id=principal.worker_instance_id,
                assignment_id=principal.assignment_id,
                isolation_mode=principal.isolation_mode.value,
                required_isolation=principal.required_isolation.value,
                jwt_key_id=principal.key_id,
                jwt_id=principal.jwt_id,
                job_id=record.job_id,
                operation_id=record.operation_id,
            )
            .returning(broker_rpc_audit.c.sequence, broker_rpc_audit.c.created_at)
        )
        try:
            async with engine.begin() as connection:
                row = (await connection.execute(statement)).mappings().first()
        except Exception as error:
            raise RpcAuditProblem() from error
        if row is None:
            raise RpcAuditProblem()
        try:
            return StoredRpcAuditRecord(
                sequence=row["sequence"],
                created_at=row["created_at"],
                request=record,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RpcAuditProblem() from error
