"""Secret-free append-only audit records for authenticated broker RPC."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable
from uuid import UUID

from openloop.broker.models import (
    validate_positive_bigint,
    validate_timestamp,
    validate_uuid,
)
from openloop.postgres import BorrowedPostgresStore

from .identity import WorkloadIntent, WorkloadPrincipal


class AuditDecision(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    ERROR = "error"


class AuditReason(str, Enum):
    ALLOWED = "allowed"
    MISSING_INTENT = "missing_intent"
    NOT_FOUND_OR_UNAUTHORIZED = "not_found_or_unauthorized"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    STATE_CONFLICT = "state_conflict"
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


class PostgresRpcAuditSink(BorrowedPostgresStore):
    """Append authenticated RPC decisions to migration-owned durable storage."""

    async def setup(self, pool) -> None:
        async with self._setup_connection(pool) as connection:
            exists = await connection.fetchval(
                "SELECT to_regclass('broker_rpc_audit') IS NOT NULL"
            )
            if exists is not True:
                raise RpcAuditProblem()

    async def append(self, record: RpcAuditRecord) -> StoredRpcAuditRecord:
        if not isinstance(record, RpcAuditRecord):
            raise TypeError("record must be RpcAuditRecord")
        pool = self._require_pool()
        principal = record.principal
        try:
            async with pool.acquire() as connection:
                row = await connection.fetchrow(
                    """
                    INSERT INTO broker_rpc_audit (
                        request_id, method, decision, reason_code,
                        peer_pid, peer_uid, peer_gid,
                        tenant_id, workload_subject,
                        worker_instance_id, assignment_id,
                        isolation_mode, required_isolation,
                        jwt_key_id, jwt_id, job_id, operation_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9,
                        $10, $11, $12, $13, $14, $15, $16, $17
                    )
                    RETURNING sequence, created_at
                    """,
                    record.request_id,
                    record.method.value,
                    record.decision.value,
                    record.reason.value,
                    record.peer.pid,
                    record.peer.uid,
                    record.peer.gid,
                    principal.owner.tenant_id,
                    principal.owner.workload_subject,
                    principal.worker_instance_id,
                    principal.assignment_id,
                    principal.isolation_mode.value,
                    principal.required_isolation.value,
                    principal.key_id,
                    principal.jwt_id,
                    record.job_id,
                    record.operation_id,
                )
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
