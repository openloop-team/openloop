"""Frozen typed request and response values for broker RPC version one."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from openloop.broker.models import (
    JobSnapshot,
    OperationTicket,
    validate_idempotency_key,
    validate_uuid,
)

from .capability import JobCapability
from .errors import RpcFailure
from .identity import WorkloadIdentityToken, WorkloadIntent


RPC_VERSION = 1


@dataclass(frozen=True, slots=True)
class CreateJobPayload:
    idempotency_key: str

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class InspectJobPayload:
    job_id: UUID

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)


RpcPayload = CreateJobPayload | InspectJobPayload


@dataclass(frozen=True, slots=True)
class RpcRequest:
    version: int
    request_id: UUID
    method: WorkloadIntent
    identity_token: WorkloadIdentityToken
    job_capability: JobCapability | None
    payload: RpcPayload

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or self.version != RPC_VERSION:
            raise ValueError("unsupported RPC version")
        validate_uuid("request_id", self.request_id)
        if not isinstance(self.method, WorkloadIntent):
            raise TypeError("method must be WorkloadIntent")
        if not isinstance(self.identity_token, WorkloadIdentityToken):
            raise TypeError("identity_token must be WorkloadIdentityToken")
        if self.method is WorkloadIntent.CREATE_JOB:
            if self.job_capability is not None:
                raise ValueError("CREATE_JOB cannot carry a job capability")
            if not isinstance(self.payload, CreateJobPayload):
                raise TypeError("CREATE_JOB requires CreateJobPayload")
        elif self.method is WorkloadIntent.INSPECT_JOB:
            if not isinstance(self.job_capability, JobCapability):
                raise ValueError("INSPECT_JOB requires a job capability")
            if not isinstance(self.payload, InspectJobPayload):
                raise TypeError("INSPECT_JOB requires InspectJobPayload")


@dataclass(frozen=True, slots=True)
class CreateJobResult:
    ticket: OperationTicket
    capability: JobCapability

    def __post_init__(self) -> None:
        if not isinstance(self.ticket, OperationTicket):
            raise TypeError("ticket must be OperationTicket")
        if not isinstance(self.capability, JobCapability):
            raise TypeError("capability must be JobCapability")


@dataclass(frozen=True, slots=True)
class InspectJobResult:
    snapshot: JobSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, JobSnapshot):
            raise TypeError("snapshot must be JobSnapshot")


RpcResult = CreateJobResult | InspectJobResult


@dataclass(frozen=True, slots=True)
class RpcResponse:
    version: int
    request_id: UUID
    result: RpcResult | None = None
    failure: RpcFailure | None = None

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or self.version != RPC_VERSION:
            raise ValueError("unsupported RPC version")
        validate_uuid("request_id", self.request_id)
        if (self.result is None) == (self.failure is None):
            raise ValueError("response requires exactly one result or failure")
        if self.result is not None and not isinstance(
            self.result, (CreateJobResult, InspectJobResult)
        ):
            raise TypeError("result has an unsupported type")
        if self.failure is not None and not isinstance(self.failure, RpcFailure):
            raise TypeError("failure must be RpcFailure")

    @property
    def ok(self) -> bool:
        return self.result is not None

