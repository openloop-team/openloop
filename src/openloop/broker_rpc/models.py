"""Frozen typed request and response values for broker RPC version two."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from uuid import UUID

from openloop.broker.models import (
    GenerationState,
    JobState,
    JobSnapshot,
    OperationTicket,
    ReleaseTarget,
    SignedCheckpointReceipt,
    TerminalOutcome,
    validate_bigint,
    validate_idempotency_key,
    validate_identifier,
    validate_positive_bigint,
    validate_timestamp,
    validate_uuid,
)

from .capability import JobCapability
from .errors import RpcFailure
from .identity import WorkloadIdentityToken, WorkloadIntent


RPC_VERSION = 2
_ACCESS_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}\Z")


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


@dataclass(frozen=True, slots=True)
class StartSegmentPayload:
    job_id: UUID
    expected_generation: int
    idempotency_key: str

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class QuiesceSegmentPayload:
    job_id: UUID
    expected_generation: int
    idempotency_key: str
    barrier_id: str

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        validate_idempotency_key(self.idempotency_key)
        validate_identifier("barrier_id", self.barrier_id)


@dataclass(frozen=True, slots=True)
class ReleaseSegmentPayload:
    job_id: UUID
    expected_generation: int
    idempotency_key: str
    receipt: SignedCheckpointReceipt
    target: ReleaseTarget
    terminal_outcome: TerminalOutcome | None = None

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        validate_idempotency_key(self.idempotency_key)
        if not isinstance(self.receipt, SignedCheckpointReceipt):
            raise TypeError("receipt must be SignedCheckpointReceipt")
        if not isinstance(self.target, ReleaseTarget):
            raise TypeError("target must be ReleaseTarget")
        if self.target is ReleaseTarget.FINALIZING:
            if not isinstance(self.terminal_outcome, TerminalOutcome):
                raise ValueError("finalizing release requires terminal_outcome")
        elif self.terminal_outcome is not None:
            raise ValueError("parked release cannot set terminal_outcome")


@dataclass(frozen=True, slots=True)
class FinalizeJobPayload:
    job_id: UUID
    expected_generation: int
    idempotency_key: str
    terminal_outcome: TerminalOutcome

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        validate_idempotency_key(self.idempotency_key)
        if not isinstance(self.terminal_outcome, TerminalOutcome):
            raise TypeError("terminal_outcome must be TerminalOutcome")


RpcPayload = (
    CreateJobPayload
    | InspectJobPayload
    | StartSegmentPayload
    | QuiesceSegmentPayload
    | ReleaseSegmentPayload
    | FinalizeJobPayload
)


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
        elif self.method is WorkloadIntent.START_SEGMENT:
            if not isinstance(self.job_capability, JobCapability):
                raise ValueError("START_SEGMENT requires a job capability")
            if not isinstance(self.payload, StartSegmentPayload):
                raise TypeError("START_SEGMENT requires StartSegmentPayload")
        elif self.method is WorkloadIntent.QUIESCE_SEGMENT:
            if not isinstance(self.job_capability, JobCapability):
                raise ValueError("QUIESCE_SEGMENT requires a job capability")
            if not isinstance(self.payload, QuiesceSegmentPayload):
                raise TypeError("QUIESCE_SEGMENT requires QuiesceSegmentPayload")
        elif self.method is WorkloadIntent.RELEASE_SEGMENT:
            if not isinstance(self.job_capability, JobCapability):
                raise ValueError("RELEASE_SEGMENT requires a job capability")
            if not isinstance(self.payload, ReleaseSegmentPayload):
                raise TypeError("RELEASE_SEGMENT requires ReleaseSegmentPayload")
        elif self.method is WorkloadIntent.FINALIZE_JOB:
            if not isinstance(self.job_capability, JobCapability):
                raise ValueError("FINALIZE_JOB requires a job capability")
            if not isinstance(self.payload, FinalizeJobPayload):
                raise TypeError("FINALIZE_JOB requires FinalizeJobPayload")
        else:
            raise ValueError("unsupported RPC method")


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
    access: RunningGenerationAccess | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, JobSnapshot):
            raise TypeError("snapshot must be JobSnapshot")
        if self.access is not None and not isinstance(
            self.access, RunningGenerationAccess
        ):
            raise TypeError("access must be RunningGenerationAccess")


@dataclass(frozen=True, slots=True, repr=False)
class RunningGenerationAccess:
    job_id: UUID
    conversation_id: UUID
    generation: int
    deadline: datetime
    socket_path: Path
    relay_capability: str
    session_api_key: str

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        validate_uuid("conversation_id", self.conversation_id)
        validate_positive_bigint("generation", self.generation)
        validate_timestamp("deadline", self.deadline)
        if (
            self.deadline.utcoffset() != UTC.utcoffset(self.deadline)
            or self.deadline.microsecond
        ):
            raise ValueError("deadline must be whole-second UTC")
        if (
            not isinstance(self.socket_path, Path)
            or not self.socket_path.is_absolute()
        ):
            raise ValueError("socket_path must be an absolute pathlib.Path")
        rendered_path = str(self.socket_path)
        if (
            "\0" in rendered_path
            or self.socket_path.name != "agent.sock"
            or len(rendered_path.encode("utf-8")) > 100
        ):
            raise ValueError("socket_path is outside the relay UDS profile")
        for name in ("relay_capability", "session_api_key"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or _ACCESS_TOKEN.fullmatch(value) is None
            ):
                raise ValueError(f"{name} is invalid")

    def __repr__(self) -> str:
        return (
            "RunningGenerationAccess("
            f"job_id={str(self.job_id)!r}, "
            f"conversation_id={str(self.conversation_id)!r}, "
            f"generation={self.generation}, deadline={self.deadline!r}, "
            f"socket_path={str(self.socket_path)!r}, "
            "relay_capability=<redacted>, session_api_key=<redacted>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class CheckpointGenerationAccess(RunningGenerationAccess):
    def __repr__(self) -> str:
        return (
            "CheckpointGenerationAccess("
            f"job_id={str(self.job_id)!r}, "
            f"conversation_id={str(self.conversation_id)!r}, "
            f"generation={self.generation}, deadline={self.deadline!r}, "
            f"socket_path={str(self.socket_path)!r}, "
            "relay_capability=<redacted>, session_api_key=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class StartSegmentResult:
    operation_id: UUID
    replayed: bool
    access: RunningGenerationAccess

    def __post_init__(self) -> None:
        validate_uuid("operation_id", self.operation_id)
        if type(self.replayed) is not bool:
            raise TypeError("replayed must be a bool")
        if not isinstance(self.access, RunningGenerationAccess):
            raise TypeError("access must be RunningGenerationAccess")


@dataclass(frozen=True, slots=True)
class QuiesceSegmentResult:
    operation_id: UUID
    replayed: bool
    access: CheckpointGenerationAccess

    def __post_init__(self) -> None:
        validate_uuid("operation_id", self.operation_id)
        if type(self.replayed) is not bool:
            raise TypeError("replayed must be a bool")
        if not isinstance(self.access, CheckpointGenerationAccess):
            raise TypeError("access must be CheckpointGenerationAccess")


@dataclass(frozen=True, slots=True)
class ReleaseSegmentResult:
    operation_id: UUID
    replayed: bool
    job_state: JobState
    generation_state: GenerationState

    def __post_init__(self) -> None:
        validate_uuid("operation_id", self.operation_id)
        if type(self.replayed) is not bool:
            raise TypeError("replayed must be a bool")
        if self.job_state not in {JobState.PARKED, JobState.FINALIZING}:
            raise ValueError("released job state is invalid")
        if self.generation_state is not GenerationState.RELEASED:
            raise ValueError("released generation state is invalid")


@dataclass(frozen=True, slots=True)
class FinalizeJobResult:
    operation_id: UUID
    replayed: bool
    job_state: JobState

    def __post_init__(self) -> None:
        validate_uuid("operation_id", self.operation_id)
        if type(self.replayed) is not bool:
            raise TypeError("replayed must be a bool")
        if self.job_state is not JobState.TERMINAL:
            raise ValueError("finalized job state must be terminal")


RpcResult = (
    CreateJobResult
    | InspectJobResult
    | StartSegmentResult
    | QuiesceSegmentResult
    | ReleaseSegmentResult
    | FinalizeJobResult
)


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
            self.result,
            (
                CreateJobResult,
                InspectJobResult,
                StartSegmentResult,
                QuiesceSegmentResult,
                ReleaseSegmentResult,
                FinalizeJobResult,
            ),
        ):
            raise TypeError("result has an unsupported type")
        if self.failure is not None and not isinstance(self.failure, RpcFailure):
            raise TypeError("failure must be RpcFailure")

    @property
    def ok(self) -> bool:
        return self.result is not None
