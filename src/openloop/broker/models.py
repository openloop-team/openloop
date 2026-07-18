"""Immutable bounded domain values for the broker lifecycle ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import unicodedata
from uuid import UUID


POSTGRES_BIGINT_MAX = 2**63 - 1
MAX_BROKER_JSON_BYTES = 16 * 1024

_TOKEN = re.compile(r"[a-z0-9_-]+\Z")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_BASE_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SIGNED_RECEIPT = re.compile(r"[A-Za-z0-9_.-]+\Z")


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class JobState(_StringEnum):
    CREATED = "created"
    ACTIVE = "active"
    PARKED = "parked"
    FINALIZING = "finalizing"
    TERMINAL = "terminal"


class GenerationState(_StringEnum):
    STARTING = "starting"
    RUNNING = "running"
    QUIESCING = "quiescing"
    QUIESCED = "quiesced"
    RELEASING = "releasing"
    RELEASED = "released"
    ABANDONED = "abandoned"


class OperationStatus(_StringEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class OperationSource(_StringEnum):
    CALLER = "caller"
    INTERNAL = "internal"


class ReleaseTarget(_StringEnum):
    PARKED = "parked"
    FINALIZING = "finalizing"


class IsolationMode(_StringEnum):
    SHARED = "shared"
    DEDICATED = "dedicated"

    def allows(self, required: "IsolationMode") -> bool:
        if not isinstance(required, IsolationMode):
            raise TypeError("required must be an IsolationMode")
        rank = {IsolationMode.SHARED: 0, IsolationMode.DEDICATED: 1}
        return rank[self] >= rank[required]


class TerminalOutcome(_StringEnum):
    SUCCESS = "success"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CommandKind(_StringEnum):
    CREATE_JOB = "create_job"
    BEGIN_START = "begin_start"
    MARK_RUNNING = "mark_running"
    ABANDON_GENERATION = "abandon_generation"
    BEGIN_QUIESCE = "begin_quiesce"
    MARK_QUIESCED = "mark_quiesced"
    BEGIN_RELEASE = "begin_release"
    MARK_RELEASED = "mark_released"
    BEGIN_FINALIZE = "begin_finalize"
    MARK_TERMINAL = "mark_terminal"


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


def _validate_utf8(
    name: str, value: object, *, minimum: int, maximum: int
) -> str:
    text = _require_string(name, value)
    length = len(text.encode("utf-8"))
    if not minimum <= length <= maximum:
        raise ValueError(f"{name} must be {minimum}-{maximum} UTF-8 bytes")
    if _contains_control(text):
        raise ValueError(f"{name} must not contain control characters")
    return text


def validate_tenant_id(value: object) -> str:
    return _validate_utf8("tenant_id", value, minimum=1, maximum=128)


def validate_workload_subject(value: object) -> str:
    return _validate_utf8("workload_subject", value, minimum=1, maximum=256)


def validate_idempotency_key(value: object) -> str:
    text = _require_string("idempotency_key", value)
    if not 16 <= len(text) <= 128 or any(
        not 33 <= ord(character) <= 126 for character in text
    ):
        raise ValueError("idempotency_key must be 16-128 visible ASCII bytes")
    return text


def validate_token(name: str, value: object) -> str:
    text = _require_string(name, value)
    if not 1 <= len(text) <= 64 or _TOKEN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a 1-64 byte lowercase token")
    return text


def validate_identifier(name: str, value: object) -> str:
    return _validate_utf8(name, value, minimum=1, maximum=256)


def validate_opaque_ref(name: str, value: object) -> str:
    return _validate_utf8(name, value, minimum=1, maximum=1024)


def validate_sha256(name: str, value: object) -> str:
    text = _require_string(name, value)
    if _LOWER_HEX_64.fullmatch(text) is None:
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return text


def validate_base_commit(value: object) -> str:
    text = _require_string("base_commit", value)
    if _BASE_COMMIT.fullmatch(text) is None:
        raise ValueError("base_commit must be 40 or 64 lowercase hexadecimal characters")
    return text


def validate_bigint(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= POSTGRES_BIGINT_MAX:
        raise ValueError(f"{name} must fit a nonnegative PostgreSQL BIGINT")
    return value


def validate_positive_bigint(name: str, value: object) -> int:
    integer = validate_bigint(name, value)
    if integer == 0:
        raise ValueError(f"{name} must be positive")
    return integer


def validate_lease_seconds(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("execution_lease_seconds must be an integer")
    if not 1 <= value <= 86_400:
        raise ValueError("execution_lease_seconds must be between 1 and 86400")
    return value


def validate_uuid(name: str, value: object) -> UUID:
    if not isinstance(value, UUID):
        raise TypeError(f"{name} must be a UUID")
    return value


def validate_timestamp(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _validate_optional_uuid(name: str, value: UUID | None) -> None:
    if value is not None:
        validate_uuid(name, value)


def _validate_optional_identifier(name: str, value: str | None) -> None:
    if value is not None:
        validate_identifier(name, value)


def _validate_optional_token(name: str, value: str | None) -> None:
    if value is not None:
        validate_token(name, value)


def _validate_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True)
class BrokerOwner:
    tenant_id: str
    workload_subject: str

    def __post_init__(self) -> None:
        validate_tenant_id(self.tenant_id)
        validate_workload_subject(self.workload_subject)


@dataclass(frozen=True, slots=True)
class JobAuthorizationMetadata:
    key_version: str
    epoch: int
    capability_digest: str = field(repr=False)

    def __post_init__(self) -> None:
        validate_identifier("key_version", self.key_version)
        validate_positive_bigint("epoch", self.epoch)
        validate_sha256("capability_digest", self.capability_digest)


@dataclass(frozen=True, slots=True)
class JobAuthorizationRecord:
    job_id: UUID
    owner: BrokerOwner
    minimum_isolation: IsolationMode
    authorization: JobAuthorizationMetadata

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        if not isinstance(self.owner, BrokerOwner):
            raise TypeError("owner must be a BrokerOwner")
        if not isinstance(self.minimum_isolation, IsolationMode):
            raise TypeError("minimum_isolation must be an IsolationMode")
        if not isinstance(self.authorization, JobAuthorizationMetadata):
            raise TypeError("authorization must be JobAuthorizationMetadata")


@dataclass(frozen=True, slots=True, repr=False)
class SignedCheckpointReceipt:
    """Opaque checkpoint-store assertion; trusted only after signature verification."""

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or not self.value.isascii()
            or not 1 <= len(self.value) <= 16 * 1024
            or _SIGNED_RECEIPT.fullmatch(self.value) is None
        ):
            raise ValueError("signed checkpoint receipt encoding is invalid")


@dataclass(frozen=True, slots=True)
class VerifiedCheckpointReceipt:
    issuer: str
    receipt_id: str
    tenant_id: str
    job_id: UUID
    conversation_id: UUID
    generation: int
    barrier_id: str
    artifact_id: str
    base_commit: str
    ciphertext_sha256: str = field(repr=False)
    plaintext_sha256: str = field(repr=False)
    byte_count: int
    store_version: str
    envelope_version: str
    key_version: str
    durable_write_sequence: int

    def __post_init__(self) -> None:
        validate_identifier("issuer", self.issuer)
        validate_identifier("receipt_id", self.receipt_id)
        validate_tenant_id(self.tenant_id)
        validate_uuid("job_id", self.job_id)
        validate_uuid("conversation_id", self.conversation_id)
        validate_positive_bigint("generation", self.generation)
        validate_identifier("barrier_id", self.barrier_id)
        validate_identifier("artifact_id", self.artifact_id)
        validate_base_commit(self.base_commit)
        validate_sha256("ciphertext_sha256", self.ciphertext_sha256)
        validate_sha256("plaintext_sha256", self.plaintext_sha256)
        validate_bigint("byte_count", self.byte_count)
        validate_identifier("store_version", self.store_version)
        validate_identifier("envelope_version", self.envelope_version)
        validate_identifier("key_version", self.key_version)
        validate_bigint("durable_write_sequence", self.durable_write_sequence)


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: UUID
    conversation_id: UUID
    owner: BrokerOwner
    profile: str
    runtime_driver: str
    durable_state_driver: str
    state: JobState
    revision: int
    generation: int
    current_generation: int | None
    pending_operation_id: UUID | None
    durable_state_ref: str | None = field(repr=False)
    durable_key_version: str | None
    durable_digest: str | None = field(repr=False)
    terminal_outcome: TerminalOutcome | None
    created_at: datetime
    updated_at: datetime
    minimum_isolation: IsolationMode | None = None
    authorization: JobAuthorizationMetadata | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        validate_uuid("conversation_id", self.conversation_id)
        if not isinstance(self.owner, BrokerOwner):
            raise TypeError("owner must be a BrokerOwner")
        validate_token("profile", self.profile)
        validate_token("runtime_driver", self.runtime_driver)
        validate_token("durable_state_driver", self.durable_state_driver)
        if not isinstance(self.state, JobState):
            raise TypeError("state must be a JobState")
        validate_positive_bigint("revision", self.revision)
        validate_bigint("generation", self.generation)
        if self.current_generation is not None:
            validate_positive_bigint("current_generation", self.current_generation)
            if self.current_generation > self.generation:
                raise ValueError("current_generation cannot exceed generation")
        _validate_optional_uuid("pending_operation_id", self.pending_operation_id)
        if self.durable_state_ref is not None:
            validate_opaque_ref("durable_state_ref", self.durable_state_ref)
        _validate_optional_identifier("durable_key_version", self.durable_key_version)
        if self.durable_digest is not None:
            validate_sha256("durable_digest", self.durable_digest)
        if self.state in {JobState.FINALIZING, JobState.TERMINAL}:
            if self.terminal_outcome is None:
                raise ValueError("terminal_outcome is required while finalizing")
        elif self.terminal_outcome is not None:
            raise ValueError("terminal_outcome is only valid while finalizing")
        validate_timestamp("created_at", self.created_at)
        validate_timestamp("updated_at", self.updated_at)
        if (self.minimum_isolation is None) != (self.authorization is None):
            raise ValueError("job authorization fields must be all null or all present")
        if self.minimum_isolation is not None and not isinstance(
            self.minimum_isolation, IsolationMode
        ):
            raise TypeError("minimum_isolation must be an IsolationMode")
        if self.authorization is not None and not isinstance(
            self.authorization, JobAuthorizationMetadata
        ):
            raise TypeError("authorization must be JobAuthorizationMetadata")


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    job_id: UUID
    generation: int
    state: GenerationState
    revision: int
    previous_job_state: JobState
    start_operation_id: UUID
    pending_operation_id: UUID | None
    runtime_ref: str | None = field(repr=False)
    durable_state_ref: str | None = field(repr=False)
    runtime_key_version: str | None
    durable_key_version: str | None
    capability_digest: str | None = field(repr=False)
    durable_digest: str | None = field(repr=False)
    execution_lease_deadline: datetime
    barrier_id: str | None
    receipt: VerifiedCheckpointReceipt | None = field(repr=False)
    release_target: ReleaseTarget | None
    release_terminal_outcome: TerminalOutcome | None
    failure_reason_code: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        validate_positive_bigint("generation", self.generation)
        if not isinstance(self.state, GenerationState):
            raise TypeError("state must be a GenerationState")
        validate_positive_bigint("revision", self.revision)
        if self.previous_job_state not in {JobState.CREATED, JobState.PARKED}:
            raise ValueError("previous_job_state must be created or parked")
        validate_uuid("start_operation_id", self.start_operation_id)
        _validate_optional_uuid("pending_operation_id", self.pending_operation_id)
        if self.runtime_ref is not None:
            validate_opaque_ref("runtime_ref", self.runtime_ref)
        if self.durable_state_ref is not None:
            validate_opaque_ref("durable_state_ref", self.durable_state_ref)
        _validate_optional_identifier("runtime_key_version", self.runtime_key_version)
        _validate_optional_identifier("durable_key_version", self.durable_key_version)
        if self.capability_digest is not None:
            validate_sha256("capability_digest", self.capability_digest)
        if self.durable_digest is not None:
            validate_sha256("durable_digest", self.durable_digest)
        validate_timestamp("execution_lease_deadline", self.execution_lease_deadline)
        _validate_optional_identifier("barrier_id", self.barrier_id)
        if self.receipt is not None and not isinstance(
            self.receipt, VerifiedCheckpointReceipt
        ):
            raise TypeError("receipt must be a VerifiedCheckpointReceipt")
        if self.release_target is not None and not isinstance(
            self.release_target, ReleaseTarget
        ):
            raise TypeError("release_target must be a ReleaseTarget")
        if self.release_terminal_outcome is not None and not isinstance(
            self.release_terminal_outcome, TerminalOutcome
        ):
            raise TypeError("release_terminal_outcome must be a TerminalOutcome")
        _validate_optional_token("failure_reason_code", self.failure_reason_code)
        validate_timestamp("created_at", self.created_at)
        validate_timestamp("updated_at", self.updated_at)


@dataclass(frozen=True, slots=True)
class OperationTicket:
    operation_id: UUID
    command: CommandKind
    job_id: UUID | None = None
    conversation_id: UUID | None = None
    generation: int | None = None
    job_state: JobState | None = None
    generation_state: GenerationState | None = None
    replayed: bool = False

    def __post_init__(self) -> None:
        validate_uuid("operation_id", self.operation_id)
        if not isinstance(self.command, CommandKind):
            raise TypeError("command must be a CommandKind")
        _validate_optional_uuid("job_id", self.job_id)
        _validate_optional_uuid("conversation_id", self.conversation_id)
        if self.generation is not None:
            validate_positive_bigint("generation", self.generation)
        _validate_bool("replayed", self.replayed)


@dataclass(frozen=True, slots=True)
class OperationResult:
    operation_id: UUID
    command: CommandKind
    job_id: UUID
    generation: int | None
    job_state: JobState
    generation_state: GenerationState | None
    replayed: bool = False

    def __post_init__(self) -> None:
        validate_uuid("operation_id", self.operation_id)
        if not isinstance(self.command, CommandKind):
            raise TypeError("command must be a CommandKind")
        validate_uuid("job_id", self.job_id)
        if self.generation is not None:
            validate_positive_bigint("generation", self.generation)
        if not isinstance(self.job_state, JobState):
            raise TypeError("job_state must be a JobState")
        if self.generation_state is not None and not isinstance(
            self.generation_state, GenerationState
        ):
            raise TypeError("generation_state must be a GenerationState")
        _validate_bool("replayed", self.replayed)


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: UUID
    owner: BrokerOwner
    source: OperationSource
    idempotency_key: str | None
    command: CommandKind
    request_digest: str = field(repr=False)
    job_id: UUID | None
    generation: int | None
    status: OperationStatus
    intent_ticket: OperationTicket
    completion_result: OperationResult | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        validate_uuid("operation_id", self.operation_id)
        if not isinstance(self.owner, BrokerOwner):
            raise TypeError("owner must be a BrokerOwner")
        if not isinstance(self.source, OperationSource):
            raise TypeError("source must be an OperationSource")
        if self.source is OperationSource.CALLER:
            if self.idempotency_key is None:
                raise ValueError("caller operation requires an idempotency key")
            validate_idempotency_key(self.idempotency_key)
        elif self.idempotency_key is not None:
            raise ValueError("internal operation cannot use an idempotency key")
        if not isinstance(self.command, CommandKind):
            raise TypeError("command must be a CommandKind")
        validate_sha256("request_digest", self.request_digest)
        _validate_optional_uuid("job_id", self.job_id)
        if self.generation is not None:
            validate_positive_bigint("generation", self.generation)
        if not isinstance(self.status, OperationStatus):
            raise TypeError("status must be an OperationStatus")
        if not isinstance(self.intent_ticket, OperationTicket):
            raise TypeError("intent_ticket must be an OperationTicket")
        if self.completion_result is not None and not isinstance(
            self.completion_result, OperationResult
        ):
            raise TypeError("completion_result must be an OperationResult")
        validate_timestamp("created_at", self.created_at)
        validate_timestamp("updated_at", self.updated_at)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    sequence: int
    command: CommandKind
    owner: BrokerOwner
    job_id: UUID
    generation: int | None
    operation_id: UUID
    before_job_state: JobState | None
    after_job_state: JobState
    before_generation_state: GenerationState | None
    after_generation_state: GenerationState | None
    reason_code: str | None
    created_at: datetime

    def __post_init__(self) -> None:
        validate_positive_bigint("sequence", self.sequence)
        if not isinstance(self.command, CommandKind):
            raise TypeError("command must be a CommandKind")
        if not isinstance(self.owner, BrokerOwner):
            raise TypeError("owner must be a BrokerOwner")
        validate_uuid("job_id", self.job_id)
        if self.generation is not None:
            validate_positive_bigint("generation", self.generation)
        validate_uuid("operation_id", self.operation_id)
        _validate_optional_token("reason_code", self.reason_code)
        validate_timestamp("created_at", self.created_at)


@dataclass(frozen=True, slots=True)
class GenerationSnapshot:
    generation: int
    state: GenerationState
    revision: int
    previous_job_state: JobState
    start_operation_id: UUID
    pending_operation_id: UUID | None
    runtime_key_version: str | None
    durable_key_version: str | None
    execution_lease_deadline: datetime
    barrier_id: str | None
    receipt_id: str | None
    release_target: ReleaseTarget | None
    release_terminal_outcome: TerminalOutcome | None
    failure_reason_code: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    job_id: UUID
    conversation_id: UUID
    owner: BrokerOwner
    profile: str
    runtime_driver: str
    durable_state_driver: str
    state: JobState
    revision: int
    generation: int
    current_generation: int | None
    pending_operation_id: UUID | None
    durable_key_version: str | None
    terminal_outcome: TerminalOutcome | None
    created_at: datetime
    updated_at: datetime
    generation_record: GenerationSnapshot | None


@dataclass(frozen=True, slots=True)
class RecoveryGenerationSnapshot:
    generation: int
    state: GenerationState
    revision: int
    previous_job_state: JobState
    start_operation_id: UUID
    pending_operation_id: UUID | None
    runtime_ref: str | None = field(repr=False)
    durable_state_ref: str | None = field(repr=False)
    runtime_key_version: str | None
    durable_key_version: str | None
    capability_digest: str | None = field(repr=False)
    durable_digest: str | None = field(repr=False)
    execution_lease_deadline: datetime
    barrier_id: str | None
    receipt: VerifiedCheckpointReceipt | None = field(repr=False)
    release_target: ReleaseTarget | None
    release_terminal_outcome: TerminalOutcome | None
    failure_reason_code: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    job_id: UUID
    conversation_id: UUID
    owner: BrokerOwner
    profile: str
    runtime_driver: str
    durable_state_driver: str
    state: JobState
    revision: int
    generation: int
    current_generation: int | None
    pending_operation_id: UUID | None
    durable_state_ref: str | None = field(repr=False)
    durable_key_version: str | None
    durable_digest: str | None = field(repr=False)
    terminal_outcome: TerminalOutcome | None
    created_at: datetime
    updated_at: datetime
    generation_record: RecoveryGenerationSnapshot | None
    minimum_isolation: IsolationMode | None = None
    authorization: JobAuthorizationMetadata | None = field(
        default=None, repr=False
    )


def _project_generation(record: GenerationRecord) -> GenerationSnapshot:
    return GenerationSnapshot(
        generation=record.generation,
        state=record.state,
        revision=record.revision,
        previous_job_state=record.previous_job_state,
        start_operation_id=record.start_operation_id,
        pending_operation_id=record.pending_operation_id,
        runtime_key_version=record.runtime_key_version,
        durable_key_version=record.durable_key_version,
        execution_lease_deadline=record.execution_lease_deadline,
        barrier_id=record.barrier_id,
        receipt_id=record.receipt.receipt_id if record.receipt else None,
        release_target=record.release_target,
        release_terminal_outcome=record.release_terminal_outcome,
        failure_reason_code=record.failure_reason_code,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def project_job_snapshot(
    job: JobRecord, generation: GenerationRecord | None
) -> JobSnapshot:
    return JobSnapshot(
        job_id=job.job_id,
        conversation_id=job.conversation_id,
        owner=job.owner,
        profile=job.profile,
        runtime_driver=job.runtime_driver,
        durable_state_driver=job.durable_state_driver,
        state=job.state,
        revision=job.revision,
        generation=job.generation,
        current_generation=job.current_generation,
        pending_operation_id=job.pending_operation_id,
        durable_key_version=job.durable_key_version,
        terminal_outcome=job.terminal_outcome,
        created_at=job.created_at,
        updated_at=job.updated_at,
        generation_record=_project_generation(generation) if generation else None,
    )


def project_recovery_snapshot(
    job: JobRecord, generation: GenerationRecord | None
) -> RecoverySnapshot:
    recovery_generation = None
    if generation is not None:
        recovery_generation = RecoveryGenerationSnapshot(
            generation=generation.generation,
            state=generation.state,
            revision=generation.revision,
            previous_job_state=generation.previous_job_state,
            start_operation_id=generation.start_operation_id,
            pending_operation_id=generation.pending_operation_id,
            runtime_ref=generation.runtime_ref,
            durable_state_ref=generation.durable_state_ref,
            runtime_key_version=generation.runtime_key_version,
            durable_key_version=generation.durable_key_version,
            capability_digest=generation.capability_digest,
            durable_digest=generation.durable_digest,
            execution_lease_deadline=generation.execution_lease_deadline,
            barrier_id=generation.barrier_id,
            receipt=generation.receipt,
            release_target=generation.release_target,
            release_terminal_outcome=generation.release_terminal_outcome,
            failure_reason_code=generation.failure_reason_code,
            created_at=generation.created_at,
            updated_at=generation.updated_at,
        )
    return RecoverySnapshot(
        job_id=job.job_id,
        conversation_id=job.conversation_id,
        owner=job.owner,
        profile=job.profile,
        runtime_driver=job.runtime_driver,
        durable_state_driver=job.durable_state_driver,
        state=job.state,
        revision=job.revision,
        generation=job.generation,
        current_generation=job.current_generation,
        pending_operation_id=job.pending_operation_id,
        durable_state_ref=job.durable_state_ref,
        durable_key_version=job.durable_key_version,
        durable_digest=job.durable_digest,
        terminal_outcome=job.terminal_outcome,
        created_at=job.created_at,
        updated_at=job.updated_at,
        generation_record=recovery_generation,
        minimum_isolation=job.minimum_isolation,
        authorization=job.authorization,
    )
