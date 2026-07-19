"""Typed repository commands and pure lifecycle transition rules."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
from typing import Any, ClassVar, Protocol, runtime_checkable
from uuid import UUID

from .errors import InvalidTransition, TransitionEntity
from .models import (
    MAX_BROKER_JSON_BYTES,
    BrokerOwner,
    CommandKind,
    GenerationState,
    IsolationMode,
    JobAuthorizationMetadata,
    JobAuthorizationRecord,
    JobSnapshot,
    JobState,
    OperationResult,
    OperationTicket,
    RecoverySnapshot,
    RecoveryCandidate,
    ReleaseTarget,
    TerminalOutcome,
    VerifiedCheckpointReceipt,
    validate_bigint,
    validate_idempotency_key,
    validate_identifier,
    validate_lease_seconds,
    validate_opaque_ref,
    validate_positive_bigint,
    validate_sha256,
    validate_token,
    validate_uuid,
)


_NO_DIGEST = {"digest": False}
_OMIT_NONE_DIGEST = {"omit_none": True}


class _DigestCommand:
    kind: ClassVar[CommandKind]

    @property
    def request_digest(self) -> str:
        return hashlib.sha256(canonical_request_json(self).encode("utf-8")).hexdigest()


def _require_owner(owner: object) -> BrokerOwner:
    if not isinstance(owner, BrokerOwner):
        raise TypeError("owner must be a BrokerOwner")
    return owner


def _require_enum(name: str, value: object, enum_type: type[Any]) -> Any:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}")
    return value


def _canonical_value(value: object) -> object:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(
        value,
        (
            CommandKind,
            JobState,
            GenerationState,
            IsolationMode,
            ReleaseTarget,
            TerminalOutcome,
        ),
    ):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    if value is None or type(value) in {bool, int, str}:
        return value
    raise TypeError(f"unsupported canonical request value: {type(value).__name__}")


def canonical_request_json(command: _DigestCommand) -> str:
    if not isinstance(command, _DigestCommand):
        raise TypeError("command does not support canonical request digests")
    request = {
        item.name: _canonical_value(getattr(command, item.name))
        for item in fields(command)
        if item.metadata.get("digest", True)
        and not (
            item.metadata.get("omit_none", False)
            and getattr(command, item.name) is None
        )
    }
    encoded = json.dumps(
        {
            "schema_version": 1,
            "command": command.kind.value,
            "request": request,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > MAX_BROKER_JSON_BYTES:
        raise ValueError("canonical broker request exceeds 16 KiB")
    return encoded


@dataclass(frozen=True, slots=True)
class CreateJobCommand(_DigestCommand):
    kind: ClassVar[CommandKind] = CommandKind.CREATE_JOB

    owner: BrokerOwner
    idempotency_key: str = field(metadata=_NO_DIGEST)
    operation_id: UUID = field(metadata=_NO_DIGEST)
    job_id: UUID = field(metadata=_NO_DIGEST)
    conversation_id: UUID = field(metadata=_NO_DIGEST)
    profile: str
    runtime_driver: str
    durable_state_driver: str
    minimum_isolation: IsolationMode | None = field(
        default=None, metadata=_OMIT_NONE_DIGEST
    )
    authorization: JobAuthorizationMetadata | None = field(
        default=None, repr=False, metadata=_NO_DIGEST
    )

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_idempotency_key(self.idempotency_key)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_uuid("conversation_id", self.conversation_id)
        validate_token("profile", self.profile)
        validate_token("runtime_driver", self.runtime_driver)
        validate_token("durable_state_driver", self.durable_state_driver)
        if (self.minimum_isolation is None) != (self.authorization is None):
            raise ValueError(
                "create authorization fields must be all null or all present"
            )
        if self.minimum_isolation is not None and not isinstance(
            self.minimum_isolation, IsolationMode
        ):
            raise TypeError("minimum_isolation must be an IsolationMode")
        if self.authorization is not None and not isinstance(
            self.authorization, JobAuthorizationMetadata
        ):
            raise TypeError("authorization must be JobAuthorizationMetadata")


@dataclass(frozen=True, slots=True)
class BeginStartCommand(_DigestCommand):
    kind: ClassVar[CommandKind] = CommandKind.BEGIN_START

    owner: BrokerOwner
    idempotency_key: str = field(metadata=_NO_DIGEST)
    operation_id: UUID = field(metadata=_NO_DIGEST)
    job_id: UUID
    expected_generation: int
    execution_lease_seconds: int
    runtime_key_version: str = field(metadata=_NO_DIGEST)
    durable_state_ref: str = field(repr=False, metadata=_NO_DIGEST)
    durable_key_version: str = field(metadata=_NO_DIGEST)
    durable_digest: str = field(repr=False, metadata=_NO_DIGEST)

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_idempotency_key(self.idempotency_key)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        validate_lease_seconds(self.execution_lease_seconds)
        validate_identifier("runtime_key_version", self.runtime_key_version)
        validate_opaque_ref("durable_state_ref", self.durable_state_ref)
        validate_identifier("durable_key_version", self.durable_key_version)
        validate_sha256("durable_digest", self.durable_digest)


@dataclass(frozen=True, slots=True)
class MarkRunningCommand:
    kind: ClassVar[CommandKind] = CommandKind.MARK_RUNNING

    owner: BrokerOwner
    operation_id: UUID
    job_id: UUID
    generation: int
    runtime_ref: str = field(repr=False)
    capability_digest: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_positive_bigint("generation", self.generation)
        validate_opaque_ref("runtime_ref", self.runtime_ref)
        validate_sha256("capability_digest", self.capability_digest)


@dataclass(frozen=True, slots=True)
class BeginQuiesceCommand(_DigestCommand):
    kind: ClassVar[CommandKind] = CommandKind.BEGIN_QUIESCE

    owner: BrokerOwner
    idempotency_key: str = field(metadata=_NO_DIGEST)
    operation_id: UUID = field(metadata=_NO_DIGEST)
    job_id: UUID
    expected_generation: int
    barrier_id: str

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_idempotency_key(self.idempotency_key)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        validate_identifier("barrier_id", self.barrier_id)


@dataclass(frozen=True, slots=True)
class MarkQuiescedCommand:
    kind: ClassVar[CommandKind] = CommandKind.MARK_QUIESCED

    owner: BrokerOwner
    operation_id: UUID
    job_id: UUID
    generation: int

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_positive_bigint("generation", self.generation)


@dataclass(frozen=True, slots=True)
class BeginReleaseCommand(_DigestCommand):
    kind: ClassVar[CommandKind] = CommandKind.BEGIN_RELEASE

    owner: BrokerOwner
    idempotency_key: str = field(metadata=_NO_DIGEST)
    operation_id: UUID = field(metadata=_NO_DIGEST)
    job_id: UUID
    expected_generation: int
    receipt: VerifiedCheckpointReceipt = field(repr=False)
    target: ReleaseTarget
    terminal_outcome: TerminalOutcome | None

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_idempotency_key(self.idempotency_key)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        if not isinstance(self.receipt, VerifiedCheckpointReceipt):
            raise TypeError("receipt must be a VerifiedCheckpointReceipt")
        _require_enum("target", self.target, ReleaseTarget)
        if self.target is ReleaseTarget.FINALIZING:
            if self.terminal_outcome is None:
                raise ValueError("a finalizing release requires terminal_outcome")
        elif self.terminal_outcome is not None:
            raise ValueError("a parked release cannot set terminal_outcome")
        if self.terminal_outcome is not None:
            _require_enum("terminal_outcome", self.terminal_outcome, TerminalOutcome)


@dataclass(frozen=True, slots=True)
class BeginInternalReleaseCommand(_DigestCommand):
    kind: ClassVar[CommandKind] = CommandKind.BEGIN_RELEASE

    owner: BrokerOwner
    operation_id: UUID = field(metadata=_NO_DIGEST)
    job_id: UUID
    expected_generation: int
    receipt: VerifiedCheckpointReceipt = field(repr=False)
    target: ReleaseTarget
    terminal_outcome: TerminalOutcome | None

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        if not isinstance(self.receipt, VerifiedCheckpointReceipt):
            raise TypeError("receipt must be a VerifiedCheckpointReceipt")
        _require_enum("target", self.target, ReleaseTarget)
        if self.target is ReleaseTarget.FINALIZING:
            if self.terminal_outcome is None:
                raise ValueError("a finalizing release requires terminal_outcome")
        elif self.terminal_outcome is not None:
            raise ValueError("a parked release cannot set terminal_outcome")
        if self.terminal_outcome is not None:
            _require_enum("terminal_outcome", self.terminal_outcome, TerminalOutcome)


@dataclass(frozen=True, slots=True)
class MarkReleasedCommand:
    kind: ClassVar[CommandKind] = CommandKind.MARK_RELEASED

    owner: BrokerOwner
    operation_id: UUID
    job_id: UUID
    generation: int

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_positive_bigint("generation", self.generation)


@dataclass(frozen=True, slots=True)
class AbandonGenerationCommand(_DigestCommand):
    kind: ClassVar[CommandKind] = CommandKind.ABANDON_GENERATION

    owner: BrokerOwner
    operation_id: UUID = field(metadata=_NO_DIGEST)
    job_id: UUID
    generation: int
    expected_state: GenerationState
    reason_code: str
    terminal_outcome: TerminalOutcome | None
    replay_operation: bool = field(default=False, metadata=_NO_DIGEST)

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_positive_bigint("generation", self.generation)
        _require_enum("expected_state", self.expected_state, GenerationState)
        validate_token("reason_code", self.reason_code)
        if self.expected_state is GenerationState.STARTING:
            if self.terminal_outcome is not None:
                raise ValueError("starting abandonment cannot set terminal_outcome")
        elif self.terminal_outcome not in {
            TerminalOutcome.CANCELLED,
            TerminalOutcome.FAILED,
        }:
            raise ValueError(
                "active generation abandonment requires failed or cancelled outcome"
            )
        if type(self.replay_operation) is not bool:
            raise TypeError("replay_operation must be a bool")


@dataclass(frozen=True, slots=True)
class BeginFinalizeCommand(_DigestCommand):
    kind: ClassVar[CommandKind] = CommandKind.BEGIN_FINALIZE

    owner: BrokerOwner
    idempotency_key: str = field(metadata=_NO_DIGEST)
    operation_id: UUID = field(metadata=_NO_DIGEST)
    job_id: UUID
    expected_generation: int
    terminal_outcome: TerminalOutcome

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_idempotency_key(self.idempotency_key)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        _require_enum("terminal_outcome", self.terminal_outcome, TerminalOutcome)


@dataclass(frozen=True, slots=True)
class BeginInternalFinalizeCommand(_DigestCommand):
    kind: ClassVar[CommandKind] = CommandKind.BEGIN_FINALIZE

    owner: BrokerOwner
    operation_id: UUID = field(metadata=_NO_DIGEST)
    job_id: UUID
    expected_generation: int
    terminal_outcome: TerminalOutcome

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)
        validate_bigint("expected_generation", self.expected_generation)
        _require_enum("terminal_outcome", self.terminal_outcome, TerminalOutcome)


@dataclass(frozen=True, slots=True)
class MarkTerminalCommand:
    kind: ClassVar[CommandKind] = CommandKind.MARK_TERMINAL

    owner: BrokerOwner
    operation_id: UUID
    job_id: UUID

    def __post_init__(self) -> None:
        _require_owner(self.owner)
        validate_uuid("operation_id", self.operation_id)
        validate_uuid("job_id", self.job_id)


_JOB_TRANSITIONS = frozenset(
    {
        (CommandKind.BEGIN_START, JobState.CREATED, JobState.CREATED),
        (CommandKind.BEGIN_START, JobState.PARKED, JobState.PARKED),
        (CommandKind.MARK_RUNNING, JobState.CREATED, JobState.ACTIVE),
        (CommandKind.MARK_RUNNING, JobState.PARKED, JobState.ACTIVE),
        (CommandKind.ABANDON_GENERATION, JobState.CREATED, JobState.CREATED),
        (CommandKind.ABANDON_GENERATION, JobState.PARKED, JobState.PARKED),
        (CommandKind.ABANDON_GENERATION, JobState.ACTIVE, JobState.FINALIZING),
        (CommandKind.BEGIN_QUIESCE, JobState.ACTIVE, JobState.ACTIVE),
        (CommandKind.MARK_QUIESCED, JobState.ACTIVE, JobState.ACTIVE),
        (CommandKind.BEGIN_RELEASE, JobState.ACTIVE, JobState.ACTIVE),
        (CommandKind.MARK_RELEASED, JobState.ACTIVE, JobState.PARKED),
        (CommandKind.MARK_RELEASED, JobState.ACTIVE, JobState.FINALIZING),
        (CommandKind.BEGIN_FINALIZE, JobState.CREATED, JobState.FINALIZING),
        (CommandKind.BEGIN_FINALIZE, JobState.PARKED, JobState.FINALIZING),
        (CommandKind.BEGIN_FINALIZE, JobState.FINALIZING, JobState.FINALIZING),
        (CommandKind.MARK_TERMINAL, JobState.FINALIZING, JobState.TERMINAL),
    }
)

_GENERATION_TRANSITIONS = frozenset(
    {
        (CommandKind.MARK_RUNNING, GenerationState.STARTING, GenerationState.RUNNING),
        (
            CommandKind.BEGIN_QUIESCE,
            GenerationState.RUNNING,
            GenerationState.QUIESCING,
        ),
        (
            CommandKind.MARK_QUIESCED,
            GenerationState.QUIESCING,
            GenerationState.QUIESCED,
        ),
        (
            CommandKind.BEGIN_RELEASE,
            GenerationState.QUIESCED,
            GenerationState.RELEASING,
        ),
        (
            CommandKind.MARK_RELEASED,
            GenerationState.RELEASING,
            GenerationState.RELEASED,
        ),
        *{
            (CommandKind.ABANDON_GENERATION, state, GenerationState.ABANDONED)
            for state in (
                GenerationState.STARTING,
                GenerationState.RUNNING,
                GenerationState.QUIESCING,
                GenerationState.QUIESCED,
            )
        },
    }
)


def require_job_transition(
    command: CommandKind, current: JobState, target: JobState
) -> None:
    _require_enum("command", command, CommandKind)
    _require_enum("current", current, JobState)
    _require_enum("target", target, JobState)
    if (command, current, target) not in _JOB_TRANSITIONS:
        raise InvalidTransition(TransitionEntity.JOB, current, command)


def require_generation_transition(
    command: CommandKind, current: GenerationState, target: GenerationState
) -> None:
    _require_enum("command", command, CommandKind)
    _require_enum("current", current, GenerationState)
    _require_enum("target", target, GenerationState)
    if (command, current, target) not in _GENERATION_TRANSITIONS:
        raise InvalidTransition(TransitionEntity.GENERATION, current, command)


@runtime_checkable
class BrokerRepository(Protocol):
    async def scan_recovery_candidates(
        self, after_job_id: UUID | None, limit: int
    ) -> tuple[RecoveryCandidate, ...]: ...

    async def create_job(self, command: CreateJobCommand) -> OperationTicket: ...

    async def begin_start(self, command: BeginStartCommand) -> OperationTicket: ...

    async def mark_running(self, command: MarkRunningCommand) -> OperationResult: ...

    async def abandon_generation(
        self, command: AbandonGenerationCommand
    ) -> OperationResult: ...

    async def begin_quiesce(
        self, command: BeginQuiesceCommand
    ) -> OperationTicket: ...

    async def mark_quiesced(self, command: MarkQuiescedCommand) -> OperationResult: ...

    async def begin_release(
        self, command: BeginReleaseCommand
    ) -> OperationTicket: ...

    async def begin_internal_release(
        self, command: BeginInternalReleaseCommand
    ) -> OperationTicket: ...

    async def mark_released(self, command: MarkReleasedCommand) -> OperationResult: ...

    async def begin_finalize(
        self, command: BeginFinalizeCommand
    ) -> OperationTicket: ...

    async def begin_internal_finalize(
        self, command: BeginInternalFinalizeCommand
    ) -> OperationTicket: ...

    async def mark_terminal(self, command: MarkTerminalCommand) -> OperationResult: ...

    async def inspect_job(self, owner: BrokerOwner, job_id: UUID) -> JobSnapshot: ...

    async def inspect_job_authorization(
        self, owner: BrokerOwner, job_id: UUID
    ) -> JobAuthorizationRecord: ...

    async def inspect_job_for_recovery(
        self, owner: BrokerOwner, job_id: UUID
    ) -> RecoverySnapshot: ...
