"""Safe domain failures for the broker lifecycle ledger."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class MigrationProblem(_StringEnum):
    NUMBERING_GAP = "numbering_gap"
    CHECKSUM_DRIFT = "checksum_drift"
    FUTURE_VERSION = "future_version"
    MALFORMED_NAME = "malformed_name"
    DUPLICATE_VERSION = "duplicate_version"


class TransitionEntity(_StringEnum):
    JOB = "job"
    GENERATION = "generation"


class ReceiptField(_StringEnum):
    ISSUER = "issuer"
    RECEIPT_ID = "receipt_id"
    TENANT_ID = "tenant_id"
    JOB_ID = "job_id"
    CONVERSATION_ID = "conversation_id"
    GENERATION = "generation"
    BARRIER_ID = "barrier_id"
    ARTIFACT_ID = "artifact_id"
    BASE_COMMIT = "base_commit"
    CIPHERTEXT_SHA256 = "ciphertext_sha256"
    PLAINTEXT_SHA256 = "plaintext_sha256"
    BYTE_COUNT = "byte_count"
    STORE_VERSION = "store_version"
    ENVELOPE_VERSION = "envelope_version"
    KEY_VERSION = "key_version"
    DURABLE_WRITE_SEQUENCE = "durable_write_sequence"


class BrokerError(Exception):
    """Base class whose rendered form contains safe domain context only."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"


class JobNotFound(BrokerError):
    def __init__(self, job_id: UUID) -> None:
        self.job_id = job_id
        super().__init__(f"job {job_id} was not found")


class OwnerMismatch(BrokerError):
    def __init__(self, job_id: UUID) -> None:
        self.job_id = job_id
        super().__init__(f"owner does not match job {job_id}")


class StaleGeneration(BrokerError):
    def __init__(self, job_id: UUID, *, expected: int, actual: int) -> None:
        self.job_id = job_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"job {job_id} generation is stale: expected {expected}, actual {actual}"
        )


class InvalidTransition(BrokerError):
    def __init__(self, entity: TransitionEntity, state: Any, command: Any) -> None:
        if not isinstance(entity, TransitionEntity):
            raise TypeError("entity must be a TransitionEntity")
        if not isinstance(state, Enum) or not isinstance(command, Enum):
            raise TypeError("state and command must be enums")
        self.entity = entity
        self.state = state
        self.command = command
        state_value = state.value
        command_value = command.value
        super().__init__(
            f"{entity.value} cannot apply {command_value} from state {state_value}"
        )


class IdempotencyConflict(BrokerError):
    def __init__(self) -> None:
        super().__init__("idempotency key was already used for another request")


class OperationMismatch(BrokerError):
    def __init__(self, operation_id: UUID) -> None:
        self.operation_id = operation_id
        super().__init__(f"operation {operation_id} does not match the transition")


class ReceiptBindingMismatch(BrokerError):
    def __init__(self, field: ReceiptField) -> None:
        if not isinstance(field, ReceiptField):
            raise TypeError("field must be a ReceiptField")
        self.field = field
        super().__init__(f"checkpoint receipt does not match field {field.value}")


class ConcurrentMutation(BrokerError):
    def __init__(self, job_id: UUID) -> None:
        self.job_id = job_id
        super().__init__(f"job {job_id} changed concurrently")


class MigrationVersionError(BrokerError):
    def __init__(self, version: int, problem: MigrationProblem) -> None:
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("version must be a nonnegative integer")
        if not isinstance(problem, MigrationProblem):
            raise TypeError("problem must be a MigrationProblem")
        self.version = version
        self.problem = problem
        super().__init__(f"broker migration {version}: {problem.value}")
