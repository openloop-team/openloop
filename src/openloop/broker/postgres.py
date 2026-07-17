"""PostgreSQL-backed broker repository and append-only migration runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
from importlib import resources
import json
import re
from typing import Any
from uuid import UUID

from openloop.postgres import BorrowedPostgresStore

from .errors import (
    ConcurrentMutation,
    IdempotencyConflict,
    InvalidTransition,
    JobNotFound,
    MigrationProblem,
    MigrationVersionError,
    OperationMismatch,
    OwnerMismatch,
    ReceiptBindingMismatch,
    ReceiptField,
    StaleGeneration,
    TransitionEntity,
)
from .models import (
    BrokerOwner,
    CommandKind,
    GenerationRecord,
    GenerationState,
    JobRecord,
    JobState,
    OperationRecord,
    OperationResult,
    OperationSource,
    OperationStatus,
    OperationTicket,
    ReleaseTarget,
    TerminalOutcome,
    VerifiedCheckpointReceipt,
    project_job_snapshot,
    project_recovery_snapshot,
)
from .repository import (
    AbandonGenerationCommand,
    BeginFinalizeCommand,
    BeginQuiesceCommand,
    BeginReleaseCommand,
    BeginStartCommand,
    CreateJobCommand,
    MarkQuiescedCommand,
    MarkReleasedCommand,
    MarkRunningCommand,
    MarkTerminalCommand,
    require_generation_transition,
    require_job_transition,
)


# ASCII "BROKERLD" interpreted as a positive signed 64-bit advisory-lock key.
BROKER_MIGRATION_LOCK_ID = 0x42524F4B45524C44

_MIGRATION_NAME = re.compile(r"(?P<version>[0-9]{4})_(?P<name>[a-z0-9_]+)\.sql\Z")

_BOOTSTRAP_MIGRATION_TABLE = """
CREATE TABLE IF NOT EXISTS broker_schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    name TEXT NOT NULL,
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
)
"""

_READ_APPLIED_MIGRATIONS = """
SELECT version, name, checksum
FROM broker_schema_migrations
ORDER BY version
"""

_RECORD_MIGRATION = """
INSERT INTO broker_schema_migrations (version, name, checksum, applied_at)
VALUES ($1, $2, $3, clock_timestamp())
"""


def _json_object(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        decoded = json.loads(value)
    elif isinstance(value, dict):
        decoded = value
    else:
        raise TypeError("broker JSON value must be an object or encoded object")
    if not isinstance(decoded, dict):
        raise TypeError("broker JSON value must decode to an object")
    return decoded


def _encode_ticket(ticket: OperationTicket) -> str:
    return json.dumps(
        {
            "operation_id": str(ticket.operation_id),
            "command": ticket.command.value,
            "job_id": str(ticket.job_id) if ticket.job_id else None,
            "conversation_id": (
                str(ticket.conversation_id) if ticket.conversation_id else None
            ),
            "generation": ticket.generation,
            "job_state": ticket.job_state.value if ticket.job_state else None,
            "generation_state": (
                ticket.generation_state.value if ticket.generation_state else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_ticket(value: str | dict[str, Any]) -> OperationTicket:
    payload = _json_object(value)
    return OperationTicket(
        operation_id=UUID(payload["operation_id"]),
        command=CommandKind(payload["command"]),
        job_id=UUID(payload["job_id"]) if payload.get("job_id") else None,
        conversation_id=(
            UUID(payload["conversation_id"])
            if payload.get("conversation_id")
            else None
        ),
        generation=payload.get("generation"),
        job_state=JobState(payload["job_state"]) if payload.get("job_state") else None,
        generation_state=(
            GenerationState(payload["generation_state"])
            if payload.get("generation_state")
            else None
        ),
    )


def _encode_result(result: OperationResult) -> str:
    return json.dumps(
        {
            "operation_id": str(result.operation_id),
            "command": result.command.value,
            "job_id": str(result.job_id),
            "generation": result.generation,
            "job_state": result.job_state.value,
            "generation_state": (
                result.generation_state.value if result.generation_state else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_result(value: str | dict[str, Any]) -> OperationResult:
    payload = _json_object(value)
    return OperationResult(
        operation_id=UUID(payload["operation_id"]),
        command=CommandKind(payload["command"]),
        job_id=UUID(payload["job_id"]),
        generation=payload.get("generation"),
        job_state=JobState(payload["job_state"]),
        generation_state=(
            GenerationState(payload["generation_state"])
            if payload.get("generation_state")
            else None
        ),
    )


def _job_from_row(row: Any) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        conversation_id=row["conversation_id"],
        owner=BrokerOwner(row["tenant_id"], row["workload_subject"]),
        profile=row["profile"],
        runtime_driver=row["runtime_driver"],
        durable_state_driver=row["durable_state_driver"],
        state=JobState(row["state"]),
        revision=int(row["revision"]),
        generation=int(row["generation"]),
        current_generation=(
            int(row["current_generation"])
            if row["current_generation"] is not None
            else None
        ),
        pending_operation_id=row["pending_operation_id"],
        durable_state_ref=row["durable_state_ref"],
        durable_key_version=row["durable_key_version"],
        durable_digest=row["durable_digest"],
        terminal_outcome=(
            TerminalOutcome(row["terminal_outcome"])
            if row["terminal_outcome"] is not None
            else None
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _receipt_from_row(row: Any) -> VerifiedCheckpointReceipt | None:
    if row["receipt_id"] is None:
        return None
    return VerifiedCheckpointReceipt(
        issuer=row["receipt_issuer"],
        receipt_id=row["receipt_id"],
        tenant_id=row["receipt_tenant_id"],
        job_id=row["receipt_job_id"],
        conversation_id=row["receipt_conversation_id"],
        generation=int(row["receipt_generation"]),
        barrier_id=row["receipt_barrier_id"],
        artifact_id=row["receipt_artifact_id"],
        base_commit=row["receipt_base_commit"],
        ciphertext_sha256=row["receipt_ciphertext_sha256"],
        plaintext_sha256=row["receipt_plaintext_sha256"],
        byte_count=int(row["receipt_byte_count"]),
        store_version=row["receipt_store_version"],
        envelope_version=row["receipt_envelope_version"],
        key_version=row["receipt_key_version"],
        durable_write_sequence=int(row["receipt_durable_write_sequence"]),
    )


def _generation_from_row(row: Any) -> GenerationRecord:
    return GenerationRecord(
        job_id=row["job_id"],
        generation=int(row["generation"]),
        state=GenerationState(row["state"]),
        revision=int(row["revision"]),
        previous_job_state=JobState(row["previous_job_state"]),
        start_operation_id=row["start_operation_id"],
        pending_operation_id=row["pending_operation_id"],
        runtime_ref=row["runtime_ref"],
        durable_state_ref=row["durable_state_ref"],
        runtime_key_version=row["runtime_key_version"],
        durable_key_version=row["durable_key_version"],
        capability_digest=row["capability_digest"],
        durable_digest=row["durable_digest"],
        execution_lease_deadline=row["execution_lease_deadline"],
        barrier_id=row["barrier_id"],
        receipt=_receipt_from_row(row),
        release_target=(
            ReleaseTarget(row["release_target"])
            if row["release_target"] is not None
            else None
        ),
        release_terminal_outcome=(
            TerminalOutcome(row["release_terminal_outcome"])
            if row["release_terminal_outcome"] is not None
            else None
        ),
        failure_reason_code=row["failure_reason_code"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _operation_from_row(row: Any) -> OperationRecord:
    return OperationRecord(
        operation_id=row["operation_id"],
        owner=BrokerOwner(row["tenant_id"], row["workload_subject"]),
        source=OperationSource(row["source"]),
        idempotency_key=row["idempotency_key"],
        command=CommandKind(row["command_kind"]),
        request_digest=row["request_digest"],
        job_id=row["job_id"],
        generation=int(row["generation"]) if row["generation"] is not None else None,
        status=OperationStatus(row["status"]),
        intent_ticket=_decode_ticket(row["intent_ticket"]),
        completion_result=(
            _decode_result(row["completion_result"])
            if row["completion_result"] is not None
            else None
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


_INSERT_OPERATION = """
INSERT INTO broker_operations (
    operation_id, tenant_id, workload_subject, source, idempotency_key,
    command_kind, request_digest, job_id, generation, status, intent_ticket,
    completion_result, created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, NULL, $12, $12)
ON CONFLICT DO NOTHING
RETURNING *
"""

_SELECT_CALLER_OPERATION = """
SELECT * FROM broker_operations
WHERE tenant_id = $1 AND workload_subject = $2 AND idempotency_key = $3
FOR UPDATE
"""

_SELECT_OPERATION = """
SELECT * FROM broker_operations WHERE operation_id = $1 FOR UPDATE
"""

_UPDATE_OPERATION_TICKET = """
UPDATE broker_operations
SET intent_ticket = $2::jsonb, generation = $3, updated_at = $4
WHERE operation_id = $1
RETURNING *
"""

_COMPLETE_OPERATION = """
UPDATE broker_operations
SET status = 'completed', completion_result = $2::jsonb, updated_at = $3
WHERE operation_id = $1 AND status = 'pending'
RETURNING *
"""

_COMPLETE_OPERATION_WITHOUT_RESULT = """
UPDATE broker_operations
SET status = 'completed', updated_at = $2
WHERE operation_id = $1 AND status = 'pending'
RETURNING *
"""

_FAIL_OPERATION = """
UPDATE broker_operations
SET status = 'failed', updated_at = $2
WHERE operation_id = $1 AND status = 'pending'
RETURNING *
"""

_SELECT_JOB = "SELECT * FROM broker_jobs WHERE job_id = $1 FOR UPDATE"
_SELECT_JOB_READ = "SELECT * FROM broker_jobs WHERE job_id = $1"

_INSERT_JOB = """
INSERT INTO broker_jobs (
    job_id, conversation_id, tenant_id, workload_subject, profile,
    runtime_driver, durable_state_driver, state, revision, generation,
    current_generation, pending_operation_id, durable_state_ref,
    durable_key_version, durable_digest, terminal_outcome, created_at, updated_at
)
VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18
)
ON CONFLICT DO NOTHING
RETURNING *
"""

_UPDATE_JOB = """
UPDATE broker_jobs
SET state = $2, revision = $3, generation = $4, current_generation = $5,
    pending_operation_id = $6, durable_state_ref = $7,
    durable_key_version = $8, durable_digest = $9, terminal_outcome = $10,
    updated_at = $11
WHERE job_id = $1 AND revision = $12
RETURNING *
"""

_SELECT_GENERATION = """
SELECT * FROM broker_generations
WHERE job_id = $1 AND generation = $2
FOR UPDATE
"""

_SELECT_GENERATION_READ = """
SELECT * FROM broker_generations WHERE job_id = $1 AND generation = $2
"""

_INSERT_GENERATION = """
INSERT INTO broker_generations (
    job_id, generation, state, revision, previous_job_state,
    start_operation_id, pending_operation_id, runtime_ref, durable_state_ref,
    runtime_key_version, durable_key_version, capability_digest, durable_digest,
    execution_lease_deadline, barrier_id, receipt_issuer, receipt_id,
    receipt_tenant_id, receipt_job_id, receipt_conversation_id,
    receipt_generation, receipt_barrier_id, receipt_artifact_id,
    receipt_base_commit, receipt_ciphertext_sha256, receipt_plaintext_sha256,
    receipt_byte_count, receipt_store_version, receipt_envelope_version,
    receipt_key_version, receipt_durable_write_sequence, release_target,
    release_terminal_outcome, failure_reason_code, created_at, updated_at
)
VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
    $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
    $29, $30, $31, $32, $33, $34, $35, $36
)
ON CONFLICT DO NOTHING
RETURNING *
"""

_UPDATE_GENERATION = """
UPDATE broker_generations
SET state = $3, revision = $4, pending_operation_id = $5,
    runtime_ref = $6, durable_state_ref = $7, runtime_key_version = $8,
    durable_key_version = $9, capability_digest = $10, durable_digest = $11,
    execution_lease_deadline = $12, barrier_id = $13,
    receipt_issuer = $14, receipt_id = $15, receipt_tenant_id = $16,
    receipt_job_id = $17, receipt_conversation_id = $18,
    receipt_generation = $19, receipt_barrier_id = $20,
    receipt_artifact_id = $21, receipt_base_commit = $22,
    receipt_ciphertext_sha256 = $23, receipt_plaintext_sha256 = $24,
    receipt_byte_count = $25, receipt_store_version = $26,
    receipt_envelope_version = $27, receipt_key_version = $28,
    receipt_durable_write_sequence = $29, release_target = $30,
    release_terminal_outcome = $31, failure_reason_code = $32, updated_at = $33
WHERE job_id = $1 AND generation = $2 AND revision = $34
RETURNING *
"""

_INSERT_AUDIT = """
INSERT INTO broker_audit (
    command_kind, tenant_id, workload_subject, job_id, generation, operation_id,
    before_job_state, after_job_state, before_generation_state,
    after_generation_state, reason_code, created_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
"""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str

    @classmethod
    def from_bytes(cls, version: int, name: str, content: bytes) -> "Migration":
        return cls(
            version=version,
            name=name,
            sql=content.decode("utf-8"),
            checksum=hashlib.sha256(content).hexdigest(),
        )


def discover_migrations(root: Any) -> tuple[Migration, ...]:
    discovered: list[Migration] = []
    versions: set[int] = set()
    for entry in root.iterdir():
        name = entry.name
        if not name.lower().endswith(".sql"):
            continue
        matched = _MIGRATION_NAME.fullmatch(name)
        if matched is None:
            raise MigrationVersionError(0, MigrationProblem.MALFORMED_NAME)
        version = int(matched.group("version"))
        if version == 0:
            raise MigrationVersionError(0, MigrationProblem.MALFORMED_NAME)
        if version in versions:
            raise MigrationVersionError(version, MigrationProblem.DUPLICATE_VERSION)
        versions.add(version)
        discovered.append(
            Migration.from_bytes(version, matched.group("name"), entry.read_bytes())
        )
    discovered.sort(key=lambda item: item.version)
    for expected, migration in enumerate(discovered, start=1):
        if migration.version != expected:
            raise MigrationVersionError(expected, MigrationProblem.NUMBERING_GAP)
    return tuple(discovered)


def _load_packaged_migrations() -> tuple[Migration, ...]:
    return discover_migrations(resources.files("openloop.broker.migrations"))


class PostgresBrokerRepository(BorrowedPostgresStore):
    async def setup(self, pool: Any) -> None:
        async with self._setup_connection(pool) as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock($1)", BROKER_MIGRATION_LOCK_ID
                )
                await connection.execute(_BOOTSTRAP_MIGRATION_TABLE)
                migrations = _load_packaged_migrations()
                applied_rows = await connection.fetch(_READ_APPLIED_MIGRATIONS)
                self._validate_applied(migrations, applied_rows)
                applied_versions = {int(row["version"]) for row in applied_rows}
                for migration in migrations:
                    if migration.version in applied_versions:
                        continue
                    await connection.execute(migration.sql)
                    await connection.execute(
                        _RECORD_MIGRATION,
                        migration.version,
                        migration.name,
                        migration.checksum,
                    )

    @staticmethod
    async def _database_now(connection: Any) -> datetime:
        return await connection.fetchval("SELECT clock_timestamp()")

    @staticmethod
    async def _acquire_caller_operation(
        connection: Any,
        command: Any,
        ticket: OperationTicket,
        now: datetime,
    ) -> tuple[OperationRecord, OperationTicket | None]:
        row = await connection.fetchrow(
            _INSERT_OPERATION,
            command.operation_id,
            command.owner.tenant_id,
            command.owner.workload_subject,
            OperationSource.CALLER.value,
            command.idempotency_key,
            command.kind.value,
            command.request_digest,
            command.job_id,
            ticket.generation,
            OperationStatus.PENDING.value,
            _encode_ticket(ticket),
            now,
        )
        if row is not None:
            return _operation_from_row(row), None
        row = await connection.fetchrow(
            _SELECT_CALLER_OPERATION,
            command.owner.tenant_id,
            command.owner.workload_subject,
            command.idempotency_key,
        )
        if row is None:
            raise ConcurrentMutation(command.job_id)
        operation = _operation_from_row(row)
        if (
            operation.command is not command.kind
            or operation.request_digest != command.request_digest
        ):
            raise IdempotencyConflict()
        return operation, replace(operation.intent_ticket, replayed=True)

    @staticmethod
    async def _acquire_internal_operation(
        connection: Any,
        command: AbandonGenerationCommand,
        ticket: OperationTicket,
        now: datetime,
    ) -> tuple[OperationRecord, OperationResult | None]:
        row = await connection.fetchrow(_SELECT_OPERATION, command.operation_id)
        if row is not None:
            operation = _operation_from_row(row)
            if (
                operation.source is not OperationSource.INTERNAL
                or operation.command is not command.kind
                or operation.owner != command.owner
                or operation.job_id != command.job_id
                or operation.generation != command.generation
                or operation.request_digest != command.request_digest
                or operation.status is not OperationStatus.COMPLETED
                or operation.completion_result is None
            ):
                raise OperationMismatch(command.operation_id)
            return operation, replace(operation.completion_result, replayed=True)
        if command.replay_operation:
            raise OperationMismatch(command.operation_id)
        row = await connection.fetchrow(
            _INSERT_OPERATION,
            command.operation_id,
            command.owner.tenant_id,
            command.owner.workload_subject,
            OperationSource.INTERNAL.value,
            None,
            command.kind.value,
            command.request_digest,
            command.job_id,
            command.generation,
            OperationStatus.PENDING.value,
            _encode_ticket(ticket),
            now,
        )
        if row is None:
            raise ConcurrentMutation(command.job_id)
        return _operation_from_row(row), None

    @staticmethod
    async def _job(
        connection: Any, owner: BrokerOwner, job_id: UUID, *, lock: bool = True
    ) -> JobRecord:
        row = await connection.fetchrow(_SELECT_JOB if lock else _SELECT_JOB_READ, job_id)
        if row is None:
            raise JobNotFound(job_id)
        job = _job_from_row(row)
        if job.owner != owner:
            raise OwnerMismatch(job_id)
        return job

    @staticmethod
    async def _generation(
        connection: Any, job_id: UUID, generation: int, *, lock: bool = True
    ) -> GenerationRecord:
        row = await connection.fetchrow(
            _SELECT_GENERATION if lock else _SELECT_GENERATION_READ,
            job_id,
            generation,
        )
        if row is None:
            raise StaleGeneration(job_id, expected=generation, actual=0)
        return _generation_from_row(row)

    @staticmethod
    def _expected_generation(job: JobRecord, expected: int) -> None:
        if job.generation != expected:
            raise StaleGeneration(job.job_id, expected=expected, actual=job.generation)

    @staticmethod
    def _invalid_job(job: JobRecord, command: CommandKind) -> InvalidTransition:
        return InvalidTransition(TransitionEntity.JOB, job.state, command)

    @staticmethod
    async def _insert_job(connection: Any, job: JobRecord) -> None:
        row = await connection.fetchrow(
            _INSERT_JOB,
            job.job_id,
            job.conversation_id,
            job.owner.tenant_id,
            job.owner.workload_subject,
            job.profile,
            job.runtime_driver,
            job.durable_state_driver,
            job.state.value,
            job.revision,
            job.generation,
            job.current_generation,
            job.pending_operation_id,
            job.durable_state_ref,
            job.durable_key_version,
            job.durable_digest,
            job.terminal_outcome.value if job.terminal_outcome else None,
            job.created_at,
            job.updated_at,
        )
        if row is None:
            raise ConcurrentMutation(job.job_id)

    @staticmethod
    async def _update_job(
        connection: Any, before: JobRecord, after: JobRecord
    ) -> None:
        row = await connection.fetchrow(
            _UPDATE_JOB,
            after.job_id,
            after.state.value,
            after.revision,
            after.generation,
            after.current_generation,
            after.pending_operation_id,
            after.durable_state_ref,
            after.durable_key_version,
            after.durable_digest,
            after.terminal_outcome.value if after.terminal_outcome else None,
            after.updated_at,
            before.revision,
        )
        if row is None:
            raise ConcurrentMutation(after.job_id)

    @staticmethod
    def _receipt_values(
        receipt: VerifiedCheckpointReceipt | None,
    ) -> tuple[Any, ...]:
        if receipt is None:
            return (None,) * 16
        return (
            receipt.issuer,
            receipt.receipt_id,
            receipt.tenant_id,
            receipt.job_id,
            receipt.conversation_id,
            receipt.generation,
            receipt.barrier_id,
            receipt.artifact_id,
            receipt.base_commit,
            receipt.ciphertext_sha256,
            receipt.plaintext_sha256,
            receipt.byte_count,
            receipt.store_version,
            receipt.envelope_version,
            receipt.key_version,
            receipt.durable_write_sequence,
        )

    @classmethod
    async def _insert_generation(
        cls, connection: Any, generation: GenerationRecord
    ) -> None:
        row = await connection.fetchrow(
            _INSERT_GENERATION,
            generation.job_id,
            generation.generation,
            generation.state.value,
            generation.revision,
            generation.previous_job_state.value,
            generation.start_operation_id,
            generation.pending_operation_id,
            generation.runtime_ref,
            generation.durable_state_ref,
            generation.runtime_key_version,
            generation.durable_key_version,
            generation.capability_digest,
            generation.durable_digest,
            generation.execution_lease_deadline,
            generation.barrier_id,
            *cls._receipt_values(generation.receipt),
            generation.release_target.value if generation.release_target else None,
            (
                generation.release_terminal_outcome.value
                if generation.release_terminal_outcome
                else None
            ),
            generation.failure_reason_code,
            generation.created_at,
            generation.updated_at,
        )
        if row is None:
            raise ConcurrentMutation(generation.job_id)

    @classmethod
    async def _update_generation(
        cls,
        connection: Any,
        before: GenerationRecord,
        after: GenerationRecord,
    ) -> None:
        row = await connection.fetchrow(
            _UPDATE_GENERATION,
            after.job_id,
            after.generation,
            after.state.value,
            after.revision,
            after.pending_operation_id,
            after.runtime_ref,
            after.durable_state_ref,
            after.runtime_key_version,
            after.durable_key_version,
            after.capability_digest,
            after.durable_digest,
            after.execution_lease_deadline,
            after.barrier_id,
            *cls._receipt_values(after.receipt),
            after.release_target.value if after.release_target else None,
            (
                after.release_terminal_outcome.value
                if after.release_terminal_outcome
                else None
            ),
            after.failure_reason_code,
            after.updated_at,
            before.revision,
        )
        if row is None:
            raise ConcurrentMutation(after.job_id)

    @staticmethod
    async def _update_operation_ticket(
        connection: Any,
        operation_id: UUID,
        ticket: OperationTicket,
        now: datetime,
    ) -> OperationRecord:
        row = await connection.fetchrow(
            _UPDATE_OPERATION_TICKET,
            operation_id,
            _encode_ticket(ticket),
            ticket.generation,
            now,
        )
        if row is None:
            raise OperationMismatch(operation_id)
        return _operation_from_row(row)

    @staticmethod
    async def _complete_operation(
        connection: Any, result: OperationResult, now: datetime
    ) -> None:
        row = await connection.fetchrow(
            _COMPLETE_OPERATION,
            result.operation_id,
            _encode_result(result),
            now,
        )
        if row is None:
            raise OperationMismatch(result.operation_id)

    @staticmethod
    async def _complete_operation_without_result(
        connection: Any, operation_id: UUID, now: datetime
    ) -> None:
        row = await connection.fetchrow(
            _COMPLETE_OPERATION_WITHOUT_RESULT, operation_id, now
        )
        if row is None:
            raise OperationMismatch(operation_id)

    @staticmethod
    async def _fail_operation(
        connection: Any, operation_id: UUID, now: datetime
    ) -> None:
        row = await connection.fetchrow(_FAIL_OPERATION, operation_id, now)
        if row is None:
            raise OperationMismatch(operation_id)

    @staticmethod
    async def _audit(
        connection: Any,
        *,
        command: CommandKind,
        owner: BrokerOwner,
        job_id: UUID,
        generation: int | None,
        operation_id: UUID,
        before_job_state: JobState | None,
        after_job_state: JobState,
        before_generation_state: GenerationState | None,
        after_generation_state: GenerationState | None,
        reason_code: str | None,
        now: datetime,
    ) -> None:
        await connection.execute(
            _INSERT_AUDIT,
            command.value,
            owner.tenant_id,
            owner.workload_subject,
            job_id,
            generation,
            operation_id,
            before_job_state.value if before_job_state else None,
            after_job_state.value,
            before_generation_state.value if before_generation_state else None,
            after_generation_state.value if after_generation_state else None,
            reason_code,
            now,
        )

    @staticmethod
    async def _pending_completion(
        connection: Any,
        *,
        owner: BrokerOwner,
        operation_id: UUID,
        job_id: UUID,
        generation: int | None,
        expected_command: CommandKind,
    ) -> tuple[OperationRecord, OperationResult | None]:
        row = await connection.fetchrow(_SELECT_OPERATION, operation_id)
        if row is None:
            raise OperationMismatch(operation_id)
        operation = _operation_from_row(row)
        if (
            operation.owner != owner
            or operation.job_id != job_id
            or operation.generation != generation
            or operation.command is not expected_command
        ):
            raise OperationMismatch(operation_id)
        if operation.status is OperationStatus.COMPLETED:
            if operation.completion_result is None:
                raise OperationMismatch(operation_id)
            return operation, replace(operation.completion_result, replayed=True)
        if operation.status is not OperationStatus.PENDING:
            raise OperationMismatch(operation_id)
        return operation, None

    async def create_job(self, command: CreateJobCommand) -> OperationTicket:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                now = await self._database_now(connection)
                ticket = OperationTicket(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=command.job_id,
                    conversation_id=command.conversation_id,
                    job_state=JobState.CREATED,
                )
                _, replay = await self._acquire_caller_operation(
                    connection, command, ticket, now
                )
                if replay is not None:
                    return replay
                job = JobRecord(
                    job_id=command.job_id,
                    conversation_id=command.conversation_id,
                    owner=command.owner,
                    profile=command.profile,
                    runtime_driver=command.runtime_driver,
                    durable_state_driver=command.durable_state_driver,
                    state=JobState.CREATED,
                    revision=1,
                    generation=0,
                    current_generation=None,
                    pending_operation_id=None,
                    durable_state_ref=None,
                    durable_key_version=None,
                    durable_digest=None,
                    terminal_outcome=None,
                    created_at=now,
                    updated_at=now,
                )
                await self._insert_job(connection, job)
                await self._complete_operation_without_result(
                    connection, command.operation_id, now
                )
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=command.job_id,
                    generation=None,
                    operation_id=command.operation_id,
                    before_job_state=None,
                    after_job_state=JobState.CREATED,
                    before_generation_state=None,
                    after_generation_state=None,
                    reason_code=None,
                    now=now,
                )
                return ticket

    async def begin_start(self, command: BeginStartCommand) -> OperationTicket:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                operation_time = await self._database_now(connection)
                provisional = OperationTicket(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=command.job_id,
                    generation=command.expected_generation + 1,
                    generation_state=GenerationState.STARTING,
                )
                _, replay = await self._acquire_caller_operation(
                    connection, command, provisional, operation_time
                )
                if replay is not None:
                    return replay
                job = await self._job(connection, command.owner, command.job_id)
                self._expected_generation(job, command.expected_generation)
                require_job_transition(command.kind, job.state, job.state)
                if (
                    job.pending_operation_id is not None
                    or job.current_generation is not None
                ):
                    raise self._invalid_job(job, command.kind)
                now = await self._database_now(connection)
                generation_number = job.generation + 1
                generation = GenerationRecord(
                    job_id=job.job_id,
                    generation=generation_number,
                    state=GenerationState.STARTING,
                    revision=1,
                    previous_job_state=job.state,
                    start_operation_id=command.operation_id,
                    pending_operation_id=command.operation_id,
                    runtime_ref=None,
                    durable_state_ref=None,
                    runtime_key_version=None,
                    durable_key_version=None,
                    capability_digest=None,
                    durable_digest=None,
                    execution_lease_deadline=now
                    + timedelta(seconds=command.execution_lease_seconds),
                    barrier_id=None,
                    receipt=None,
                    release_target=None,
                    release_terminal_outcome=None,
                    failure_reason_code=None,
                    created_at=now,
                    updated_at=now,
                )
                updated_job = replace(
                    job,
                    revision=job.revision + 1,
                    generation=generation_number,
                    pending_operation_id=command.operation_id,
                    updated_at=now,
                )
                ticket = replace(
                    provisional,
                    conversation_id=job.conversation_id,
                    generation=generation_number,
                    job_state=job.state,
                )
                await self._update_operation_ticket(
                    connection, command.operation_id, ticket, now
                )
                await self._insert_generation(connection, generation)
                await self._update_job(connection, job, updated_job)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=generation_number,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=job.state,
                    before_generation_state=None,
                    after_generation_state=GenerationState.STARTING,
                    reason_code=None,
                    now=now,
                )
                return ticket

    async def mark_running(self, command: MarkRunningCommand) -> OperationResult:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                operation, replay = await self._pending_completion(
                    connection,
                    owner=command.owner,
                    operation_id=command.operation_id,
                    job_id=command.job_id,
                    generation=command.generation,
                    expected_command=CommandKind.BEGIN_START,
                )
                if replay is not None:
                    generation = await self._generation(
                        connection,
                        command.job_id,
                        command.generation,
                        lock=False,
                    )
                    if (
                        generation.runtime_ref != command.runtime_ref
                        or generation.durable_state_ref != command.durable_state_ref
                        or generation.runtime_key_version
                        != command.runtime_key_version
                        or generation.durable_key_version
                        != command.durable_key_version
                        or generation.capability_digest != command.capability_digest
                        or generation.durable_digest != command.durable_digest
                    ):
                        raise OperationMismatch(command.operation_id)
                    return replay
                job = await self._job(connection, command.owner, command.job_id)
                generation = await self._generation(
                    connection, command.job_id, command.generation
                )
                if (
                    job.generation != command.generation
                    or job.pending_operation_id != command.operation_id
                    or generation.pending_operation_id != command.operation_id
                ):
                    raise OperationMismatch(command.operation_id)
                require_generation_transition(
                    command.kind, generation.state, GenerationState.RUNNING
                )
                require_job_transition(command.kind, job.state, JobState.ACTIVE)
                now = await self._database_now(connection)
                updated_generation = replace(
                    generation,
                    state=GenerationState.RUNNING,
                    revision=generation.revision + 1,
                    pending_operation_id=None,
                    runtime_ref=command.runtime_ref,
                    durable_state_ref=command.durable_state_ref,
                    runtime_key_version=command.runtime_key_version,
                    durable_key_version=command.durable_key_version,
                    capability_digest=command.capability_digest,
                    durable_digest=command.durable_digest,
                    updated_at=now,
                )
                updated_job = replace(
                    job,
                    state=JobState.ACTIVE,
                    revision=job.revision + 1,
                    current_generation=command.generation,
                    pending_operation_id=None,
                    durable_state_ref=command.durable_state_ref,
                    durable_key_version=command.durable_key_version,
                    durable_digest=command.durable_digest,
                    updated_at=now,
                )
                result = OperationResult(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=job.job_id,
                    generation=command.generation,
                    job_state=JobState.ACTIVE,
                    generation_state=GenerationState.RUNNING,
                )
                await self._update_generation(
                    connection, generation, updated_generation
                )
                await self._update_job(connection, job, updated_job)
                await self._complete_operation(connection, result, now)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=command.generation,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=JobState.ACTIVE,
                    before_generation_state=generation.state,
                    after_generation_state=GenerationState.RUNNING,
                    reason_code=None,
                    now=now,
                )
                return result

    async def abandon_generation(
        self, command: AbandonGenerationCommand
    ) -> OperationResult:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                operation_time = await self._database_now(connection)
                provisional = OperationTicket(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=command.job_id,
                    generation=command.generation,
                    generation_state=command.expected_state,
                )
                _, replay = await self._acquire_internal_operation(
                    connection, command, provisional, operation_time
                )
                if replay is not None:
                    return replay
                job = await self._job(connection, command.owner, command.job_id)
                if job.generation != command.generation:
                    raise StaleGeneration(
                        job.job_id,
                        expected=command.generation,
                        actual=job.generation,
                    )
                generation = await self._generation(
                    connection, command.job_id, command.generation
                )
                if generation.state is not command.expected_state:
                    raise InvalidTransition(
                        TransitionEntity.GENERATION, generation.state, command.kind
                    )
                require_generation_transition(
                    command.kind, generation.state, GenerationState.ABANDONED
                )
                target_state = (
                    generation.previous_job_state
                    if generation.state is GenerationState.STARTING
                    else JobState.FINALIZING
                )
                require_job_transition(command.kind, job.state, target_state)
                now = await self._database_now(connection)
                updated_generation = replace(
                    generation,
                    state=GenerationState.ABANDONED,
                    revision=generation.revision + 1,
                    pending_operation_id=None,
                    failure_reason_code=command.reason_code,
                    updated_at=now,
                )
                updated_job = replace(
                    job,
                    state=target_state,
                    revision=job.revision + 1,
                    current_generation=None,
                    pending_operation_id=None,
                    terminal_outcome=command.terminal_outcome,
                    updated_at=now,
                )
                ticket = replace(
                    provisional,
                    conversation_id=job.conversation_id,
                    job_state=job.state,
                )
                result = OperationResult(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=job.job_id,
                    generation=generation.generation,
                    job_state=target_state,
                    generation_state=GenerationState.ABANDONED,
                )
                await self._update_operation_ticket(
                    connection, command.operation_id, ticket, now
                )
                await self._update_generation(
                    connection, generation, updated_generation
                )
                await self._update_job(connection, job, updated_job)
                if generation.pending_operation_id is not None:
                    await self._fail_operation(
                        connection, generation.pending_operation_id, now
                    )
                await self._complete_operation(connection, result, now)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=generation.generation,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=target_state,
                    before_generation_state=generation.state,
                    after_generation_state=GenerationState.ABANDONED,
                    reason_code=command.reason_code,
                    now=now,
                )
                return result

    async def inspect_job(self, owner: BrokerOwner, job_id: UUID):
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await self._job(connection, owner, job_id)
                generation = (
                    await self._generation(
                        connection, job_id, job.generation, lock=False
                    )
                    if job.generation > 0
                    else None
                )
                return project_job_snapshot(job, generation)

    async def inspect_job_for_recovery(self, owner: BrokerOwner, job_id: UUID):
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await self._job(connection, owner, job_id)
                generation = (
                    await self._generation(
                        connection, job_id, job.generation, lock=False
                    )
                    if job.generation > 0
                    else None
                )
                return project_recovery_snapshot(job, generation)

    async def begin_quiesce(self, command: BeginQuiesceCommand) -> OperationTicket:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                operation_time = await self._database_now(connection)
                provisional = OperationTicket(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=command.job_id,
                    generation=command.expected_generation,
                    job_state=JobState.ACTIVE,
                    generation_state=GenerationState.QUIESCING,
                )
                _, replay = await self._acquire_caller_operation(
                    connection, command, provisional, operation_time
                )
                if replay is not None:
                    return replay
                job = await self._job(connection, command.owner, command.job_id)
                self._expected_generation(job, command.expected_generation)
                if (
                    job.pending_operation_id is not None
                    or job.current_generation != command.expected_generation
                ):
                    raise self._invalid_job(job, command.kind)
                generation = await self._generation(
                    connection, job.job_id, command.expected_generation
                )
                require_job_transition(command.kind, job.state, JobState.ACTIVE)
                require_generation_transition(
                    command.kind, generation.state, GenerationState.QUIESCING
                )
                now = await self._database_now(connection)
                updated_job = replace(
                    job,
                    revision=job.revision + 1,
                    pending_operation_id=command.operation_id,
                    updated_at=now,
                )
                updated_generation = replace(
                    generation,
                    state=GenerationState.QUIESCING,
                    revision=generation.revision + 1,
                    pending_operation_id=command.operation_id,
                    barrier_id=command.barrier_id,
                    updated_at=now,
                )
                ticket = replace(provisional, conversation_id=job.conversation_id)
                await self._update_operation_ticket(
                    connection, command.operation_id, ticket, now
                )
                await self._update_generation(
                    connection, generation, updated_generation
                )
                await self._update_job(connection, job, updated_job)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=generation.generation,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=job.state,
                    before_generation_state=generation.state,
                    after_generation_state=GenerationState.QUIESCING,
                    reason_code=None,
                    now=now,
                )
                return ticket

    async def mark_quiesced(self, command: MarkQuiescedCommand) -> OperationResult:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                _, replay = await self._pending_completion(
                    connection,
                    owner=command.owner,
                    operation_id=command.operation_id,
                    job_id=command.job_id,
                    generation=command.generation,
                    expected_command=CommandKind.BEGIN_QUIESCE,
                )
                if replay is not None:
                    return replay
                job = await self._job(connection, command.owner, command.job_id)
                generation = await self._generation(
                    connection, command.job_id, command.generation
                )
                if (
                    job.current_generation != command.generation
                    or job.pending_operation_id != command.operation_id
                    or generation.pending_operation_id != command.operation_id
                ):
                    raise OperationMismatch(command.operation_id)
                require_job_transition(command.kind, job.state, JobState.ACTIVE)
                require_generation_transition(
                    command.kind, generation.state, GenerationState.QUIESCED
                )
                now = await self._database_now(connection)
                updated_job = replace(
                    job,
                    revision=job.revision + 1,
                    pending_operation_id=None,
                    updated_at=now,
                )
                updated_generation = replace(
                    generation,
                    state=GenerationState.QUIESCED,
                    revision=generation.revision + 1,
                    pending_operation_id=None,
                    updated_at=now,
                )
                result = OperationResult(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=job.job_id,
                    generation=command.generation,
                    job_state=JobState.ACTIVE,
                    generation_state=GenerationState.QUIESCED,
                )
                await self._update_generation(
                    connection, generation, updated_generation
                )
                await self._update_job(connection, job, updated_job)
                await self._complete_operation(connection, result, now)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=command.generation,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=job.state,
                    before_generation_state=generation.state,
                    after_generation_state=GenerationState.QUIESCED,
                    reason_code=None,
                    now=now,
                )
                return result

    @staticmethod
    def _verify_receipt(
        command: BeginReleaseCommand,
        job: JobRecord,
        generation: GenerationRecord,
    ) -> None:
        receipt = command.receipt
        checks = (
            (ReceiptField.TENANT_ID, receipt.tenant_id, job.owner.tenant_id),
            (ReceiptField.JOB_ID, receipt.job_id, job.job_id),
            (
                ReceiptField.CONVERSATION_ID,
                receipt.conversation_id,
                job.conversation_id,
            ),
            (ReceiptField.GENERATION, receipt.generation, generation.generation),
            (ReceiptField.BARRIER_ID, receipt.barrier_id, generation.barrier_id),
        )
        for field_name, actual, expected in checks:
            if actual != expected:
                raise ReceiptBindingMismatch(field_name)

    async def begin_release(self, command: BeginReleaseCommand) -> OperationTicket:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                operation_time = await self._database_now(connection)
                provisional = OperationTicket(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=command.job_id,
                    generation=command.expected_generation,
                    job_state=JobState.ACTIVE,
                    generation_state=GenerationState.RELEASING,
                )
                _, replay = await self._acquire_caller_operation(
                    connection, command, provisional, operation_time
                )
                if replay is not None:
                    return replay
                job = await self._job(connection, command.owner, command.job_id)
                self._expected_generation(job, command.expected_generation)
                if (
                    job.pending_operation_id is not None
                    or job.current_generation != command.expected_generation
                ):
                    raise self._invalid_job(job, command.kind)
                generation = await self._generation(
                    connection, job.job_id, command.expected_generation
                )
                require_job_transition(command.kind, job.state, JobState.ACTIVE)
                require_generation_transition(
                    command.kind, generation.state, GenerationState.RELEASING
                )
                self._verify_receipt(command, job, generation)
                now = await self._database_now(connection)
                updated_job = replace(
                    job,
                    revision=job.revision + 1,
                    pending_operation_id=command.operation_id,
                    updated_at=now,
                )
                updated_generation = replace(
                    generation,
                    state=GenerationState.RELEASING,
                    revision=generation.revision + 1,
                    pending_operation_id=command.operation_id,
                    receipt=command.receipt,
                    release_target=command.target,
                    release_terminal_outcome=command.terminal_outcome,
                    updated_at=now,
                )
                ticket = replace(provisional, conversation_id=job.conversation_id)
                await self._update_operation_ticket(
                    connection, command.operation_id, ticket, now
                )
                await self._update_generation(
                    connection, generation, updated_generation
                )
                await self._update_job(connection, job, updated_job)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=generation.generation,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=job.state,
                    before_generation_state=generation.state,
                    after_generation_state=GenerationState.RELEASING,
                    reason_code=None,
                    now=now,
                )
                return ticket

    async def mark_released(self, command: MarkReleasedCommand) -> OperationResult:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                _, replay = await self._pending_completion(
                    connection,
                    owner=command.owner,
                    operation_id=command.operation_id,
                    job_id=command.job_id,
                    generation=command.generation,
                    expected_command=CommandKind.BEGIN_RELEASE,
                )
                if replay is not None:
                    return replay
                job = await self._job(connection, command.owner, command.job_id)
                generation = await self._generation(
                    connection, command.job_id, command.generation
                )
                if (
                    job.current_generation != command.generation
                    or job.pending_operation_id != command.operation_id
                    or generation.pending_operation_id != command.operation_id
                    or generation.release_target is None
                ):
                    raise OperationMismatch(command.operation_id)
                target_state = JobState(generation.release_target.value)
                require_job_transition(command.kind, job.state, target_state)
                require_generation_transition(
                    command.kind, generation.state, GenerationState.RELEASED
                )
                now = await self._database_now(connection)
                updated_job = replace(
                    job,
                    state=target_state,
                    revision=job.revision + 1,
                    current_generation=None,
                    pending_operation_id=None,
                    terminal_outcome=generation.release_terminal_outcome,
                    updated_at=now,
                )
                updated_generation = replace(
                    generation,
                    state=GenerationState.RELEASED,
                    revision=generation.revision + 1,
                    pending_operation_id=None,
                    updated_at=now,
                )
                result = OperationResult(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=job.job_id,
                    generation=command.generation,
                    job_state=target_state,
                    generation_state=GenerationState.RELEASED,
                )
                await self._update_generation(
                    connection, generation, updated_generation
                )
                await self._update_job(connection, job, updated_job)
                await self._complete_operation(connection, result, now)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=command.generation,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=target_state,
                    before_generation_state=generation.state,
                    after_generation_state=GenerationState.RELEASED,
                    reason_code=None,
                    now=now,
                )
                return result

    async def begin_finalize(self, command: BeginFinalizeCommand) -> OperationTicket:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                operation_time = await self._database_now(connection)
                provisional = OperationTicket(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=command.job_id,
                    generation=command.expected_generation or None,
                    job_state=JobState.FINALIZING,
                )
                _, replay = await self._acquire_caller_operation(
                    connection, command, provisional, operation_time
                )
                if replay is not None:
                    return replay
                job = await self._job(connection, command.owner, command.job_id)
                self._expected_generation(job, command.expected_generation)
                if (
                    job.pending_operation_id is not None
                    or job.current_generation is not None
                ):
                    raise self._invalid_job(job, command.kind)
                require_job_transition(command.kind, job.state, JobState.FINALIZING)
                if (
                    job.state is JobState.FINALIZING
                    and job.terminal_outcome is not command.terminal_outcome
                ):
                    raise self._invalid_job(job, command.kind)
                latest = (
                    await self._generation(
                        connection, job.job_id, job.generation, lock=False
                    )
                    if job.generation > 0
                    else None
                )
                now = await self._database_now(connection)
                updated_job = replace(
                    job,
                    state=JobState.FINALIZING,
                    revision=job.revision + 1,
                    pending_operation_id=command.operation_id,
                    terminal_outcome=command.terminal_outcome,
                    updated_at=now,
                )
                ticket = replace(
                    provisional,
                    conversation_id=job.conversation_id,
                    generation_state=latest.state if latest else None,
                )
                await self._update_operation_ticket(
                    connection, command.operation_id, ticket, now
                )
                await self._update_job(connection, job, updated_job)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=job.generation or None,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=JobState.FINALIZING,
                    before_generation_state=latest.state if latest else None,
                    after_generation_state=latest.state if latest else None,
                    reason_code=None,
                    now=now,
                )
                return ticket

    async def mark_terminal(self, command: MarkTerminalCommand) -> OperationResult:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(_SELECT_OPERATION, command.operation_id)
                if row is None:
                    raise OperationMismatch(command.operation_id)
                operation = _operation_from_row(row)
                if (
                    operation.owner != command.owner
                    or operation.job_id != command.job_id
                    or operation.command is not CommandKind.BEGIN_FINALIZE
                ):
                    raise OperationMismatch(command.operation_id)
                if operation.status is OperationStatus.COMPLETED:
                    if operation.completion_result is None:
                        raise OperationMismatch(command.operation_id)
                    return replace(operation.completion_result, replayed=True)
                if operation.status is not OperationStatus.PENDING:
                    raise OperationMismatch(command.operation_id)
                job = await self._job(connection, command.owner, command.job_id)
                if (
                    job.pending_operation_id != command.operation_id
                    or job.current_generation is not None
                    or operation.generation != (job.generation or None)
                ):
                    raise OperationMismatch(command.operation_id)
                require_job_transition(command.kind, job.state, JobState.TERMINAL)
                latest = (
                    await self._generation(
                        connection, job.job_id, job.generation, lock=False
                    )
                    if job.generation > 0
                    else None
                )
                now = await self._database_now(connection)
                updated_job = replace(
                    job,
                    state=JobState.TERMINAL,
                    revision=job.revision + 1,
                    pending_operation_id=None,
                    updated_at=now,
                )
                result = OperationResult(
                    operation_id=command.operation_id,
                    command=command.kind,
                    job_id=job.job_id,
                    generation=job.generation or None,
                    job_state=JobState.TERMINAL,
                    generation_state=latest.state if latest else None,
                )
                await self._update_job(connection, job, updated_job)
                await self._complete_operation(connection, result, now)
                await self._audit(
                    connection,
                    command=command.kind,
                    owner=command.owner,
                    job_id=job.job_id,
                    generation=job.generation or None,
                    operation_id=command.operation_id,
                    before_job_state=job.state,
                    after_job_state=JobState.TERMINAL,
                    before_generation_state=latest.state if latest else None,
                    after_generation_state=latest.state if latest else None,
                    reason_code=None,
                    now=now,
                )
                return result

    @staticmethod
    def _validate_applied(
        migrations: tuple[Migration, ...], applied_rows: list[Any]
    ) -> None:
        by_version = {migration.version: migration for migration in migrations}
        latest_packaged = migrations[-1].version if migrations else 0
        for row in applied_rows:
            version = int(row["version"])
            if version > latest_packaged:
                raise MigrationVersionError(version, MigrationProblem.FUTURE_VERSION)
        for expected, row in enumerate(applied_rows, start=1):
            version = int(row["version"])
            if version != expected:
                raise MigrationVersionError(expected, MigrationProblem.NUMBERING_GAP)
            migration = by_version.get(version)
            if migration is None:
                raise MigrationVersionError(version, MigrationProblem.FUTURE_VERSION)
            if (
                str(row["name"]) != migration.name
                or str(row["checksum"]) != migration.checksum
            ):
                raise MigrationVersionError(version, MigrationProblem.CHECKSUM_DRIFT)
