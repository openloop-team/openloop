"""Deterministic in-memory implementation of the broker repository contract."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

from .errors import (
    ConcurrentMutation,
    IdempotencyConflict,
    InvalidTransition,
    JobNotFound,
    OperationMismatch,
    OwnerMismatch,
    ReceiptBindingMismatch,
    ReceiptField,
    StaleGeneration,
    TransitionEntity,
)
from .models import (
    AuditRecord,
    BrokerOwner,
    CommandKind,
    GenerationRecord,
    GenerationState,
    JobAuthorizationRecord,
    JobRecord,
    JobState,
    OperationRecord,
    OperationResult,
    OperationSource,
    OperationStatus,
    OperationTicket,
    ReleaseTarget,
    TerminalOutcome,
    project_job_snapshot,
    project_recovery_snapshot,
    validate_timestamp,
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


def _utc_now() -> datetime:
    return datetime.now(UTC)


class InMemoryBrokerRepository:
    """Atomic process-local repository used for contracts and deterministic tests."""

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock
        self._lock = asyncio.Lock()
        self._jobs: dict[UUID, JobRecord] = {}
        self._generations: dict[tuple[UUID, int], GenerationRecord] = {}
        self._operations: dict[UUID, OperationRecord] = {}
        self._caller_operations: dict[tuple[str, str, str], UUID] = {}
        self._audit: list[AuditRecord] = []

    def _now(self) -> datetime:
        return validate_timestamp("clock result", self._clock())

    @staticmethod
    def _caller_key(owner: BrokerOwner, idempotency_key: str) -> tuple[str, str, str]:
        return owner.tenant_id, owner.workload_subject, idempotency_key

    def _caller_replay(self, command) -> OperationTicket | None:
        operation_id = self._caller_operations.get(
            self._caller_key(command.owner, command.idempotency_key)
        )
        if operation_id is None:
            return None
        operation = self._operations[operation_id]
        if (
            operation.command is not command.kind
            or operation.request_digest != command.request_digest
        ):
            raise IdempotencyConflict()
        return replace(operation.intent_ticket, replayed=True)

    def _job(self, owner: BrokerOwner, job_id: UUID) -> JobRecord:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        if job.owner != owner:
            raise OwnerMismatch(job_id)
        return job

    @staticmethod
    def _expected_generation(job: JobRecord, expected: int) -> None:
        if job.generation != expected:
            raise StaleGeneration(
                job.job_id, expected=expected, actual=job.generation
            )

    def _generation(self, job_id: UUID, generation: int) -> GenerationRecord:
        record = self._generations.get((job_id, generation))
        if record is None:
            raise StaleGeneration(job_id, expected=generation, actual=0)
        return record

    def _latest_generation(self, job: JobRecord) -> GenerationRecord | None:
        if job.generation == 0:
            return None
        return self._generations.get((job.job_id, job.generation))

    @staticmethod
    def _invalid_job(job: JobRecord, command: CommandKind) -> InvalidTransition:
        return InvalidTransition(TransitionEntity.JOB, job.state, command)

    def _audit_record(
        self,
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
    ) -> AuditRecord:
        return AuditRecord(
            sequence=len(self._audit) + 1,
            command=command,
            owner=owner,
            job_id=job_id,
            generation=generation,
            operation_id=operation_id,
            before_job_state=before_job_state,
            after_job_state=after_job_state,
            before_generation_state=before_generation_state,
            after_generation_state=after_generation_state,
            reason_code=reason_code,
            created_at=now,
        )

    def _new_caller_operation(
        self,
        *,
        command,
        status: OperationStatus,
        ticket: OperationTicket,
        now: datetime,
    ) -> OperationRecord:
        if command.operation_id in self._operations:
            raise ConcurrentMutation(command.job_id)
        return OperationRecord(
            operation_id=command.operation_id,
            owner=command.owner,
            source=OperationSource.CALLER,
            idempotency_key=command.idempotency_key,
            command=command.kind,
            request_digest=command.request_digest,
            job_id=command.job_id,
            generation=ticket.generation,
            status=status,
            intent_ticket=ticket,
            completion_result=None,
            created_at=now,
            updated_at=now,
        )

    def _publish_caller_operation(self, operation: OperationRecord) -> None:
        assert operation.idempotency_key is not None
        self._operations[operation.operation_id] = operation
        self._caller_operations[
            self._caller_key(operation.owner, operation.idempotency_key)
        ] = operation.operation_id

    def _pending_completion(
        self,
        *,
        owner: BrokerOwner,
        operation_id: UUID,
        job_id: UUID,
        generation: int | None,
        expected_command: CommandKind,
    ) -> tuple[OperationRecord, OperationResult | None]:
        operation = self._operations.get(operation_id)
        if (
            operation is None
            or operation.owner != owner
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
        async with self._lock:
            replay = self._caller_replay(command)
            if replay is not None:
                return replay
            if command.job_id in self._jobs or any(
                job.conversation_id == command.conversation_id
                for job in self._jobs.values()
            ):
                raise ConcurrentMutation(command.job_id)
            now = self._now()
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
                minimum_isolation=command.minimum_isolation,
                authorization=command.authorization,
            )
            ticket = OperationTicket(
                operation_id=command.operation_id,
                command=command.kind,
                job_id=command.job_id,
                conversation_id=command.conversation_id,
                job_state=JobState.CREATED,
            )
            operation = self._new_caller_operation(
                command=command,
                status=OperationStatus.COMPLETED,
                ticket=ticket,
                now=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = job
            self._publish_caller_operation(operation)
            self._audit.append(audit)
            return ticket

    async def begin_start(self, command: BeginStartCommand) -> OperationTicket:
        async with self._lock:
            replay = self._caller_replay(command)
            if replay is not None:
                return replay
            job = self._job(command.owner, command.job_id)
            self._expected_generation(job, command.expected_generation)
            require_job_transition(command.kind, job.state, job.state)
            if job.pending_operation_id is not None or job.current_generation is not None:
                raise self._invalid_job(job, command.kind)
            if command.operation_id in self._operations:
                raise ConcurrentMutation(job.job_id)
            generation_number = job.generation + 1
            if (job.job_id, generation_number) in self._generations:
                raise ConcurrentMutation(job.job_id)
            now = self._now()
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
            ticket = OperationTicket(
                operation_id=command.operation_id,
                command=command.kind,
                job_id=job.job_id,
                conversation_id=job.conversation_id,
                generation=generation_number,
                job_state=job.state,
                generation_state=GenerationState.STARTING,
            )
            operation = self._new_caller_operation(
                command=command,
                status=OperationStatus.PENDING,
                ticket=ticket,
                now=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._generations[(job.job_id, generation_number)] = generation
            self._publish_caller_operation(operation)
            self._audit.append(audit)
            return ticket

    async def mark_running(self, command: MarkRunningCommand) -> OperationResult:
        async with self._lock:
            operation, replay = self._pending_completion(
                owner=command.owner,
                operation_id=command.operation_id,
                job_id=command.job_id,
                generation=command.generation,
                expected_command=CommandKind.BEGIN_START,
            )
            if replay is not None:
                generation = self._generation(command.job_id, command.generation)
                if (
                    generation.runtime_ref != command.runtime_ref
                    or generation.durable_state_ref != command.durable_state_ref
                    or generation.runtime_key_version != command.runtime_key_version
                    or generation.durable_key_version != command.durable_key_version
                    or generation.capability_digest != command.capability_digest
                ):
                    raise OperationMismatch(command.operation_id)
                job = self._job(command.owner, command.job_id)
                if generation.durable_digest != command.durable_digest:
                    raise OperationMismatch(command.operation_id)
                return replay
            job = self._job(command.owner, command.job_id)
            generation = self._generation(command.job_id, command.generation)
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
            now = self._now()
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
            updated_operation = replace(
                operation,
                status=OperationStatus.COMPLETED,
                completion_result=result,
                updated_at=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._generations[(job.job_id, command.generation)] = updated_generation
            self._operations[operation.operation_id] = updated_operation
            self._audit.append(audit)
            return result

    async def begin_quiesce(
        self, command: BeginQuiesceCommand
    ) -> OperationTicket:
        async with self._lock:
            replay = self._caller_replay(command)
            if replay is not None:
                return replay
            job = self._job(command.owner, command.job_id)
            self._expected_generation(job, command.expected_generation)
            if (
                job.pending_operation_id is not None
                or job.current_generation != command.expected_generation
            ):
                raise self._invalid_job(job, command.kind)
            generation = self._generation(job.job_id, command.expected_generation)
            require_job_transition(command.kind, job.state, JobState.ACTIVE)
            require_generation_transition(
                command.kind, generation.state, GenerationState.QUIESCING
            )
            now = self._now()
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
            ticket = OperationTicket(
                operation_id=command.operation_id,
                command=command.kind,
                job_id=job.job_id,
                conversation_id=job.conversation_id,
                generation=generation.generation,
                job_state=JobState.ACTIVE,
                generation_state=GenerationState.QUIESCING,
            )
            operation = self._new_caller_operation(
                command=command,
                status=OperationStatus.PENDING,
                ticket=ticket,
                now=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._generations[(job.job_id, generation.generation)] = updated_generation
            self._publish_caller_operation(operation)
            self._audit.append(audit)
            return ticket

    async def mark_quiesced(self, command: MarkQuiescedCommand) -> OperationResult:
        async with self._lock:
            operation, replay = self._pending_completion(
                owner=command.owner,
                operation_id=command.operation_id,
                job_id=command.job_id,
                generation=command.generation,
                expected_command=CommandKind.BEGIN_QUIESCE,
            )
            if replay is not None:
                return replay
            job = self._job(command.owner, command.job_id)
            generation = self._generation(command.job_id, command.generation)
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
            now = self._now()
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
            updated_operation = replace(
                operation,
                status=OperationStatus.COMPLETED,
                completion_result=result,
                updated_at=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._generations[(job.job_id, command.generation)] = updated_generation
            self._operations[operation.operation_id] = updated_operation
            self._audit.append(audit)
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

    async def begin_release(
        self, command: BeginReleaseCommand
    ) -> OperationTicket:
        async with self._lock:
            replay = self._caller_replay(command)
            if replay is not None:
                return replay
            job = self._job(command.owner, command.job_id)
            self._expected_generation(job, command.expected_generation)
            if (
                job.pending_operation_id is not None
                or job.current_generation != command.expected_generation
            ):
                raise self._invalid_job(job, command.kind)
            generation = self._generation(job.job_id, command.expected_generation)
            require_job_transition(command.kind, job.state, JobState.ACTIVE)
            require_generation_transition(
                command.kind, generation.state, GenerationState.RELEASING
            )
            self._verify_receipt(command, job, generation)
            now = self._now()
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
            ticket = OperationTicket(
                operation_id=command.operation_id,
                command=command.kind,
                job_id=job.job_id,
                conversation_id=job.conversation_id,
                generation=generation.generation,
                job_state=JobState.ACTIVE,
                generation_state=GenerationState.RELEASING,
            )
            operation = self._new_caller_operation(
                command=command,
                status=OperationStatus.PENDING,
                ticket=ticket,
                now=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._generations[(job.job_id, generation.generation)] = updated_generation
            self._publish_caller_operation(operation)
            self._audit.append(audit)
            return ticket

    async def mark_released(self, command: MarkReleasedCommand) -> OperationResult:
        async with self._lock:
            operation, replay = self._pending_completion(
                owner=command.owner,
                operation_id=command.operation_id,
                job_id=command.job_id,
                generation=command.generation,
                expected_command=CommandKind.BEGIN_RELEASE,
            )
            if replay is not None:
                return replay
            job = self._job(command.owner, command.job_id)
            generation = self._generation(command.job_id, command.generation)
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
            now = self._now()
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
            updated_operation = replace(
                operation,
                status=OperationStatus.COMPLETED,
                completion_result=result,
                updated_at=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._generations[(job.job_id, command.generation)] = updated_generation
            self._operations[operation.operation_id] = updated_operation
            self._audit.append(audit)
            return result

    async def abandon_generation(
        self, command: AbandonGenerationCommand
    ) -> OperationResult:
        async with self._lock:
            existing = self._operations.get(command.operation_id)
            if existing is not None:
                if (
                    existing.source is not OperationSource.INTERNAL
                    or existing.command is not command.kind
                    or existing.owner != command.owner
                    or existing.job_id != command.job_id
                    or existing.generation != command.generation
                    or existing.request_digest != command.request_digest
                    or existing.status is not OperationStatus.COMPLETED
                    or existing.completion_result is None
                ):
                    raise OperationMismatch(command.operation_id)
                return replace(existing.completion_result, replayed=True)
            if command.replay_operation:
                raise OperationMismatch(command.operation_id)
            job = self._job(command.owner, command.job_id)
            if job.generation != command.generation:
                raise StaleGeneration(
                    job.job_id,
                    expected=command.generation,
                    actual=job.generation,
                )
            generation = self._generation(job.job_id, command.generation)
            if generation.state is not command.expected_state:
                raise InvalidTransition(
                    TransitionEntity.GENERATION, generation.state, command.kind
                )
            require_generation_transition(
                command.kind, generation.state, GenerationState.ABANDONED
            )
            if generation.state is GenerationState.STARTING:
                target_state = generation.previous_job_state
            else:
                target_state = JobState.FINALIZING
            require_job_transition(command.kind, job.state, target_state)
            now = self._now()
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
            ticket = OperationTicket(
                operation_id=command.operation_id,
                command=command.kind,
                job_id=job.job_id,
                conversation_id=job.conversation_id,
                generation=generation.generation,
                job_state=job.state,
                generation_state=generation.state,
            )
            result = OperationResult(
                operation_id=command.operation_id,
                command=command.kind,
                job_id=job.job_id,
                generation=generation.generation,
                job_state=target_state,
                generation_state=GenerationState.ABANDONED,
            )
            operation = OperationRecord(
                operation_id=command.operation_id,
                owner=command.owner,
                source=OperationSource.INTERNAL,
                idempotency_key=None,
                command=command.kind,
                request_digest=command.request_digest,
                job_id=job.job_id,
                generation=generation.generation,
                status=OperationStatus.COMPLETED,
                intent_ticket=ticket,
                completion_result=result,
                created_at=now,
                updated_at=now,
            )
            superseded_operation = None
            if generation.pending_operation_id is not None:
                pending = self._operations.get(generation.pending_operation_id)
                if pending is not None and pending.status is OperationStatus.PENDING:
                    superseded_operation = replace(
                        pending, status=OperationStatus.FAILED, updated_at=now
                    )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._generations[(job.job_id, generation.generation)] = updated_generation
            self._operations[operation.operation_id] = operation
            if superseded_operation is not None:
                self._operations[
                    superseded_operation.operation_id
                ] = superseded_operation
            self._audit.append(audit)
            return result

    async def begin_finalize(
        self, command: BeginFinalizeCommand
    ) -> OperationTicket:
        async with self._lock:
            replay = self._caller_replay(command)
            if replay is not None:
                return replay
            job = self._job(command.owner, command.job_id)
            self._expected_generation(job, command.expected_generation)
            if job.pending_operation_id is not None or job.current_generation is not None:
                raise self._invalid_job(job, command.kind)
            require_job_transition(command.kind, job.state, JobState.FINALIZING)
            if (
                job.state is JobState.FINALIZING
                and job.terminal_outcome is not command.terminal_outcome
            ):
                raise self._invalid_job(job, command.kind)
            now = self._now()
            updated_job = replace(
                job,
                state=JobState.FINALIZING,
                revision=job.revision + 1,
                pending_operation_id=command.operation_id,
                terminal_outcome=command.terminal_outcome,
                updated_at=now,
            )
            latest = self._latest_generation(job)
            ticket = OperationTicket(
                operation_id=command.operation_id,
                command=command.kind,
                job_id=job.job_id,
                conversation_id=job.conversation_id,
                generation=job.generation or None,
                job_state=JobState.FINALIZING,
                generation_state=latest.state if latest else None,
            )
            operation = self._new_caller_operation(
                command=command,
                status=OperationStatus.PENDING,
                ticket=ticket,
                now=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._publish_caller_operation(operation)
            self._audit.append(audit)
            return ticket

    async def mark_terminal(self, command: MarkTerminalCommand) -> OperationResult:
        async with self._lock:
            job_for_generation = self._jobs.get(command.job_id)
            expected_generation = (
                job_for_generation.generation
                if job_for_generation is not None and job_for_generation.generation > 0
                else None
            )
            operation, replay = self._pending_completion(
                owner=command.owner,
                operation_id=command.operation_id,
                job_id=command.job_id,
                generation=expected_generation,
                expected_command=CommandKind.BEGIN_FINALIZE,
            )
            if replay is not None:
                return replay
            job = self._job(command.owner, command.job_id)
            if (
                job.pending_operation_id != command.operation_id
                or job.current_generation is not None
            ):
                raise OperationMismatch(command.operation_id)
            require_job_transition(command.kind, job.state, JobState.TERMINAL)
            now = self._now()
            updated_job = replace(
                job,
                state=JobState.TERMINAL,
                revision=job.revision + 1,
                pending_operation_id=None,
                updated_at=now,
            )
            latest = self._latest_generation(job)
            result = OperationResult(
                operation_id=command.operation_id,
                command=command.kind,
                job_id=job.job_id,
                generation=job.generation or None,
                job_state=JobState.TERMINAL,
                generation_state=latest.state if latest else None,
            )
            updated_operation = replace(
                operation,
                status=OperationStatus.COMPLETED,
                completion_result=result,
                updated_at=now,
            )
            audit = self._audit_record(
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
            self._jobs[job.job_id] = updated_job
            self._operations[operation.operation_id] = updated_operation
            self._audit.append(audit)
            return result

    async def inspect_job(self, owner: BrokerOwner, job_id: UUID):
        async with self._lock:
            job = self._job(owner, job_id)
            return project_job_snapshot(job, self._latest_generation(job))

    async def inspect_job_authorization(
        self, owner: BrokerOwner, job_id: UUID
    ) -> JobAuthorizationRecord:
        async with self._lock:
            job = self._job(owner, job_id)
            if job.minimum_isolation is None or job.authorization is None:
                raise JobNotFound(job_id)
            return JobAuthorizationRecord(
                job_id=job.job_id,
                owner=job.owner,
                minimum_isolation=job.minimum_isolation,
                authorization=job.authorization,
            )

    async def inspect_job_for_recovery(self, owner: BrokerOwner, job_id: UUID):
        async with self._lock:
            job = self._job(owner, job_id)
            return project_recovery_snapshot(job, self._latest_generation(job))

    async def audit_records_for_test(self) -> tuple[AuditRecord, ...]:
        """Return immutable audit state for contract tests; not repository API."""
        async with self._lock:
            return tuple(self._audit)

    async def operations_for_test(self) -> tuple[OperationRecord, ...]:
        """Return immutable operations for contract tests; not repository API."""
        async with self._lock:
            return tuple(self._operations.values())

    async def operation_for_test(self, operation_id: UUID) -> OperationRecord:
        """Return one immutable operation for contract tests; not repository API."""
        async with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                raise OperationMismatch(operation_id)
            return operation
