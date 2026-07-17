"""Validated application boundary for broker lifecycle repositories."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

from .models import (
    BrokerOwner,
    GenerationState,
    JobSnapshot,
    OperationResult,
    OperationTicket,
    RecoverySnapshot,
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
from .repository import (
    AbandonGenerationCommand,
    BeginFinalizeCommand,
    BeginQuiesceCommand,
    BeginReleaseCommand,
    BeginStartCommand,
    BrokerRepository,
    CreateJobCommand,
    MarkQuiescedCommand,
    MarkReleasedCommand,
    MarkRunningCommand,
    MarkTerminalCommand,
)


class BrokerLedger:
    def __init__(
        self,
        repository: BrokerRepository,
        *,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._repository = repository
        self._id_factory = id_factory

    @staticmethod
    def _owner(owner: object) -> BrokerOwner:
        if not isinstance(owner, BrokerOwner):
            raise TypeError("owner must be a BrokerOwner")
        return owner

    @staticmethod
    def _enum(name: str, value: object, enum_type):
        if not isinstance(value, enum_type):
            raise TypeError(f"{name} must be a {enum_type.__name__}")
        return value

    def _mint_id(self) -> UUID:
        return validate_uuid("broker-minted ID", self._id_factory())

    @staticmethod
    def _prepare(command):
        if hasattr(command, "request_digest"):
            command.request_digest
        return command

    async def create_job(
        self,
        owner: BrokerOwner,
        idempotency_key: str,
        profile: str,
        runtime_driver: str,
        durable_state_driver: str,
    ) -> OperationTicket:
        self._owner(owner)
        validate_idempotency_key(idempotency_key)
        validate_token("profile", profile)
        validate_token("runtime_driver", runtime_driver)
        validate_token("durable_state_driver", durable_state_driver)
        job_id = self._mint_id()
        conversation_id = self._mint_id()
        operation_id = self._mint_id()
        command = CreateJobCommand(
            owner=owner,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
            job_id=job_id,
            conversation_id=conversation_id,
            profile=profile,
            runtime_driver=runtime_driver,
            durable_state_driver=durable_state_driver,
        )
        return await self._repository.create_job(self._prepare(command))

    async def begin_start(
        self,
        owner: BrokerOwner,
        idempotency_key: str,
        job_id: UUID,
        expected_generation: int,
        execution_lease_seconds: int,
    ) -> OperationTicket:
        self._owner(owner)
        validate_idempotency_key(idempotency_key)
        validate_uuid("job_id", job_id)
        validate_bigint("expected_generation", expected_generation)
        validate_lease_seconds(execution_lease_seconds)
        command = BeginStartCommand(
            owner=owner,
            idempotency_key=idempotency_key,
            operation_id=self._mint_id(),
            job_id=job_id,
            expected_generation=expected_generation,
            execution_lease_seconds=execution_lease_seconds,
        )
        return await self._repository.begin_start(self._prepare(command))

    async def mark_running(
        self,
        owner: BrokerOwner,
        operation_id: UUID,
        job_id: UUID,
        generation: int,
        runtime_ref: str,
        durable_state_ref: str,
        runtime_key_version: str,
        durable_key_version: str,
        capability_digest: str,
        durable_digest: str,
    ) -> OperationResult:
        self._owner(owner)
        validate_uuid("operation_id", operation_id)
        validate_uuid("job_id", job_id)
        validate_positive_bigint("generation", generation)
        validate_opaque_ref("runtime_ref", runtime_ref)
        validate_opaque_ref("durable_state_ref", durable_state_ref)
        validate_identifier("runtime_key_version", runtime_key_version)
        validate_identifier("durable_key_version", durable_key_version)
        validate_sha256("capability_digest", capability_digest)
        validate_sha256("durable_digest", durable_digest)
        return await self._repository.mark_running(
            MarkRunningCommand(
                owner=owner,
                operation_id=operation_id,
                job_id=job_id,
                generation=generation,
                runtime_ref=runtime_ref,
                durable_state_ref=durable_state_ref,
                runtime_key_version=runtime_key_version,
                durable_key_version=durable_key_version,
                capability_digest=capability_digest,
                durable_digest=durable_digest,
            )
        )

    async def begin_quiesce(
        self,
        owner: BrokerOwner,
        idempotency_key: str,
        job_id: UUID,
        expected_generation: int,
        barrier_id: str,
    ) -> OperationTicket:
        self._owner(owner)
        validate_idempotency_key(idempotency_key)
        validate_uuid("job_id", job_id)
        validate_bigint("expected_generation", expected_generation)
        validate_identifier("barrier_id", barrier_id)
        command = BeginQuiesceCommand(
            owner=owner,
            idempotency_key=idempotency_key,
            operation_id=self._mint_id(),
            job_id=job_id,
            expected_generation=expected_generation,
            barrier_id=barrier_id,
        )
        return await self._repository.begin_quiesce(self._prepare(command))

    async def mark_quiesced(
        self,
        owner: BrokerOwner,
        operation_id: UUID,
        job_id: UUID,
        generation: int,
    ) -> OperationResult:
        self._owner(owner)
        validate_uuid("operation_id", operation_id)
        validate_uuid("job_id", job_id)
        validate_positive_bigint("generation", generation)
        return await self._repository.mark_quiesced(
            MarkQuiescedCommand(owner, operation_id, job_id, generation)
        )

    async def begin_release(
        self,
        owner: BrokerOwner,
        idempotency_key: str,
        job_id: UUID,
        expected_generation: int,
        receipt: VerifiedCheckpointReceipt,
        target: ReleaseTarget,
        terminal_outcome: TerminalOutcome | None = None,
    ) -> OperationTicket:
        self._owner(owner)
        validate_idempotency_key(idempotency_key)
        validate_uuid("job_id", job_id)
        validate_bigint("expected_generation", expected_generation)
        if not isinstance(receipt, VerifiedCheckpointReceipt):
            raise TypeError("receipt must be a VerifiedCheckpointReceipt")
        self._enum("target", target, ReleaseTarget)
        if target is ReleaseTarget.FINALIZING and terminal_outcome is None:
            raise ValueError("a finalizing release requires terminal_outcome")
        if target is ReleaseTarget.PARKED and terminal_outcome is not None:
            raise ValueError("a parked release cannot set terminal_outcome")
        if terminal_outcome is not None:
            self._enum("terminal_outcome", terminal_outcome, TerminalOutcome)
        command = BeginReleaseCommand(
            owner=owner,
            idempotency_key=idempotency_key,
            operation_id=self._mint_id(),
            job_id=job_id,
            expected_generation=expected_generation,
            receipt=receipt,
            target=target,
            terminal_outcome=terminal_outcome,
        )
        return await self._repository.begin_release(self._prepare(command))

    async def mark_released(
        self,
        owner: BrokerOwner,
        operation_id: UUID,
        job_id: UUID,
        generation: int,
    ) -> OperationResult:
        self._owner(owner)
        validate_uuid("operation_id", operation_id)
        validate_uuid("job_id", job_id)
        validate_positive_bigint("generation", generation)
        return await self._repository.mark_released(
            MarkReleasedCommand(owner, operation_id, job_id, generation)
        )

    async def abandon_generation(
        self,
        owner: BrokerOwner,
        job_id: UUID,
        generation: int,
        expected_state: GenerationState,
        reason_code: str,
        terminal_outcome: TerminalOutcome | None = None,
        *,
        replay_operation_id: UUID | None = None,
    ) -> OperationResult:
        self._owner(owner)
        validate_uuid("job_id", job_id)
        validate_positive_bigint("generation", generation)
        self._enum("expected_state", expected_state, GenerationState)
        validate_token("reason_code", reason_code)
        if expected_state is GenerationState.STARTING:
            if terminal_outcome is not None:
                raise ValueError("starting abandonment cannot set terminal_outcome")
        elif terminal_outcome not in {
            TerminalOutcome.CANCELLED,
            TerminalOutcome.FAILED,
        }:
            raise ValueError(
                "active generation abandonment requires failed or cancelled outcome"
            )
        if replay_operation_id is not None:
            validate_uuid("replay_operation_id", replay_operation_id)
        operation_id = replay_operation_id or self._mint_id()
        command = AbandonGenerationCommand(
            owner=owner,
            operation_id=operation_id,
            job_id=job_id,
            generation=generation,
            expected_state=expected_state,
            reason_code=reason_code,
            terminal_outcome=terminal_outcome,
            replay_operation=replay_operation_id is not None,
        )
        return await self._repository.abandon_generation(self._prepare(command))

    async def begin_finalize(
        self,
        owner: BrokerOwner,
        idempotency_key: str,
        job_id: UUID,
        expected_generation: int,
        terminal_outcome: TerminalOutcome,
    ) -> OperationTicket:
        self._owner(owner)
        validate_idempotency_key(idempotency_key)
        validate_uuid("job_id", job_id)
        validate_bigint("expected_generation", expected_generation)
        self._enum("terminal_outcome", terminal_outcome, TerminalOutcome)
        command = BeginFinalizeCommand(
            owner=owner,
            idempotency_key=idempotency_key,
            operation_id=self._mint_id(),
            job_id=job_id,
            expected_generation=expected_generation,
            terminal_outcome=terminal_outcome,
        )
        return await self._repository.begin_finalize(self._prepare(command))

    async def mark_terminal(
        self,
        owner: BrokerOwner,
        operation_id: UUID,
        job_id: UUID,
    ) -> OperationResult:
        self._owner(owner)
        validate_uuid("operation_id", operation_id)
        validate_uuid("job_id", job_id)
        return await self._repository.mark_terminal(
            MarkTerminalCommand(owner, operation_id, job_id)
        )

    async def inspect_job(self, owner: BrokerOwner, job_id: UUID) -> JobSnapshot:
        self._owner(owner)
        validate_uuid("job_id", job_id)
        return await self._repository.inspect_job(owner, job_id)

    async def inspect_job_for_recovery(
        self, owner: BrokerOwner, job_id: UUID
    ) -> RecoverySnapshot:
        self._owner(owner)
        validate_uuid("job_id", job_id)
        return await self._repository.inspect_job_for_recovery(owner, job_id)
