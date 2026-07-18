"""Privileged composition of ledger, durable state, secrets, and runtime."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from openloop.broker.errors import (
    BrokerError,
    IdempotencyConflict,
)
from openloop.broker.ledger import BrokerLedger
from openloop.broker.models import (
    BrokerOwner,
    CommandKind,
    GenerationState,
    JobState,
    OperationResult,
    OperationTicket,
    RecoveryGenerationSnapshot,
    RecoverySnapshot,
    validate_timestamp,
    validate_uuid,
)
from openloop.broker_rpc.coordinator import (
    BrokerRpcPolicy,
    SegmentCoordinator,
    SegmentCoordinatorCode,
    SegmentCoordinatorProblem,
)
from openloop.broker_rpc.models import (
    RunningGenerationAccess,
    StartSegmentPayload,
    StartSegmentResult,
)
from openloop.broker_runtime.contract import (
    EnsuredGeneration,
    GenerationRuntimeIdentity,
    OpenHandsGenerationSpec,
    RuntimeDriver,
    RuntimeDriverError,
    RuntimeExpired,
)
from openloop.tools.openhands_relay import RelayClientEndpoint

from .durable import LocalDurableStateAdapter
from .secrets import (
    DerivedRuntimeSecrets,
    RuntimeSecretAuthority,
    RuntimeSecretProblem,
)


class BrokerSegmentCoordinator(SegmentCoordinator):
    """Trusted start state machine with explicit pre/post-commit boundaries."""

    def __init__(
        self,
        *,
        ledger: BrokerLedger,
        policy: BrokerRpcPolicy,
        runtime_driver: RuntimeDriver,
        secret_authority: RuntimeSecretAuthority,
        durable_state_adapter: LocalDurableStateAdapter,
        clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(ledger, BrokerLedger):
            raise TypeError("ledger must be BrokerLedger")
        if not isinstance(policy, BrokerRpcPolicy):
            raise TypeError("policy must be BrokerRpcPolicy")
        if not isinstance(runtime_driver, RuntimeDriver):
            raise TypeError("runtime_driver must implement RuntimeDriver")
        if not isinstance(secret_authority, RuntimeSecretAuthority):
            raise TypeError("secret_authority must be RuntimeSecretAuthority")
        if not isinstance(durable_state_adapter, LocalDurableStateAdapter):
            raise TypeError(
                "durable_state_adapter must be LocalDurableStateAdapter"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        maximum = runtime_driver.maximum_lifetime_seconds
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError("runtime maximum lifetime is invalid")
        if policy.execution_lease_seconds > maximum:
            raise ValueError("policy lease exceeds runtime maximum lifetime")
        self._ledger = ledger
        self._policy = policy
        self._runtime = runtime_driver
        self._secrets = secret_authority
        self._durable = durable_state_adapter
        self._clock = clock

    def _now(self) -> datetime:
        value = validate_timestamp("coordinator clock", self._clock())
        if value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("coordinator clock must return UTC")
        return value

    def _fixed_policy(self, snapshot: RecoverySnapshot) -> None:
        if (
            snapshot.profile != self._policy.profile
            or snapshot.runtime_driver != self._policy.runtime_driver
            or snapshot.durable_state_driver
            != self._policy.durable_state_driver
        ):
            raise SegmentCoordinatorProblem(
                SegmentCoordinatorCode.STATE_CONFLICT
            )

    @staticmethod
    def _problem(
        error: Exception,
        *,
        operation_id: UUID | None = None,
    ) -> SegmentCoordinatorProblem:
        if isinstance(error, SegmentCoordinatorProblem):
            return SegmentCoordinatorProblem(
                error.code,
                operation_id=error.operation_id or operation_id,
            )
        if isinstance(error, IdempotencyConflict):
            code = SegmentCoordinatorCode.IDEMPOTENCY_CONFLICT
        elif isinstance(error, RuntimeExpired):
            code = SegmentCoordinatorCode.DEADLINE_EXCEEDED
        elif isinstance(error, RuntimeDriverError):
            code = SegmentCoordinatorCode.RUNTIME_UNAVAILABLE
        elif isinstance(error, BrokerError):
            code = SegmentCoordinatorCode.STATE_CONFLICT
        else:
            code = SegmentCoordinatorCode.INTERNAL
        return SegmentCoordinatorProblem(code, operation_id=operation_id)

    def _candidate_durable_metadata(
        self,
        owner: BrokerOwner,
        snapshot: RecoverySnapshot,
    ) -> tuple[str, str, str]:
        pinned = (
            snapshot.durable_state_ref,
            snapshot.durable_key_version,
            snapshot.durable_digest,
        )
        if all(value is not None for value in pinned):
            durable_state_ref, durable_key_version, durable_digest = pinned
            assert durable_state_ref is not None
            assert durable_key_version is not None
            assert durable_digest is not None
            return durable_state_ref, durable_key_version, durable_digest
        if any(value is not None for value in pinned):
            raise RuntimeSecretProblem()
        durable_state_ref = self._durable.reference(snapshot.job_id)
        durable_key_version = self._secrets.current_version
        durable_digest = self._secrets.durable_digest_for(
            owner,
            snapshot.job_id,
            snapshot.conversation_id,
            durable_state_ref,
            durable_key_version,
        )
        return durable_state_ref, durable_key_version, durable_digest

    @staticmethod
    def _base_generation(
        owner: BrokerOwner,
        ticket: OperationTicket,
        snapshot: RecoverySnapshot,
    ) -> RecoveryGenerationSnapshot:
        generation = snapshot.generation_record
        if (
            ticket.command is not CommandKind.BEGIN_START
            or ticket.job_id != snapshot.job_id
            or ticket.conversation_id != snapshot.conversation_id
            or ticket.generation is None
            or snapshot.owner != owner
            or snapshot.generation != ticket.generation
            or generation is None
            or generation.generation != ticket.generation
            or generation.start_operation_id != ticket.operation_id
        ):
            raise SegmentCoordinatorProblem(SegmentCoordinatorCode.INTERNAL)
        return generation

    @staticmethod
    def _abandoned_problem(
        generation: RecoveryGenerationSnapshot,
        operation_id: UUID,
    ) -> SegmentCoordinatorProblem:
        try:
            code = SegmentCoordinatorCode(generation.failure_reason_code)
        except (TypeError, ValueError):
            code = SegmentCoordinatorCode.INTERNAL
        return SegmentCoordinatorProblem(code, operation_id=operation_id)

    def _validate_generation(
        self,
        ticket: OperationTicket,
        snapshot: RecoverySnapshot,
        generation: RecoveryGenerationSnapshot,
    ) -> None:
        self._fixed_policy(snapshot)
        if (
            snapshot.durable_state_ref is None
            or snapshot.durable_key_version is None
            or snapshot.durable_digest is None
            or generation.durable_state_ref != snapshot.durable_state_ref
            or generation.durable_key_version != snapshot.durable_key_version
            or generation.durable_digest != snapshot.durable_digest
            or generation.runtime_key_version is None
        ):
            raise RuntimeSecretProblem()
        if generation.state is GenerationState.STARTING:
            if (
                snapshot.state not in {JobState.CREATED, JobState.PARKED}
                or snapshot.current_generation is not None
                or snapshot.pending_operation_id != ticket.operation_id
                or generation.pending_operation_id != ticket.operation_id
                or generation.runtime_ref is not None
                or generation.capability_digest is not None
            ):
                raise SegmentCoordinatorProblem(
                    SegmentCoordinatorCode.STATE_CONFLICT
                )
        elif generation.state is GenerationState.RUNNING:
            if (
                snapshot.state is not JobState.ACTIVE
                or snapshot.current_generation != generation.generation
                or snapshot.pending_operation_id is not None
                or generation.pending_operation_id is not None
                or generation.runtime_ref is None
                or generation.capability_digest is None
            ):
                raise SegmentCoordinatorProblem(
                    SegmentCoordinatorCode.STATE_CONFLICT
                )
        else:
            raise SegmentCoordinatorProblem(SegmentCoordinatorCode.STATE_CONFLICT)

    def _derive_spec(
        self,
        owner: BrokerOwner,
        snapshot: RecoverySnapshot,
        generation: RecoveryGenerationSnapshot,
    ) -> tuple[OpenHandsGenerationSpec, DerivedRuntimeSecrets]:
        assert generation.runtime_key_version is not None
        assert generation.durable_state_ref is not None
        assert generation.durable_key_version is not None
        assert generation.durable_digest is not None
        secrets = self._secrets.derive(
            owner,
            snapshot.job_id,
            snapshot.conversation_id,
            generation.generation,
            generation.durable_state_ref,
            runtime_key_version=generation.runtime_key_version,
            durable_key_version=generation.durable_key_version,
        )
        if not self._secrets.verify_durable(secrets, generation.durable_digest):
            raise RuntimeSecretProblem()
        if generation.capability_digest is not None and not (
            self._secrets.verify_capability(
                secrets, generation.capability_digest
            )
        ):
            raise RuntimeSecretProblem()
        return (
            OpenHandsGenerationSpec(
                operation_id=generation.start_operation_id,
                job_id=snapshot.job_id,
                conversation_id=snapshot.conversation_id,
                generation=generation.generation,
                deadline=generation.execution_lease_deadline,
                relay_capability=secrets.relay_capability,
                session_api_key=secrets.session_api_key,
                conversation_secret=secrets.conversation_secret,
            ),
            secrets,
        )

    @staticmethod
    def _access(
        spec: OpenHandsGenerationSpec,
        endpoint: RelayClientEndpoint,
    ) -> RunningGenerationAccess:
        if endpoint.conversation_id != spec.conversation_id:
            raise RuntimeError("runtime endpoint identity mismatch")
        if (
            endpoint.relay_capability != spec.relay_capability
            or endpoint.session_api_key != spec.session_api_key
        ):
            raise RuntimeError("runtime endpoint credential mismatch")
        return RunningGenerationAccess(
            job_id=spec.job_id,
            conversation_id=spec.conversation_id,
            generation=spec.generation,
            deadline=spec.deadline,
            socket_path=endpoint.socket_path,
            relay_capability=endpoint.relay_capability,
            session_api_key=endpoint.session_api_key,
        )

    async def _cleanup_starting(
        self,
        owner: BrokerOwner,
        ticket: OperationTicket,
        problem: SegmentCoordinatorProblem,
        identity: GenerationRuntimeIdentity | None,
    ) -> None:
        if identity is not None:
            try:
                await self._runtime.release(identity)
            except Exception:
                pass
        assert ticket.job_id is not None
        assert ticket.generation is not None
        try:
            await self._ledger.abandon_generation(
                owner,
                ticket.job_id,
                ticket.generation,
                GenerationState.STARTING,
                problem.code.value,
            )
        except Exception as error:
            raise SegmentCoordinatorProblem(
                SegmentCoordinatorCode.INTERNAL,
                operation_id=ticket.operation_id,
            ) from error

    async def start_segment(
        self,
        owner: BrokerOwner,
        payload: StartSegmentPayload,
    ) -> StartSegmentResult:
        if not isinstance(owner, BrokerOwner):
            raise TypeError("owner must be BrokerOwner")
        if not isinstance(payload, StartSegmentPayload):
            raise TypeError("payload must be StartSegmentPayload")

        ticket: OperationTicket | None = None
        try:
            before = await self._ledger.inspect_job_for_recovery(
                owner, payload.job_id
            )
            self._fixed_policy(before)
            durable_ref, durable_version, durable_digest = (
                self._candidate_durable_metadata(owner, before)
            )
            ticket = await self._ledger.begin_start(
                owner,
                payload.idempotency_key,
                payload.job_id,
                payload.expected_generation,
                self._policy.execution_lease_seconds,
                self._secrets.current_version,
                durable_ref,
                durable_version,
                durable_digest,
            )
        except Exception as error:
            raise self._problem(error) from error

        identity: GenerationRuntimeIdentity | None = None
        starting = False
        authoritative_state_loaded = False
        mark_running_invoked = False
        try:
            snapshot = await self._ledger.inspect_job_for_recovery(
                owner, payload.job_id
            )
            generation = self._base_generation(owner, ticket, snapshot)
            authoritative_state_loaded = True
            if generation.state is GenerationState.ABANDONED:
                raise self._abandoned_problem(
                    generation, ticket.operation_id
                )
            if generation.state not in {
                GenerationState.STARTING,
                GenerationState.RUNNING,
            }:
                raise SegmentCoordinatorProblem(
                    SegmentCoordinatorCode.STATE_CONFLICT,
                    operation_id=ticket.operation_id,
                )
            starting = generation.state is GenerationState.STARTING
            identity = GenerationRuntimeIdentity(
                operation_id=generation.start_operation_id,
                job_id=snapshot.job_id,
                generation=generation.generation,
                deadline=generation.execution_lease_deadline,
            )
            self._validate_generation(ticket, snapshot, generation)
            if self._now() >= generation.execution_lease_deadline:
                raise RuntimeExpired("generation execution deadline elapsed")
            spec, secrets = self._derive_spec(owner, snapshot, generation)
            descriptor = self._durable.describe(
                snapshot.job_id,
                generation.durable_key_version,
                generation.durable_digest,
            )
            if descriptor.durable_state_ref != generation.durable_state_ref:
                raise RuntimeSecretProblem()
            described_endpoint = self._runtime.describe_endpoint(spec)
            await self._durable.ensure(descriptor)
            ensured = await self._runtime.ensure(spec)
            if not isinstance(ensured, EnsuredGeneration):
                raise RuntimeError("runtime ensure returned an invalid value")
            if (
                ensured.handle != identity.opaque_handle
                or ensured.observation.identity != identity
                or ensured.endpoint != described_endpoint
            ):
                raise RuntimeError("runtime ensure identity mismatch")
            if (
                generation.runtime_ref is not None
                and generation.runtime_ref != ensured.handle
            ):
                raise RuntimeSecretProblem()
            mark_running_invoked = True
            completion = await self._ledger.mark_running(
                owner,
                ticket.operation_id,
                snapshot.job_id,
                generation.generation,
                ensured.handle,
                secrets.capability_digest,
            )
            self._validate_completion(ticket, completion)
            access = self._access(spec, described_endpoint)
            return StartSegmentResult(
                operation_id=ticket.operation_id,
                replayed=ticket.replayed or completion.replayed,
                access=access,
            )
        except Exception as error:
            if mark_running_invoked:
                raise SegmentCoordinatorProblem(
                    SegmentCoordinatorCode.INTERNAL,
                    operation_id=ticket.operation_id,
                ) from error
            problem = self._problem(error, operation_id=ticket.operation_id)
            if starting or not authoritative_state_loaded:
                await self._cleanup_starting(owner, ticket, problem, identity)
            raise problem from error

    @staticmethod
    def _validate_completion(
        ticket: OperationTicket,
        completion: OperationResult,
    ) -> None:
        if (
            not isinstance(completion, OperationResult)
            or completion.operation_id != ticket.operation_id
            or completion.job_id != ticket.job_id
            or completion.generation != ticket.generation
            or completion.job_state is not JobState.ACTIVE
            or completion.generation_state is not GenerationState.RUNNING
            or completion.command is not CommandKind.MARK_RUNNING
        ):
            raise RuntimeError("mark-running completion mismatch")

    async def inspect_running_access(
        self,
        owner: BrokerOwner,
        job_id: UUID,
    ) -> RunningGenerationAccess | None:
        if not isinstance(owner, BrokerOwner):
            raise TypeError("owner must be BrokerOwner")
        validate_uuid("job_id", job_id)
        try:
            snapshot = await self._ledger.inspect_job_for_recovery(owner, job_id)
            self._fixed_policy(snapshot)
            generation = snapshot.generation_record
            if (
                snapshot.state is not JobState.ACTIVE
                or snapshot.current_generation is None
                or generation is None
                or generation.state is not GenerationState.RUNNING
                or generation.generation != snapshot.current_generation
                or generation.runtime_ref is None
                or generation.capability_digest is None
                or generation.runtime_key_version is None
                or generation.durable_state_ref is None
                or generation.durable_key_version is None
                or generation.durable_digest is None
                or snapshot.durable_state_ref != generation.durable_state_ref
                or snapshot.durable_key_version != generation.durable_key_version
                or snapshot.durable_digest != generation.durable_digest
                or self._now() >= generation.execution_lease_deadline
            ):
                return None
            spec, _ = self._derive_spec(owner, snapshot, generation)
            identity = spec.identity
            if generation.runtime_ref != identity.opaque_handle:
                return None
            observation = await self._runtime.inspect(identity)
            if observation.identity != identity or not observation.complete:
                return None
            endpoint = self._runtime.describe_endpoint(spec)
            return self._access(spec, endpoint)
        except Exception:
            return None


__all__ = ["BrokerSegmentCoordinator"]
