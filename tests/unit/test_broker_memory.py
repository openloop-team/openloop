import asyncio
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from openloop.broker.errors import (
    ConcurrentMutation,
    IdempotencyConflict,
    InvalidTransition,
    JobNotFound,
    OperationMismatch,
    OwnerMismatch,
    ReceiptBindingMismatch,
    StaleGeneration,
)
from openloop.broker.ledger import BrokerLedger
from openloop.broker.memory import InMemoryBrokerRepository
from openloop.broker.models import (
    CommandKind,
    GenerationState,
    IsolationMode,
    JobAuthorizationMetadata,
    JobState,
    OperationSource,
    OperationStatus,
    ReleaseTarget,
    TerminalOutcome,
)
from tests.support.broker_repository_contract import (
    CAPABILITY_DIGEST,
    DURABLE_STATE_REF,
    DURABLE_DIGEST,
    OTHER_OWNER,
    OWNER,
    MutableClock,
    SequenceIds,
    begin_generation_start,
    exercise_complete_lifecycle,
    mark_generation_running,
    quiesce_generation,
    receipt_for,
)


@pytest.fixture
def clock():
    return MutableClock()


@pytest.fixture
def repository(clock):
    return InMemoryBrokerRepository(clock=clock)


@pytest.fixture
def ledger(repository):
    return BrokerLedger(repository, id_factory=SequenceIds())


async def _create(ledger, key="memory-create-0001"):
    return await ledger.create_job(OWNER, key, "default", "docker", "postgres")


async def _create_running(ledger, *, create_key="memory-create-0001"):
    created = await _create(ledger, create_key)
    start = await begin_generation_start(
        ledger,
        idempotency_key="memory-start-00001",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
    )
    await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=start.operation_id,
        generation=1,
    )
    return created, start


async def test_complete_park_resume_finalize_terminal_lifecycle(ledger, repository):
    trace = await exercise_complete_lifecycle(ledger)
    states = [snapshot.state for snapshot in trace.snapshots]
    assert states == [
        JobState.CREATED,
        JobState.CREATED,
        JobState.ACTIVE,
        JobState.ACTIVE,
        JobState.PARKED,
        JobState.FINALIZING,
        JobState.TERMINAL,
    ]
    assert trace.snapshots[1].generation_record.state is GenerationState.STARTING
    assert trace.snapshots[3].generation_record.state is GenerationState.RELEASING
    assert trace.snapshots[4].generation_record.state is GenerationState.RELEASED
    assert trace.snapshots[-1].generation == 2
    assert trace.snapshots[-1].current_generation is None
    assert trace.snapshots[-1].terminal_outcome is TerminalOutcome.SUCCESS
    assert len(await repository.audit_records_for_test()) == 15


async def test_recovery_scan_is_bounded_ordered_and_uses_repository_clock(
    ledger, clock
):
    first = await _create(ledger, "memory-scan-create1")
    second = await _create(ledger, "memory-scan-create2")
    first_start = await begin_generation_start(
        ledger,
        idempotency_key="memory-scan-start01",
        job_id=first.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
    )
    second_start = await begin_generation_start(
        ledger,
        idempotency_key="memory-scan-start02",
        job_id=second.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
    )
    await mark_generation_running(
        ledger,
        job_id=first.job_id,
        operation_id=first_start.operation_id,
        generation=1,
    )
    await mark_generation_running(
        ledger,
        job_id=second.job_id,
        operation_id=second_start.operation_id,
        generation=1,
    )
    await ledger.begin_quiesce(
        OWNER,
        "memory-scan-quiesce",
        second.job_id,
        1,
        "scan-barrier",
    )

    before_expiry = await ledger.scan_recovery_candidates()
    assert [item.job_id for item in before_expiry] == [second.job_id]
    assert before_expiry[0].observed_at == clock.now
    clock.now += timedelta(seconds=31)
    expected = sorted((first.job_id, second.job_id))
    first_page = await ledger.scan_recovery_candidates(limit=1)
    second_page = await ledger.scan_recovery_candidates(first_page[-1].job_id, 1)

    assert [first_page[0].job_id, second_page[0].job_id] == expected


async def test_recovery_scan_reports_finalizing_jobs_without_generation_state(
    ledger,
):
    created, _ = await _create_running(ledger, create_key="memory-scan-finalize")
    await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.RUNNING,
        "operator_abandon",
        TerminalOutcome.FAILED,
    )

    candidates = await ledger.scan_recovery_candidates()

    assert [item.job_id for item in candidates] == [created.job_id]
    assert candidates[0].job_state is JobState.FINALIZING
    assert candidates[0].generation == 1
    assert candidates[0].generation_state is None


async def test_create_exact_replay_returns_original_ids_without_audit(ledger, repository):
    first = await _create(ledger)
    replay = await _create(ledger)
    assert replay.replayed is True
    assert replay.operation_id == first.operation_id
    assert replay.job_id == first.job_id
    assert replay.conversation_id == first.conversation_id
    assert len(await repository.audit_records_for_test()) == 1
    assert len(await repository.operations_for_test()) == 1


async def test_authorized_create_replay_returns_stored_authorization(ledger, repository):
    issued = []

    def authorization_factory(owner, job_id, minimum_isolation):
        metadata = JobAuthorizationMetadata(
            key_version=f"cap-v{len(issued) + 1}",
            epoch=len(issued) + 1,
            capability_digest=f"{len(issued) + 1:064x}",
        )
        issued.append((owner, job_id, minimum_isolation, metadata))
        return metadata

    first = await ledger.create_authorized_job(
        OWNER,
        "memory-auth-create1",
        "default",
        "docker",
        "postgres",
        IsolationMode.DEDICATED,
        authorization_factory,
    )
    replay = await ledger.create_authorized_job(
        OWNER,
        "memory-auth-create1",
        "default",
        "docker",
        "postgres",
        IsolationMode.DEDICATED,
        authorization_factory,
    )
    stored = await ledger.inspect_job_authorization(OWNER, first.job_id)
    assert replay.replayed is True
    assert replay.job_id == first.job_id
    assert len(issued) == 2
    assert issued[0][1] != issued[1][1]
    assert stored.job_id == first.job_id
    assert stored.owner == OWNER
    assert stored.minimum_isolation is IsolationMode.DEDICATED
    assert stored.authorization == issued[0][3]
    assert len(await repository.audit_records_for_test()) == 1


async def test_authorized_create_isolation_change_conflicts(ledger):
    def authorization_factory(owner, job_id, minimum_isolation):
        return JobAuthorizationMetadata("cap-v1", 1, "a" * 64)

    await ledger.create_authorized_job(
        OWNER,
        "memory-auth-create2",
        "default",
        "docker",
        "postgres",
        IsolationMode.SHARED,
        authorization_factory,
    )
    with pytest.raises(IdempotencyConflict):
        await ledger.create_authorized_job(
            OWNER,
            "memory-auth-create2",
            "default",
            "docker",
            "postgres",
            IsolationMode.DEDICATED,
            authorization_factory,
        )


async def test_conflicting_idempotency_reuse_is_rejected(ledger, repository):
    await _create(ledger)
    with pytest.raises(IdempotencyConflict):
        await ledger.create_job(
            OWNER, "memory-create-0001", "gpu", "docker", "postgres"
        )
    assert len(await repository.audit_records_for_test()) == 1


async def test_owner_and_expected_generation_are_fenced(ledger, repository):
    created = await _create(ledger)
    with pytest.raises(OwnerMismatch):
        await ledger.inspect_job(OTHER_OWNER, created.job_id)
    with pytest.raises(JobNotFound):
        await ledger.inspect_job(
            OWNER, UUID("f0000000-0000-4000-8000-000000000001")
        )
    with pytest.raises(StaleGeneration):
        await begin_generation_start(
            ledger,
            idempotency_key="memory-start-stale",
            job_id=created.job_id,
            expected_generation=1,
            execution_lease_seconds=30,
        )
    assert len(await repository.audit_records_for_test()) == 1


async def test_clock_sets_exact_execution_deadline(ledger, repository, clock):
    created = await _create(ledger)
    clock.now = datetime(2026, 7, 17, 13, 0, tzinfo=UTC)
    await begin_generation_start(
        ledger,
        idempotency_key="memory-start-clock",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=86_400,
    )
    recovery = await ledger.inspect_job_for_recovery(OWNER, created.job_id)
    assert recovery.generation_record.execution_lease_deadline == (
        clock.now + timedelta(days=1)
    )
    audit = await repository.audit_records_for_test()
    assert audit[-1].created_at == clock.now


async def test_receipt_must_bind_owner_job_conversation_generation_and_barrier(
    ledger, repository
):
    created, _ = await _create_running(ledger)
    _, _, barrier = await quiesce_generation(
        ledger, job_id=created.job_id, generation=1, suffix="bind"
    )
    valid = dict(
        job_id=created.job_id,
        conversation_id=created.conversation_id,
        generation=1,
        barrier_id=barrier,
        suffix="bind",
    )
    variants = [
        receipt_for(**{**valid, "job_id": UUID(int=99)}),
        receipt_for(**{**valid, "conversation_id": UUID(int=100)}),
        receipt_for(**{**valid, "generation": 2}),
        receipt_for(**{**valid, "barrier_id": "barrier-wrong"}),
    ]
    for index, receipt in enumerate(variants):
        with pytest.raises(ReceiptBindingMismatch):
            await ledger.begin_release(
                OWNER,
                f"memory-release-bad-{index:02d}",
                created.job_id,
                1,
                receipt,
                ReleaseTarget.PARKED,
            )
    assert (await ledger.inspect_job(OWNER, created.job_id)).generation_record.state is (
        GenerationState.QUIESCED
    )
    assert len(await repository.audit_records_for_test()) == 5


async def test_start_failure_abandons_and_never_reuses_generation(ledger, repository):
    created = await _create(ledger)
    first = await begin_generation_start(
        ledger,
        idempotency_key="memory-start-fail1",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
    )
    abandoned = await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.STARTING,
        "start_failed",
    )
    snapshot = await ledger.inspect_job(OWNER, created.job_id)
    assert snapshot.state is JobState.CREATED
    assert snapshot.generation == 1
    assert snapshot.current_generation is None
    assert snapshot.generation_record.state is GenerationState.ABANDONED
    assert snapshot.durable_key_version == "durable-key-v1"
    recovery = await ledger.inspect_job_for_recovery(OWNER, created.job_id)
    assert recovery.durable_state_ref == DURABLE_STATE_REF
    assert recovery.durable_digest == DURABLE_DIGEST

    original = await begin_generation_start(
        ledger,
        idempotency_key="memory-start-fail1",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
        runtime_key_version="runtime-key-rotated",
        durable_state_ref="durable://different-proposal",
        durable_key_version="durable-key-rotated",
        durable_digest="f" * 64,
    )
    assert original.replayed is True
    assert original.operation_id == first.operation_id
    assert original.generation == 1

    second = await begin_generation_start(
        ledger,
        idempotency_key="memory-start-after",
        job_id=created.job_id,
        expected_generation=1,
        execution_lease_seconds=30,
    )
    assert second.generation == 2
    assert first.operation_id != second.operation_id
    recovery = await ledger.inspect_job_for_recovery(OWNER, created.job_id)
    assert recovery.generation_record.runtime_key_version == "runtime-key-v2"
    assert recovery.generation_record.durable_state_ref == DURABLE_STATE_REF
    assert recovery.generation_record.durable_key_version == "durable-key-v1"
    assert recovery.generation_record.durable_digest == DURABLE_DIGEST

    replay = await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.STARTING,
        "start_failed",
        replay_operation_id=abandoned.operation_id,
    )
    assert replay.replayed is True
    assert len(await repository.audit_records_for_test()) == 4


async def test_new_start_cannot_replace_pinned_durable_metadata(ledger):
    created = await _create(ledger)
    await begin_generation_start(
        ledger,
        idempotency_key="memory-pin-start-01",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
    )
    await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.STARTING,
        "start_failed",
    )

    with pytest.raises(ConcurrentMutation):
        await begin_generation_start(
            ledger,
            idempotency_key="memory-pin-start-02",
            job_id=created.job_id,
            expected_generation=1,
            execution_lease_seconds=30,
            durable_state_ref="durable://replacement",
        )

    recovery = await ledger.inspect_job_for_recovery(OWNER, created.job_id)
    assert recovery.durable_state_ref == DURABLE_STATE_REF
    assert recovery.generation == 1


async def test_unknown_abandonment_replay_operation_cannot_mutate(ledger, repository):
    created = await _create(ledger)
    await begin_generation_start(
        ledger,
        idempotency_key="memory-start-00001",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
    )
    with pytest.raises(OperationMismatch):
        await ledger.abandon_generation(
            OWNER,
            created.job_id,
            1,
            GenerationState.STARTING,
            "start_failed",
            replay_operation_id=UUID(
                "f0000000-0000-4000-8000-000000000002"
            ),
        )
    assert (await ledger.inspect_job(OWNER, created.job_id)).generation_record.state is (
        GenerationState.STARTING
    )
    assert len(await repository.audit_records_for_test()) == 2


async def test_active_abandonment_finalizes_with_bounded_outcome(ledger, repository):
    created, _ = await _create_running(ledger)
    abandoned = await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.RUNNING,
        "runtime_lost",
        TerminalOutcome.FAILED,
    )
    snapshot = await ledger.inspect_job(OWNER, created.job_id)
    assert snapshot.state is JobState.FINALIZING
    assert snapshot.terminal_outcome is TerminalOutcome.FAILED
    assert snapshot.current_generation is None
    assert snapshot.generation_record.state is GenerationState.ABANDONED
    replay = await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.RUNNING,
        "runtime_lost",
        TerminalOutcome.FAILED,
        replay_operation_id=abandoned.operation_id,
    )
    assert replay.replayed is True
    assert len(await repository.audit_records_for_test()) == 4


async def test_exact_completion_replays_after_aggregate_advances(ledger, repository):
    created, start = await _create_running(ledger)
    await quiesce_generation(
        ledger, job_id=created.job_id, generation=1, suffix="advance"
    )
    replay = await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=start.operation_id,
        generation=1,
    )
    assert replay.replayed is True
    assert replay.generation_state is GenerationState.RUNNING
    assert (await ledger.inspect_job(OWNER, created.job_id)).generation_record.state is (
        GenerationState.QUIESCED
    )
    assert len(await repository.audit_records_for_test()) == 5


async def test_running_completion_replay_uses_original_generation_metadata(
    ledger, repository
):
    created, first = await _create_running(ledger)
    _, _, barrier = await quiesce_generation(
        ledger, job_id=created.job_id, generation=1, suffix="old"
    )
    receipt = receipt_for(
        job_id=created.job_id,
        conversation_id=created.conversation_id,
        generation=1,
        barrier_id=barrier,
        suffix="old",
    )
    release = await ledger.begin_release(
        OWNER,
        "memory-release-old1",
        created.job_id,
        1,
        receipt,
        ReleaseTarget.PARKED,
    )
    await ledger.mark_released(OWNER, release.operation_id, created.job_id, 1)
    second = await begin_generation_start(
        ledger,
        idempotency_key="memory-start-new001",
        job_id=created.job_id,
        expected_generation=1,
        execution_lease_seconds=30,
    )
    await ledger.mark_running(
        OWNER,
        second.operation_id,
        created.job_id,
        2,
        "runtime://generation-2",
        "f" * 64,
    )

    replay = await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=first.operation_id,
        generation=1,
    )
    assert replay.replayed is True
    assert replay.generation == 1
    assert (await ledger.inspect_job(OWNER, created.job_id)).current_generation == 2
    assert len(await repository.audit_records_for_test()) == 9


async def test_completion_replay_requires_exact_metadata(ledger, repository):
    created, start = await _create_running(ledger)
    with pytest.raises(OperationMismatch):
        await ledger.mark_running(
            OWNER,
            start.operation_id,
            created.job_id,
            1,
            "runtime://different",
            CAPABILITY_DIGEST,
        )
    assert len(await repository.audit_records_for_test()) == 3


async def test_direct_finalize_from_created(ledger, repository):
    created = await _create(ledger)
    ticket = await ledger.begin_finalize(
        OWNER,
        "memory-finalize-01",
        created.job_id,
        0,
        TerminalOutcome.CANCELLED,
    )
    pending = await ledger.inspect_job(OWNER, created.job_id)
    assert pending.state is JobState.FINALIZING
    assert pending.pending_operation_id == ticket.operation_id
    await ledger.mark_terminal(OWNER, ticket.operation_id, created.job_id)
    assert (await ledger.inspect_job(OWNER, created.job_id)).state is JobState.TERMINAL
    assert len(await repository.audit_records_for_test()) == 3


async def test_terminal_job_and_released_generation_are_immutable(ledger):
    trace = await exercise_complete_lifecycle(ledger)
    with pytest.raises(InvalidTransition):
        await ledger.begin_finalize(
            OWNER,
            "memory-finalize-late",
            trace.job_id,
            2,
            TerminalOutcome.SUCCESS,
        )
    with pytest.raises(InvalidTransition):
        await ledger.abandon_generation(
            OWNER,
            trace.job_id,
            2,
            GenerationState.RELEASED,
            "too_late",
            TerminalOutcome.FAILED,
        )


async def test_concurrent_same_key_start_is_one_mutation_plus_replay(
    ledger, repository
):
    created = await _create(ledger)
    first, second = await asyncio.gather(
        begin_generation_start(
            ledger,
            idempotency_key="memory-concurrent-1",
            job_id=created.job_id,
            expected_generation=0,
            execution_lease_seconds=30,
        ),
        begin_generation_start(
            ledger,
            idempotency_key="memory-concurrent-1",
            job_id=created.job_id,
            expected_generation=0,
            execution_lease_seconds=30,
        ),
    )
    assert {first.replayed, second.replayed} == {False, True}
    assert first.operation_id == second.operation_id
    assert first.generation == second.generation == 1
    assert len(await repository.audit_records_for_test()) == 2


async def test_concurrent_different_start_intents_have_one_winner(
    ledger, repository
):
    created = await _create(ledger)
    results = await asyncio.gather(
        begin_generation_start(
            ledger,
            idempotency_key="memory-concurrent-a",
            job_id=created.job_id,
            expected_generation=0,
            execution_lease_seconds=30,
        ),
        begin_generation_start(
            ledger,
            idempotency_key="memory-concurrent-b",
            job_id=created.job_id,
            expected_generation=0,
            execution_lease_seconds=30,
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(
        isinstance(result, (InvalidTransition, StaleGeneration)) for result in results
    ) == 1
    assert len(await repository.audit_records_for_test()) == 2
    snapshot = await ledger.inspect_job(OWNER, created.job_id)
    assert snapshot.generation == 1
    assert snapshot.generation_record.state is GenerationState.STARTING


async def test_audit_and_public_results_contain_no_protected_values(
    ledger, repository
):
    created, _ = await _create_running(ledger)
    public = await ledger.inspect_job(OWNER, created.job_id)
    audit = await repository.audit_records_for_test()
    rendered = repr((asdict(public), audit))
    for protected in (
        "runtime://generation-1",
        DURABLE_STATE_REF,
        CAPABILITY_DIGEST,
        DURABLE_DIGEST,
    ):
        assert protected not in rendered
    assert all(item.owner == OWNER for item in audit)
    assert [item.command for item in audit] == [
        CommandKind.CREATE_JOB,
        CommandKind.BEGIN_START,
        CommandKind.MARK_RUNNING,
    ]


async def test_abandonment_fails_the_superseded_pending_operation(
    ledger, repository
):
    created = await _create(ledger)
    start = await begin_generation_start(
        ledger,
        idempotency_key="memory-start-00001",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
    )
    await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.STARTING,
        "start_failed",
    )
    operation = await repository.operation_for_test(start.operation_id)
    assert operation.source is OperationSource.CALLER
    assert operation.status is OperationStatus.FAILED
