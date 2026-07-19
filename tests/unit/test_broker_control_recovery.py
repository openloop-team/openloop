import asyncio
from datetime import timedelta

from openloop.broker.models import (
    CommandKind,
    GenerationState,
    JobState,
    OperationSource,
    RecoveryCandidate,
    ReleaseTarget,
    TerminalOutcome,
)
from openloop.broker_control.recovery import (
    BrokerLifecycleReconciler,
    RecoveryOutcome,
)
from openloop.broker_rpc.models import QuiesceSegmentPayload, StartSegmentPayload
from tests.unit.test_broker_control_coordinator import (
    NOW,
    OWNER,
    _fixture,
    _receipt_verifier,
    _signed_receipt,
)


class ReceiptLocator:
    def __init__(self, token=None):
        self.token = token
        self.keys = []

    async def lookup(self, key):
        self.keys.append(key)
        return self.token


class SequenceReceiptLocator(ReceiptLocator):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = list(tokens)

    async def lookup(self, key):
        self.keys.append(key)
        result = self.tokens.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _reconciler(ledger, coordinator, _repository, locator, *, page_limit=100):
    return BrokerLifecycleReconciler(
        ledger=ledger,
        coordinator=coordinator,
        receipt_locator=locator,
        receipt_verifier=_receipt_verifier(),
        page_limit=page_limit,
    )


async def test_recovery_completes_quiescing_before_deadline(tmp_path):
    coordinator, ledger, repository, runtime, _, _, _, job_id = await _fixture(
        tmp_path
    )
    await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-0001")
    )
    ticket = await ledger.begin_quiesce(
        OWNER,
        "recovery-quiesce-01",
        job_id,
        1,
        "recovery-barrier-1",
    )

    report = await _reconciler(
        ledger, coordinator, repository, ReceiptLocator()
    ).run_pass()

    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.generation_record.state is GenerationState.QUIESCED
    assert snapshot.pending_operation_id is None
    assert ticket.operation_id != snapshot.pending_operation_id
    assert report.repaired == 1
    assert report.items[0].outcome is RecoveryOutcome.REPAIRED
    assert runtime.quiesce_calls == 1


async def test_recovery_parks_quiesced_generation_with_valid_receipt(tmp_path):
    coordinator, ledger, repository, runtime, _, _, _, job_id = await _fixture(
        tmp_path
    )
    started = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-0002")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 1, "recovery-quiesce-02", "recovery-barrier-2"
        ),
    )
    token = _signed_receipt(
        job_id=job_id,
        conversation_id=started.access.conversation_id,
        generation=1,
        barrier_id="recovery-barrier-2",
        suffix="recovery-2",
    )

    report = await _reconciler(
        ledger, coordinator, repository, ReceiptLocator(token)
    ).run_pass()

    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.PARKED
    assert snapshot.generation_record.state is GenerationState.RELEASED
    assert report.repaired == 1
    assert runtime.release_calls == 1
    internal_release = next(
        operation
        for operation in await repository.operations_for_test()
        if operation.command is CommandKind.BEGIN_RELEASE
    )
    assert internal_release.source is OperationSource.INTERNAL
    assert internal_release.idempotency_key is None


async def test_recovery_expires_running_and_finalizes_failed(tmp_path):
    coordinator, ledger, repository, runtime, _, _, _, job_id = await _fixture(
        tmp_path
    )
    await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-0003")
    )
    repository._clock.now = NOW + timedelta(seconds=301)

    report = await _reconciler(
        ledger, coordinator, repository, ReceiptLocator()
    ).run_pass()

    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.TERMINAL
    assert snapshot.terminal_outcome is TerminalOutcome.FAILED
    assert snapshot.generation_record.state is GenerationState.ABANDONED
    assert report.failed_closed == 1
    assert runtime.release_calls == 1
    internal_finalize = next(
        operation
        for operation in await repository.operations_for_test()
        if operation.command is CommandKind.BEGIN_FINALIZE
    )
    assert internal_finalize.source is OperationSource.INTERNAL
    assert internal_finalize.idempotency_key is None


async def test_recovery_expired_quiesced_performs_second_lookup_and_fails_closed(
    tmp_path,
):
    coordinator, ledger, repository, runtime, _, _, _, job_id = await _fixture(
        tmp_path
    )
    await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-0004")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 1, "recovery-quiesce-04", "recovery-barrier-4"
        ),
    )
    repository._clock.now = NOW + timedelta(seconds=301)
    locator = ReceiptLocator()

    report = await _reconciler(
        ledger, coordinator, repository, locator
    ).run_pass()

    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.TERMINAL
    assert report.failed_closed == 1
    assert len(locator.keys) == 2
    assert runtime.release_calls == 1


async def test_recovery_expired_quiesced_parks_when_second_lookup_finds_receipt(
    tmp_path,
):
    coordinator, ledger, repository, runtime, _, _, _, job_id = await _fixture(
        tmp_path
    )
    started = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-0005")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 1, "recovery-quiesce-05", "recovery-barrier-5"
        ),
    )
    repository._clock.now = NOW + timedelta(seconds=301)
    token = _signed_receipt(
        job_id=job_id,
        conversation_id=started.access.conversation_id,
        generation=1,
        barrier_id="recovery-barrier-5",
        suffix="recovery-5",
    )
    locator = SequenceReceiptLocator([None, token])

    report = await _reconciler(
        ledger, coordinator, repository, locator
    ).run_pass()

    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.PARKED
    assert report.repaired == 1
    assert len(locator.keys) == 2
    assert runtime.release_calls == 2


async def test_recovery_expired_quiesced_retries_unavailable_second_lookup(
    tmp_path,
):
    coordinator, ledger, repository, runtime, _, _, _, job_id = await _fixture(
        tmp_path
    )
    started = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-lookup")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 1, "recovery-quiesce-lookup", "recovery-barrier-lookup"
        ),
    )
    repository._clock.now = NOW + timedelta(seconds=301)
    token = _signed_receipt(
        job_id=job_id,
        conversation_id=started.access.conversation_id,
        generation=1,
        barrier_id="recovery-barrier-lookup",
        suffix="recovery-lookup",
    )
    locator = SequenceReceiptLocator(
        [None, OSError("transient receipt lookup failure"), token]
    )
    reconciler = _reconciler(ledger, coordinator, repository, locator)

    first_report = await reconciler.run_pass()

    first_snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert first_snapshot.state is JobState.ACTIVE
    assert first_snapshot.generation_record.state is GenerationState.QUIESCED
    assert first_report.error == 1
    assert first_report.items[0].reason_code == "checkpoint_lookup_unavailable"
    assert runtime.release_calls == 1

    second_report = await reconciler.run_pass()

    second_snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert second_snapshot.state is JobState.PARKED
    assert second_report.repaired == 1
    assert runtime.release_calls == 2


async def test_recovery_running_uses_repository_observation_and_reason(
    tmp_path, monkeypatch
):
    coordinator, ledger, repository, runtime, _, _, _, job_id = await _fixture(
        tmp_path
    )
    await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-clock")
    )
    candidate = RecoveryCandidate(
        owner=OWNER,
        job_id=job_id,
        generation=1,
        job_state=JobState.ACTIVE,
        generation_state=GenerationState.RUNNING,
        observed_at=NOW,
    )

    async def scan_recovery_candidates(_cursor, _limit):
        return (candidate,)

    monkeypatch.setattr(
        ledger, "scan_recovery_candidates", scan_recovery_candidates
    )

    report = await _reconciler(
        ledger, coordinator, repository, ReceiptLocator()
    ).run_pass()

    assert report.deferred == 1
    assert report.items[0].reason_code == "execution_lease_active"
    assert runtime.release_calls == 0


async def test_concurrent_recovery_passes_converge_on_one_parked_outcome(
    tmp_path,
):
    coordinator, ledger, repository, _, _, _, _, job_id = await _fixture(tmp_path)
    started = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-0006")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 1, "recovery-quiesce-06", "recovery-barrier-6"
        ),
    )
    token = _signed_receipt(
        job_id=job_id,
        conversation_id=started.access.conversation_id,
        generation=1,
        barrier_id="recovery-barrier-6",
        suffix="recovery-6",
    )
    first = _reconciler(
        ledger, coordinator, repository, ReceiptLocator(token), page_limit=1
    )
    second = _reconciler(
        ledger, coordinator, repository, ReceiptLocator(token), page_limit=1
    )

    reports = await asyncio.gather(first.run_pass(), second.run_pass())

    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.PARKED
    assert sum(report.repaired for report in reports) >= 1
    assert sum(report.failed_closed + report.error for report in reports) == 0
    assert (await first.run_pass()).total == 0


async def test_recovery_completes_persisted_releasing_intent(tmp_path):
    coordinator, ledger, repository, runtime, _, _, _, job_id = await _fixture(
        tmp_path
    )
    started = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "recovery-start-0007")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 1, "recovery-quiesce-07", "recovery-barrier-7"
        ),
    )
    receipt = _receipt_verifier().verify(
        _signed_receipt(
            job_id=job_id,
            conversation_id=started.access.conversation_id,
            generation=1,
            barrier_id="recovery-barrier-7",
            suffix="recovery-7",
        )
    )
    await ledger.begin_internal_release(
        OWNER, job_id, 1, receipt, ReleaseTarget.PARKED
    )

    report = await _reconciler(
        ledger, coordinator, repository, ReceiptLocator()
    ).run_pass()

    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.PARKED
    assert report.repaired == 1
    assert runtime.release_calls == 1


async def test_recovery_completes_pending_finalize_intent(tmp_path):
    coordinator, ledger, repository, _, _, _, _, job_id = await _fixture(tmp_path)
    await ledger.begin_finalize(
        OWNER,
        "recovery-finalize-1",
        job_id,
        0,
        TerminalOutcome.CANCELLED,
    )

    report = await _reconciler(
        ledger, coordinator, repository, ReceiptLocator()
    ).run_pass()

    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.TERMINAL
    assert snapshot.terminal_outcome is TerminalOutcome.CANCELLED
    assert report.repaired == 1
