from datetime import UTC, datetime
import os
from pathlib import Path
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from openloop.broker.ledger import BrokerLedger
from openloop.broker.memory import InMemoryBrokerRepository
from openloop.broker.models import (
    BrokerOwner,
    GenerationState,
    JobState,
    ReleaseTarget,
    TerminalOutcome,
    VerifiedCheckpointReceipt,
)
from openloop.broker_control.coordinator import BrokerSegmentCoordinator
from openloop.broker_control.durable import LocalDurableStateAdapter
from openloop.broker_control.receipts import (
    CheckpointReceiptIssuer,
    CheckpointReceiptVerifier,
)
from openloop.broker_control.secrets import (
    RuntimeSecretAuthority,
    RuntimeSecretRootRing,
)
from openloop.broker_rpc.coordinator import (
    BrokerRpcPolicy,
    SegmentCoordinatorCode,
    SegmentCoordinatorProblem,
)
from openloop.broker_rpc.keys import VerificationKeySet
from openloop.broker_rpc.models import (
    FinalizeJobPayload,
    QuiesceSegmentPayload,
    ReleaseSegmentPayload,
    StartSegmentPayload,
)
from openloop.broker_runtime.contract import (
    GenerationRuntimeIdentity,
    RuntimeHealthFailure,
)
from openloop.broker_runtime.memory import InMemoryRuntimeDriver
from tests.support.broker_repository_contract import MutableClock, SequenceIds


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
OWNER = BrokerOwner("tenant-control", "workload-control")
_RECEIPT_PRIVATE_KEY = Ed25519PrivateKey.generate()
RECEIPT_ISSUER = CheckpointReceiptIssuer(
    private_key=_RECEIPT_PRIVATE_KEY,
    key_id="receipt-v1",
    issuer="checkpoint-store",
)


def _receipt_verifier() -> CheckpointReceiptVerifier:
    return CheckpointReceiptVerifier(
        public_keys=VerificationKeySet(
            {"receipt-v1": _RECEIPT_PRIVATE_KEY.public_key()}
        ),
        issuer="checkpoint-store",
    )


def _signed_receipt(
    *,
    job_id,
    conversation_id,
    generation: int,
    barrier_id: str,
    suffix: str,
    issuer: CheckpointReceiptIssuer = RECEIPT_ISSUER,
):
    return issuer.issue(
        VerifiedCheckpointReceipt(
            issuer="checkpoint-store",
            receipt_id=f"receipt-{suffix}",
            tenant_id=OWNER.tenant_id,
            job_id=job_id,
            conversation_id=conversation_id,
            generation=generation,
            barrier_id=barrier_id,
            artifact_id=f"artifact-{suffix}",
            base_commit="c" * 40,
            ciphertext_sha256="d" * 64,
            plaintext_sha256="e" * 64,
            byte_count=1024,
            store_version="store-v1",
            envelope_version="envelope-v1",
            key_version="key-v1",
            durable_write_sequence=generation,
        )
    )


class CountingRuntime(InMemoryRuntimeDriver):
    def __init__(self, *, clock):
        super().__init__(clock=clock, maximum_lifetime_seconds=600)
        self.ensure_calls = 0
        self.inspect_calls = 0
        self.quiesce_calls = 0
        self.release_calls = 0
        self.fail_ensure = False

    async def ensure(self, spec):
        self.ensure_calls += 1
        if self.fail_ensure:
            raise RuntimeHealthFailure("injected health failure")
        return await super().ensure(spec)

    async def inspect(self, identity):
        self.inspect_calls += 1
        return await super().inspect(identity)

    async def quiesce(self, spec):
        self.quiesce_calls += 1
        return await super().quiesce(spec)

    async def release(self, identity):
        self.release_calls += 1
        return await super().release(identity)


class CountingDurable(LocalDurableStateAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ensure_calls = 0

    async def ensure(self, descriptor):
        self.ensure_calls += 1
        await super().ensure(descriptor)


class AmbiguousLedger(BrokerLedger):
    async def mark_running(self, *args, **kwargs):
        await super().mark_running(*args, **kwargs)
        raise RuntimeError("injected lost commit acknowledgement")


class FailRecoveryRereadLedger(BrokerLedger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_reads = 0

    async def inspect_job_for_recovery(self, *args, **kwargs):
        self.recovery_reads += 1
        if self.recovery_reads == 2:
            raise RuntimeError("injected protected reread failure")
        return await super().inspect_job_for_recovery(*args, **kwargs)


def _state_root(tmp_path: Path) -> Path:
    root = tmp_path / "durable"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


async def _fixture(
    tmp_path: Path,
    *,
    ambiguous: bool = False,
    fail_reread: bool = False,
):
    clock = MutableClock(NOW)
    repository = InMemoryBrokerRepository(clock=clock)
    ledger_type = BrokerLedger
    if ambiguous:
        ledger_type = AmbiguousLedger
    elif fail_reread:
        ledger_type = FailRecoveryRereadLedger
    ledger = ledger_type(repository, id_factory=SequenceIds(start=100))
    runtime = CountingRuntime(clock=clock)
    durable = CountingDurable(
        state_root=_state_root(tmp_path),
        uid=os.getuid(),
        gid=os.getgid(),
    )
    secrets = RuntimeSecretAuthority(
        RuntimeSecretRootRing(
            {"runtime-v1": bytes(range(32))},
            current_version="runtime-v1",
        )
    )
    policy = BrokerRpcPolicy("default", "memory", "local", 300)
    coordinator = BrokerSegmentCoordinator(
        ledger=ledger,
        policy=policy,
        runtime_driver=runtime,
        secret_authority=secrets,
        durable_state_adapter=durable,
        receipt_verifier=_receipt_verifier(),
        clock=clock,
    )
    created = await ledger.create_job(
        OWNER,
        "create-control-job",
        policy.profile,
        policy.runtime_driver,
        policy.durable_state_driver,
    )
    return (
        coordinator,
        ledger,
        repository,
        runtime,
        durable,
        secrets,
        policy,
        created.job_id,
    )


async def test_start_replay_and_inspection_reconstruct_identical_access(tmp_path):
    coordinator, ledger, _, runtime, durable, _, _, job_id = await _fixture(
        tmp_path
    )
    payload = StartSegmentPayload(job_id, 0, "start-control-segment")

    first = await coordinator.start_segment(OWNER, payload)
    replay = await coordinator.start_segment(OWNER, payload)
    inspected = await coordinator.inspect_running_access(OWNER, job_id)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.operation_id == first.operation_id
    assert replay.access == first.access == inspected
    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.ACTIVE
    assert snapshot.generation_record.state is GenerationState.RUNNING
    assert snapshot.generation_record.runtime_ref is not None
    assert snapshot.generation_record.capability_digest is not None
    assert durable.ensure_calls == 2
    assert runtime.ensure_calls == 2
    assert (durable.binding.state_root / str(job_id) / "agent-server").is_dir()


async def test_checkpoint_release_and_finalize_lifecycle_is_replay_safe(tmp_path):
    coordinator, ledger, _, runtime, _, _, _, job_id = await _fixture(tmp_path)
    first_start = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "start-lifecycle-first")
    )
    first_quiesce_payload = QuiesceSegmentPayload(
        job_id, 1, "quiesce-lifecycle-first", "barrier-first"
    )

    first_quiesce = await coordinator.quiesce_segment(
        OWNER, first_quiesce_payload
    )
    first_quiesce_replay = await coordinator.quiesce_segment(
        OWNER, first_quiesce_payload
    )

    assert first_quiesce.replayed is False
    assert first_quiesce_replay.replayed is True
    assert first_quiesce_replay.access == first_quiesce.access
    assert first_quiesce.access.job_id == first_start.access.job_id
    assert runtime.quiesce_calls == 2
    assert await coordinator.inspect_running_access(OWNER, job_id) is None

    first_receipt = _signed_receipt(
        job_id=job_id,
        conversation_id=first_start.access.conversation_id,
        generation=1,
        barrier_id="barrier-first",
        suffix="first",
    )
    first_release_payload = ReleaseSegmentPayload(
        job_id,
        1,
        "release-lifecycle-first",
        first_receipt,
        ReleaseTarget.PARKED,
    )
    first_release = await coordinator.release_segment(
        OWNER, first_release_payload
    )
    first_release_replay = await coordinator.release_segment(
        OWNER, first_release_payload
    )

    assert first_release.job_state is JobState.PARKED
    assert first_release_replay.replayed is True
    assert runtime.release_calls == 2

    second_start = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 1, "start-lifecycle-second")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 2, "quiesce-lifecycle-second", "barrier-second"
        ),
    )
    second_receipt = _signed_receipt(
        job_id=job_id,
        conversation_id=second_start.access.conversation_id,
        generation=2,
        barrier_id="barrier-second",
        suffix="second",
    )
    finalizing = await coordinator.release_segment(
        OWNER,
        ReleaseSegmentPayload(
            job_id,
            2,
            "release-lifecycle-second",
            second_receipt,
            ReleaseTarget.FINALIZING,
            TerminalOutcome.SUCCESS,
        ),
    )
    finalize_payload = FinalizeJobPayload(
        job_id, 2, "finalize-lifecycle", TerminalOutcome.SUCCESS
    )
    terminal = await coordinator.finalize_job(OWNER, finalize_payload)
    terminal_replay = await coordinator.finalize_job(OWNER, finalize_payload)

    assert finalizing.job_state is JobState.FINALIZING
    assert terminal.job_state is JobState.TERMINAL
    assert terminal_replay.replayed is True
    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.TERMINAL
    assert snapshot.terminal_outcome is TerminalOutcome.SUCCESS


async def test_release_rejects_invalid_signature_before_ledger_transition(tmp_path):
    coordinator, ledger, _, _, _, _, _, job_id = await _fixture(tmp_path)
    started = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "start-invalid-receipt")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 1, "quiesce-invalid-receipt", "barrier-invalid"
        ),
    )
    attacker = CheckpointReceiptIssuer(
        private_key=Ed25519PrivateKey.generate(),
        key_id="receipt-v1",
        issuer="checkpoint-store",
    )
    receipt = _signed_receipt(
        job_id=job_id,
        conversation_id=started.access.conversation_id,
        generation=1,
        barrier_id="barrier-invalid",
        suffix="invalid",
        issuer=attacker,
    )

    with pytest.raises(SegmentCoordinatorProblem) as problem:
        await coordinator.release_segment(
            OWNER,
            ReleaseSegmentPayload(
                job_id,
                1,
                "release-invalid-receipt",
                receipt,
                ReleaseTarget.PARKED,
            ),
        )

    assert problem.value.code is SegmentCoordinatorCode.INVALID_RECEIPT
    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.state is JobState.ACTIVE
    assert snapshot.generation_record.state is GenerationState.QUIESCED


async def test_release_rejects_signed_receipt_with_wrong_barrier_binding(tmp_path):
    coordinator, ledger, _, _, _, _, _, job_id = await _fixture(tmp_path)
    started = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "start-wrong-barrier")
    )
    await coordinator.quiesce_segment(
        OWNER,
        QuiesceSegmentPayload(
            job_id, 1, "quiesce-wrong-barrier", "barrier-authoritative"
        ),
    )
    receipt = _signed_receipt(
        job_id=job_id,
        conversation_id=started.access.conversation_id,
        generation=1,
        barrier_id="barrier-wrong",
        suffix="wrong-barrier",
    )

    with pytest.raises(SegmentCoordinatorProblem) as problem:
        await coordinator.release_segment(
            OWNER,
            ReleaseSegmentPayload(
                job_id,
                1,
                "release-wrong-barrier",
                receipt,
                ReleaseTarget.PARKED,
            ),
        )

    assert problem.value.code is SegmentCoordinatorCode.STATE_CONFLICT
    snapshot = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert snapshot.generation_record.state is GenerationState.QUIESCED


async def test_same_key_replay_uses_persisted_version_after_root_rotation(
    tmp_path,
):
    (
        coordinator,
        ledger,
        _,
        runtime,
        durable,
        _,
        policy,
        job_id,
    ) = await _fixture(tmp_path)
    payload = StartSegmentPayload(job_id, 0, "start-before-root-rotation")
    first = await coordinator.start_segment(OWNER, payload)
    rotated = BrokerSegmentCoordinator(
        ledger=ledger,
        policy=policy,
        runtime_driver=runtime,
        secret_authority=RuntimeSecretAuthority(
            RuntimeSecretRootRing(
                {
                    "runtime-v1": bytes(range(32)),
                    "runtime-v2": bytes(reversed(range(32))),
                },
                current_version="runtime-v2",
            )
        ),
        durable_state_adapter=durable,
        receipt_verifier=_receipt_verifier(),
        clock=lambda: NOW,
    )

    replay = await rotated.start_segment(OWNER, payload)

    assert replay.replayed is True
    assert replay.access == first.access
    recovery = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert recovery.generation_record.runtime_key_version == "runtime-v1"


async def test_inspection_is_read_only_and_returns_none_for_missing_runtime(tmp_path):
    coordinator, ledger, _, runtime, durable, _, _, job_id = await _fixture(
        tmp_path
    )
    await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 0, "start-inspect-read-only")
    )
    recovery = await ledger.inspect_job_for_recovery(OWNER, job_id)
    generation = recovery.generation_record
    identity = GenerationRuntimeIdentity(
        generation.start_operation_id,
        job_id,
        generation.generation,
        generation.execution_lease_deadline,
    )
    await runtime.release(identity)
    ensure_calls = runtime.ensure_calls
    release_calls = runtime.release_calls
    durable_calls = durable.ensure_calls

    assert await coordinator.inspect_running_access(OWNER, job_id) is None
    assert runtime.ensure_calls == ensure_calls
    assert runtime.release_calls == release_calls
    assert durable.ensure_calls == durable_calls


async def test_pre_completion_runtime_failure_releases_and_abandons(tmp_path):
    coordinator, ledger, _, runtime, _, _, _, job_id = await _fixture(tmp_path)
    payload = StartSegmentPayload(job_id, 0, "start-runtime-failure")
    runtime.fail_ensure = True

    with pytest.raises(SegmentCoordinatorProblem) as first:
        await coordinator.start_segment(OWNER, payload)
    assert first.value.code is SegmentCoordinatorCode.RUNTIME_UNAVAILABLE
    assert runtime.release_calls == 1
    recovery = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert recovery.state is JobState.CREATED
    assert recovery.generation_record.state is GenerationState.ABANDONED

    with pytest.raises(SegmentCoordinatorProblem) as replay:
        await coordinator.start_segment(OWNER, payload)
    assert replay.value.code is SegmentCoordinatorCode.RUNTIME_UNAVAILABLE
    assert runtime.ensure_calls == 1
    assert runtime.release_calls == 1

    runtime.fail_ensure = False
    retried = await coordinator.start_segment(
        OWNER, StartSegmentPayload(job_id, 1, "start-runtime-retry")
    )
    assert retried.access.generation == 2


async def test_exception_after_mark_running_never_cleans_up(tmp_path):
    (
        coordinator,
        ledger,
        repository,
        runtime,
        durable,
        secrets,
        policy,
        job_id,
    ) = await _fixture(tmp_path, ambiguous=True)
    payload = StartSegmentPayload(job_id, 0, "start-ambiguous-commit")

    with pytest.raises(SegmentCoordinatorProblem) as problem:
        await coordinator.start_segment(OWNER, payload)
    assert problem.value.code is SegmentCoordinatorCode.INTERNAL
    assert runtime.release_calls == 0
    recovery = await ledger.inspect_job_for_recovery(OWNER, job_id)
    assert recovery.state is JobState.ACTIVE
    assert recovery.generation_record.state is GenerationState.RUNNING

    recovered = BrokerSegmentCoordinator(
        ledger=BrokerLedger(repository, id_factory=SequenceIds(start=900)),
        policy=policy,
        runtime_driver=runtime,
        secret_authority=secrets,
        durable_state_adapter=durable,
        receipt_verifier=_receipt_verifier(),
        clock=lambda: NOW,
    )
    replay = await recovered.start_segment(OWNER, payload)
    assert replay.replayed is True
    assert replay.access.generation == 1
    assert runtime.release_calls == 0


async def test_failed_authoritative_reread_abandons_without_runtime_effects(
    tmp_path,
):
    (
        coordinator,
        _,
        repository,
        runtime,
        _,
        _,
        _,
        job_id,
    ) = await _fixture(tmp_path, fail_reread=True)

    with pytest.raises(SegmentCoordinatorProblem) as problem:
        await coordinator.start_segment(
            OWNER,
            StartSegmentPayload(job_id, 0, "start-reread-failure"),
        )
    assert problem.value.code is SegmentCoordinatorCode.INTERNAL
    assert runtime.ensure_calls == 0
    assert runtime.release_calls == 0
    recovery = await BrokerLedger(repository).inspect_job_for_recovery(
        OWNER, job_id
    )
    assert recovery.state is JobState.CREATED
    assert recovery.generation_record.state is GenerationState.ABANDONED


async def test_constructor_rejects_lease_above_runtime_maximum(tmp_path):
    clock = MutableClock(NOW)
    ledger = BrokerLedger(InMemoryBrokerRepository(clock=clock))
    runtime = InMemoryRuntimeDriver(clock=clock, maximum_lifetime_seconds=60)
    durable = LocalDurableStateAdapter(
        state_root=_state_root(tmp_path),
        uid=os.getuid(),
        gid=os.getgid(),
    )
    authority = RuntimeSecretAuthority(
        RuntimeSecretRootRing({"v1": b"x" * 32}, current_version="v1")
    )

    with pytest.raises(ValueError, match="runtime maximum"):
        BrokerSegmentCoordinator(
            ledger=ledger,
            policy=BrokerRpcPolicy("default", "memory", "local", 61),
            runtime_driver=runtime,
            secret_authority=authority,
            durable_state_adapter=durable,
            receipt_verifier=_receipt_verifier(),
            clock=clock,
        )
