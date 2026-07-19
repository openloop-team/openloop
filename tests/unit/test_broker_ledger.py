from dataclasses import replace
from datetime import UTC, datetime
import inspect
from uuid import UUID

import pytest

from openloop.broker.errors import InvalidTransition
from openloop.broker.ledger import BrokerLedger
from openloop.broker.models import (
    BrokerOwner,
    CommandKind,
    GenerationState,
    IsolationMode,
    JobAuthorizationMetadata,
    JobState,
    ReleaseTarget,
    TerminalOutcome,
    VerifiedCheckpointReceipt,
)
from openloop.broker.repository import (
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
    canonical_request_json,
    require_generation_transition,
    require_job_transition,
)


OWNER = BrokerOwner("tenant-a", "workload-a")
OTHER_OWNER = BrokerOwner("tenant-b", "workload-b")
JOB_ID = UUID("00000000-0000-4000-8000-000000000001")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000002")
OPERATION_ID = UUID("00000000-0000-4000-8000-000000000003")
OTHER_OPERATION_ID = UUID("00000000-0000-4000-8000-000000000004")


def _receipt(**changes) -> VerifiedCheckpointReceipt:
    values = dict(
        issuer="checkpoint_issuer",
        receipt_id="receipt-0001",
        tenant_id=OWNER.tenant_id,
        job_id=JOB_ID,
        conversation_id=CONVERSATION_ID,
        generation=1,
        barrier_id="barrier-0001",
        artifact_id="artifact-0001",
        base_commit="a" * 40,
        ciphertext_sha256="b" * 64,
        plaintext_sha256="c" * 64,
        byte_count=100,
        store_version="store-v1",
        envelope_version="envelope-v1",
        key_version="key-v1",
        durable_write_sequence=9,
    )
    values.update(changes)
    return VerifiedCheckpointReceipt(**values)


JOB_EDGES = {
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


GENERATION_EDGES = {
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


@pytest.mark.parametrize("command", list(CommandKind))
@pytest.mark.parametrize("current", list(JobState))
@pytest.mark.parametrize("target", list(JobState))
def test_job_transition_matrix_is_exhaustive(command, current, target):
    edge = (command, current, target)
    if edge in JOB_EDGES:
        require_job_transition(command, current, target)
    else:
        with pytest.raises(InvalidTransition):
            require_job_transition(command, current, target)


@pytest.mark.parametrize("command", list(CommandKind))
@pytest.mark.parametrize("current", list(GenerationState))
@pytest.mark.parametrize("target", list(GenerationState))
def test_generation_transition_matrix_is_exhaustive(command, current, target):
    edge = (command, current, target)
    if edge in GENERATION_EDGES:
        require_generation_transition(command, current, target)
    else:
        with pytest.raises(InvalidTransition):
            require_generation_transition(command, current, target)


def test_terminal_and_completed_generation_states_have_no_outgoing_edges():
    for command in CommandKind:
        for target in JobState:
            with pytest.raises(InvalidTransition):
                require_job_transition(command, JobState.TERMINAL, target)
        for current in (GenerationState.RELEASED, GenerationState.ABANDONED):
            for target in GenerationState:
                with pytest.raises(InvalidTransition):
                    require_generation_transition(command, current, target)


def _create_command(**changes) -> CreateJobCommand:
    values = dict(
        owner=OWNER,
        idempotency_key="caller-create-0001",
        operation_id=OPERATION_ID,
        job_id=JOB_ID,
        conversation_id=CONVERSATION_ID,
        profile="default",
        runtime_driver="docker",
        durable_state_driver="postgres",
    )
    values.update(changes)
    return CreateJobCommand(**values)


def test_canonical_create_request_has_frozen_json_and_digest_v1():
    command = _create_command()
    assert canonical_request_json(command) == (
        '{"command":"create_job","request":{"durable_state_driver":"postgres",'
        '"owner":{"tenant_id":"tenant-a","workload_subject":"workload-a"},'
        '"profile":"default","runtime_driver":"docker"},"schema_version":1}'
    )
    assert command.request_digest == (
        "a83f5797265a44b3562351f52291db6329cbf215169388f6c6094bf95e990761"
    )


def test_authorized_create_digest_includes_isolation_but_not_derived_metadata():
    first = _create_command(
        minimum_isolation=IsolationMode.DEDICATED,
        authorization=JobAuthorizationMetadata("cap-v1", 1, "a" * 64),
    )
    rotated = _create_command(
        minimum_isolation=IsolationMode.DEDICATED,
        authorization=JobAuthorizationMetadata("cap-v2", 9, "b" * 64),
    )
    shared = _create_command(
        minimum_isolation=IsolationMode.SHARED,
        authorization=JobAuthorizationMetadata("cap-v1", 1, "a" * 64),
    )
    assert first.request_digest == rotated.request_digest
    assert first.request_digest != shared.request_digest
    assert '"minimum_isolation":"dedicated"' in canonical_request_json(first)
    assert "cap-v1" not in canonical_request_json(first)
    assert "a" * 64 not in canonical_request_json(first)


def test_canonical_request_excludes_idempotency_and_broker_minted_ids():
    original = _create_command()
    changed = _create_command(
        idempotency_key="caller-create-9999",
        operation_id=OTHER_OPERATION_ID,
        job_id=UUID("10000000-0000-4000-8000-000000000001"),
        conversation_id=UUID("10000000-0000-4000-8000-000000000002"),
    )
    assert canonical_request_json(changed) == canonical_request_json(original)
    assert changed.request_digest == original.request_digest


@pytest.mark.parametrize(
    "changed",
    [
        _create_command(owner=OTHER_OWNER),
        _create_command(profile="gpu"),
        _create_command(runtime_driver="containerd"),
        _create_command(durable_state_driver="s3"),
    ],
)
def test_each_create_semantic_field_changes_digest(changed):
    assert changed.request_digest != _create_command().request_digest


def test_existing_job_identity_generation_and_lease_are_digest_semantics():
    command = BeginStartCommand(
        owner=OWNER,
        idempotency_key="caller-start-0001",
        operation_id=OPERATION_ID,
        job_id=JOB_ID,
        expected_generation=0,
        execution_lease_seconds=30,
        runtime_key_version="runtime-v1",
        durable_state_ref="local-openhands:v1:job",
        durable_key_version="durable-v1",
        durable_digest="d" * 64,
    )
    variants = [
        replace(command, owner=OTHER_OWNER),
        replace(command, job_id=CONVERSATION_ID),
        replace(command, expected_generation=1),
        replace(command, execution_lease_seconds=31),
    ]
    assert all(value.request_digest != command.request_digest for value in variants)
    assert replace(
        command,
        idempotency_key="caller-start-9999",
        operation_id=OTHER_OPERATION_ID,
    ).request_digest == command.request_digest
    assert replace(
        command,
        runtime_key_version="runtime-v2",
        durable_state_ref="local-openhands:v1:rotated",
        durable_key_version="durable-v2",
        durable_digest="e" * 64,
    ).request_digest == command.request_digest


def test_receipt_release_target_and_outcome_are_digest_semantics():
    command = BeginReleaseCommand(
        owner=OWNER,
        idempotency_key="caller-release-01",
        operation_id=OPERATION_ID,
        job_id=JOB_ID,
        expected_generation=1,
        receipt=_receipt(),
        target=ReleaseTarget.FINALIZING,
        terminal_outcome=TerminalOutcome.SUCCESS,
    )
    variants = [
        replace(command, receipt=_receipt(barrier_id="barrier-0002")),
        replace(command, target=ReleaseTarget.PARKED, terminal_outcome=None),
        replace(command, terminal_outcome=TerminalOutcome.FAILED),
        replace(command, expected_generation=2),
    ]
    assert all(value.request_digest != command.request_digest for value in variants)


class RecordingRepository:
    def __init__(self):
        self.calls = []

    async def _record(self, name, value):
        self.calls.append((name, value))
        return value

    async def create_job(self, command):
        return await self._record("create_job", command)

    async def begin_start(self, command):
        return await self._record("begin_start", command)

    async def mark_running(self, command):
        return await self._record("mark_running", command)

    async def abandon_generation(self, command):
        return await self._record("abandon_generation", command)

    async def begin_quiesce(self, command):
        return await self._record("begin_quiesce", command)

    async def mark_quiesced(self, command):
        return await self._record("mark_quiesced", command)

    async def begin_release(self, command):
        return await self._record("begin_release", command)

    async def begin_internal_release(self, command):
        return await self._record("begin_internal_release", command)

    async def mark_released(self, command):
        return await self._record("mark_released", command)

    async def begin_finalize(self, command):
        return await self._record("begin_finalize", command)

    async def begin_internal_finalize(self, command):
        return await self._record("begin_internal_finalize", command)

    async def mark_terminal(self, command):
        return await self._record("mark_terminal", command)

    async def inspect_job(self, owner, job_id):
        return await self._record("inspect_job", (owner, job_id))

    async def inspect_job_authorization(self, owner, job_id):
        return await self._record("inspect_job_authorization", (owner, job_id))

    async def inspect_job_for_recovery(self, owner, job_id):
        return await self._record("inspect_job_for_recovery", (owner, job_id))

    async def scan_recovery_candidates(self, after_job_id, limit):
        return await self._record(
            "scan_recovery_candidates", (after_job_id, limit)
        )


class IdFactory:
    def __init__(self):
        self.values = [
            UUID(f"00000000-0000-4000-8000-{number:012d}")
            for number in range(100, 130)
        ]
        self.calls = 0

    def __call__(self):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_repository_protocol_exposes_only_named_lifecycle_methods():
    public_methods = {
        name
        for name, member in BrokerRepository.__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }
    assert public_methods == {
        "create_job",
        "begin_start",
        "mark_running",
        "abandon_generation",
        "begin_quiesce",
        "mark_quiesced",
        "begin_release",
        "mark_released",
        "begin_finalize",
        "mark_terminal",
        "inspect_job",
        "inspect_job_authorization",
        "inspect_job_for_recovery",
        "scan_recovery_candidates",
        "begin_internal_release",
        "begin_internal_finalize",
    }
    assert isinstance(RecordingRepository(), BrokerRepository)


async def test_ledger_builds_and_delegates_every_named_command():
    repository = RecordingRepository()
    ids = IdFactory()
    ledger = BrokerLedger(repository, id_factory=ids)

    created = await ledger.create_job(
        OWNER, "caller-create-0001", "default", "docker", "postgres"
    )
    job_id = created.job_id
    conversation_id = created.conversation_id
    assert isinstance(created, CreateJobCommand)

    started = await ledger.begin_start(
        OWNER,
        "caller-start-0001",
        job_id,
        0,
        30,
        "runtime-key-v1",
        "durable://job-state",
        "durable-key-v1",
        "e" * 64,
    )
    assert isinstance(started, BeginStartCommand)

    running = await ledger.mark_running(
        OWNER,
        started.operation_id,
        job_id,
        1,
        "runtime://handle",
        "d" * 64,
    )
    assert isinstance(running, MarkRunningCommand)

    quiesce = await ledger.begin_quiesce(
        OWNER, "caller-quiesce-01", job_id, 1, "barrier-0001"
    )
    assert isinstance(quiesce, BeginQuiesceCommand)
    assert isinstance(
        await ledger.mark_quiesced(
            OWNER, quiesce.operation_id, job_id, 1
        ),
        MarkQuiescedCommand,
    )

    receipt = _receipt(job_id=job_id, conversation_id=conversation_id)
    release = await ledger.begin_release(
        OWNER,
        "caller-release-01",
        job_id,
        1,
        receipt,
        ReleaseTarget.PARKED,
    )
    assert isinstance(release, BeginReleaseCommand)
    assert isinstance(
        await ledger.mark_released(OWNER, release.operation_id, job_id, 1),
        MarkReleasedCommand,
    )

    finalizing = await ledger.begin_finalize(
        OWNER,
        "caller-finalize-1",
        job_id,
        1,
        TerminalOutcome.SUCCESS,
    )
    assert isinstance(finalizing, BeginFinalizeCommand)
    assert isinstance(
        await ledger.mark_terminal(OWNER, finalizing.operation_id, job_id),
        MarkTerminalCommand,
    )

    abandoned = await ledger.abandon_generation(
        OWNER,
        job_id,
        1,
        GenerationState.RUNNING,
        "runtime_lost",
        TerminalOutcome.FAILED,
    )
    assert isinstance(abandoned, AbandonGenerationCommand)

    assert await ledger.inspect_job(OWNER, job_id) == (OWNER, job_id)
    assert await ledger.inspect_job_for_recovery(OWNER, job_id) == (OWNER, job_id)
    assert [name for name, _ in repository.calls] == [
        "create_job",
        "begin_start",
        "mark_running",
        "begin_quiesce",
        "mark_quiesced",
        "begin_release",
        "mark_released",
        "begin_finalize",
        "mark_terminal",
        "abandon_generation",
        "inspect_job",
        "inspect_job_for_recovery",
    ]


async def test_boundary_validation_fails_before_id_minting_or_repository_access():
    repository = RecordingRepository()
    ids = IdFactory()
    ledger = BrokerLedger(repository, id_factory=ids)

    with pytest.raises(ValueError, match="profile"):
        await ledger.create_job(
            OWNER, "caller-create-0001", "INVALID", "docker", "postgres"
        )
    with pytest.raises(ValueError, match="idempotency"):
        await ledger.begin_start(
            OWNER,
            "short",
            JOB_ID,
            0,
            30,
            "runtime-key-v1",
            "durable://job-state",
            "durable-key-v1",
            "d" * 64,
        )
    with pytest.raises(ValueError, match="execution_lease"):
        await ledger.begin_start(
            OWNER,
            "caller-start-0001",
            JOB_ID,
            0,
            0,
            "runtime-key-v1",
            "durable://job-state",
            "durable-key-v1",
            "d" * 64,
        )

    assert ids.calls == 0
    assert repository.calls == []


async def test_abandonment_mints_once_and_accepts_operation_id_only_for_replay():
    repository = RecordingRepository()
    ids = IdFactory()
    ledger = BrokerLedger(repository, id_factory=ids)

    first = await ledger.abandon_generation(
        OWNER,
        JOB_ID,
        1,
        GenerationState.STARTING,
        "start_failed",
    )
    assert first.operation_id == ids.values[0]
    assert ids.calls == 1

    replay = await ledger.abandon_generation(
        OWNER,
        JOB_ID,
        1,
        GenerationState.STARTING,
        "start_failed",
        replay_operation_id=first.operation_id,
    )
    assert replay.operation_id == first.operation_id
    assert replay.request_digest == first.request_digest
    assert first.replay_operation is False
    assert replay.replay_operation is True
    assert ids.calls == 1


def test_trusted_completion_signatures_have_no_idempotency_key():
    for method_name in (
        "mark_running",
        "mark_quiesced",
        "mark_released",
        "mark_terminal",
    ):
        parameters = inspect.signature(getattr(BrokerLedger, method_name)).parameters
        assert "idempotency_key" not in parameters
