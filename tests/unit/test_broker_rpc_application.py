from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from openloop.broker.ledger import BrokerLedger
from openloop.broker.memory import InMemoryBrokerRepository
from openloop.broker.models import (
    BrokerOwner,
    GenerationState,
    IsolationMode,
    JobState,
    ReleaseTarget,
    SignedCheckpointReceipt,
    TerminalOutcome,
)
from openloop.broker_rpc.application import BrokerRpcApplication, BrokerRpcPolicy
from openloop.broker_rpc.audit import (
    InMemoryRpcAuditSink,
    PeerCredentials,
    RpcAuditProblem,
)
from openloop.broker_rpc.capability import (
    CapabilityRootRing,
    JobCapability,
    JobCapabilityAuthority,
)
from openloop.broker_rpc.coordinator import (
    SegmentCoordinatorCode,
    SegmentCoordinatorProblem,
)
from openloop.broker_rpc.errors import RpcErrorCode
from openloop.broker_rpc.identity import (
    WorkloadIdentityIssuer,
    WorkloadIdentityVerifier,
    WorkloadIntent,
)
from openloop.broker_rpc.models import (
    RPC_VERSION,
    CheckpointGenerationAccess,
    CreateJobPayload,
    CreateJobResult,
    FinalizeJobPayload,
    FinalizeJobResult,
    InspectJobPayload,
    InspectJobResult,
    QuiesceSegmentPayload,
    QuiesceSegmentResult,
    ReleaseSegmentPayload,
    ReleaseSegmentResult,
    RunningGenerationAccess,
    RpcRequest,
    StartSegmentPayload,
    StartSegmentResult,
)
from tests.support.broker_repository_contract import MutableClock, SequenceIds


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
OWNER = BrokerOwner("tenant-a", "workload-a")
OTHER_OWNER = BrokerOwner("tenant-b", "workload-b")
WORKER_ID = UUID("00000000-0000-4000-8000-000000000401")
ASSIGNMENT_ID = UUID("00000000-0000-4000-8000-000000000402")
PEER = PeerCredentials(pid=4001, uid=1000, gid=1000)


class FailOnceAuditSink(InMemoryRpcAuditSink):
    def __init__(self):
        super().__init__(clock=lambda: NOW)
        self.failed = False

    async def append(self, record):
        if not self.failed:
            self.failed = True
            raise RpcAuditProblem()
        return await super().append(record)


class FakeSegmentCoordinator:
    def __init__(self):
        self.start_calls = []
        self.quiesce_calls = []
        self.release_calls = []
        self.finalize_calls = []
        self.inspect_calls = []
        self.start_result = None
        self.quiesce_result = None
        self.release_result = None
        self.finalize_result = None
        self.start_problem = None
        self.quiesce_problem = None
        self.release_problem = None
        self.finalize_problem = None
        self.inspect_access = None

    async def start_segment(self, owner, payload):
        self.start_calls.append((owner, payload))
        if self.start_problem is not None:
            raise self.start_problem
        if self.start_result is None:
            raise RuntimeError("fake start result was not configured")
        return self.start_result

    async def inspect_running_access(self, owner, job_id):
        self.inspect_calls.append((owner, job_id))
        return self.inspect_access

    async def quiesce_segment(self, owner, payload):
        self.quiesce_calls.append((owner, payload))
        if self.quiesce_problem is not None:
            raise self.quiesce_problem
        if self.quiesce_result is None:
            raise RuntimeError("fake quiesce result was not configured")
        return self.quiesce_result

    async def release_segment(self, owner, payload):
        self.release_calls.append((owner, payload))
        if self.release_problem is not None:
            raise self.release_problem
        if self.release_result is None:
            raise RuntimeError("fake release result was not configured")
        return self.release_result

    async def finalize_job(self, owner, payload):
        self.finalize_calls.append((owner, payload))
        if self.finalize_problem is not None:
            raise self.finalize_problem
        if self.finalize_result is None:
            raise RuntimeError("fake finalize result was not configured")
        return self.finalize_result


def _fixture(*, audit=None, coordinator=None):
    private_key = Ed25519PrivateKey.generate()
    identity_ids = SequenceIds(start=900)
    issuer = WorkloadIdentityIssuer(
        private_key=private_key,
        key_id="issuer-v1",
        issuer="openloop-control",
        audience="openloop:broker-control",
        clock=lambda: NOW,
        id_factory=identity_ids,
    )
    verifier = WorkloadIdentityVerifier(
        public_keys={"issuer-v1": private_key.public_key()},
        issuer="openloop-control",
        audience="openloop:broker-control",
        clock=lambda: NOW,
    )
    clock = MutableClock(NOW)
    repository = InMemoryBrokerRepository(clock=clock)
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=1000))
    capability = JobCapabilityAuthority(
        CapabilityRootRing({"cap-v1": bytes(range(32))}, current_version="cap-v1")
    )
    audit = audit or InMemoryRpcAuditSink(clock=lambda: NOW)
    coordinator = coordinator or FakeSegmentCoordinator()
    app = BrokerRpcApplication(
        ledger=ledger,
        identity_verifier=verifier,
        capability_authority=capability,
        audit_sink=audit,
        policy=BrokerRpcPolicy("default", "docker", "local", 300),
        segment_coordinator=coordinator,
    )
    return app, issuer, repository, audit


def _token(
    issuer,
    *,
    owner=OWNER,
    isolation=IsolationMode.DEDICATED,
    required=IsolationMode.SHARED,
    intents=frozenset({WorkloadIntent.CREATE_JOB, WorkloadIntent.INSPECT_JOB}),
):
    return issuer.issue(
        owner=owner,
        worker_instance_id=WORKER_ID,
        assignment_id=ASSIGNMENT_ID,
        isolation_mode=isolation,
        required_isolation=required,
        intents=intents,
    )


def _create_request(issuer, *, request_number=1, owner=OWNER, required=IsolationMode.SHARED):
    return RpcRequest(
        RPC_VERSION,
        UUID(f"00000000-0000-4000-8000-{request_number:012d}"),
        WorkloadIntent.CREATE_JOB,
        _token(issuer, owner=owner, required=required),
        None,
        CreateJobPayload("rpc-create-key-01"),
    )


def _access(created):
    return RunningGenerationAccess(
        job_id=created.ticket.job_id,
        conversation_id=created.ticket.conversation_id,
        generation=1,
        deadline=NOW + timedelta(seconds=300),
        socket_path=Path("/tmp/openloop-test/agent.sock"),
        relay_capability="R" * 43,
        session_api_key="S" * 43,
    )


def _start_request(issuer, created, *, request_number=50, capability=None):
    return RpcRequest(
        RPC_VERSION,
        UUID(f"00000000-0000-4000-8000-{request_number:012d}"),
        WorkloadIntent.START_SEGMENT,
        _token(issuer, intents=frozenset({WorkloadIntent.START_SEGMENT})),
        capability or created.capability,
        StartSegmentPayload(
            created.ticket.job_id,
            0,
            "rpc-start-segment-01",
        ),
    )


async def test_create_replay_returns_same_capability_and_audits_each_rpc():
    app, issuer, repository, audit = _fixture()
    first = await app.handle(_create_request(issuer, request_number=1), PEER)
    replay = await app.handle(_create_request(issuer, request_number=2), PEER)
    assert isinstance(first.result, CreateJobResult)
    assert isinstance(replay.result, CreateJobResult)
    assert replay.result.ticket.replayed is True
    assert replay.result.ticket.job_id == first.result.ticket.job_id
    assert replay.result.capability == first.result.capability
    assert len(await repository.audit_records_for_test()) == 1
    records = await audit.records_for_test()
    assert len(records) == 2
    rendered = repr(records)
    assert first.result.capability.value not in rendered
    assert first.result.ticket.job_id == records[0].job_id


async def test_inspect_requires_owner_capability_and_isolation_floor():
    app, issuer, _, audit = _fixture()
    created = await app.handle(
        _create_request(
            issuer,
            request_number=11,
            required=IsolationMode.DEDICATED,
        ),
        PEER,
    )
    assert isinstance(created.result, CreateJobResult)
    job_id = created.result.ticket.job_id
    capability = created.result.capability

    allowed = RpcRequest(
        RPC_VERSION,
        UUID("00000000-0000-4000-8000-000000000012"),
        WorkloadIntent.INSPECT_JOB,
        _token(
            issuer,
            isolation=IsolationMode.DEDICATED,
            required=IsolationMode.DEDICATED,
        ),
        capability,
        InspectJobPayload(job_id),
    )
    response = await app.handle(allowed, PEER)
    assert isinstance(response.result, InspectJobResult)
    assert response.result.snapshot.job_id == job_id

    denied_requests = [
        RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000000013"),
            WorkloadIntent.INSPECT_JOB,
            _token(issuer, owner=OTHER_OWNER),
            capability,
            InspectJobPayload(job_id),
        ),
        RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000000014"),
            WorkloadIntent.INSPECT_JOB,
            _token(issuer),
            JobCapability("A" * 43),
            InspectJobPayload(job_id),
        ),
        RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000000015"),
            WorkloadIntent.INSPECT_JOB,
            _token(issuer, isolation=IsolationMode.SHARED),
            capability,
            InspectJobPayload(job_id),
        ),
    ]
    for denied in denied_requests:
        response = await app.handle(denied, PEER)
        assert response.failure.code is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    assert len(await audit.records_for_test()) == 5


async def test_missing_intent_is_method_not_allowed_and_durably_audited():
    app, issuer, _, audit = _fixture()
    request = RpcRequest(
        RPC_VERSION,
        UUID("00000000-0000-4000-8000-000000000021"),
        WorkloadIntent.CREATE_JOB,
        _token(issuer, intents=frozenset({WorkloadIntent.INSPECT_JOB})),
        None,
        CreateJobPayload("rpc-create-key-02"),
    )
    response = await app.handle(request, PEER)
    assert response.failure.code is RpcErrorCode.METHOD_NOT_ALLOWED
    assert len(await audit.records_for_test()) == 1


async def test_post_commit_audit_failure_returns_no_success_and_retry_converges():
    failing = FailOnceAuditSink()
    app, issuer, repository, _ = _fixture(audit=failing)
    first = await app.handle(_create_request(issuer, request_number=31), PEER)
    assert first.failure.code is RpcErrorCode.INTERNAL
    retry = await app.handle(_create_request(issuer, request_number=32), PEER)
    assert isinstance(retry.result, CreateJobResult)
    assert retry.result.ticket.replayed is True
    assert len(await repository.audit_records_for_test()) == 1
    assert len(await failing.records_for_test()) == 1


async def test_start_authorizes_before_coordinator_and_audits_operation():
    coordinator = FakeSegmentCoordinator()
    app, issuer, _, audit = _fixture(coordinator=coordinator)
    created_response = await app.handle(
        _create_request(issuer, request_number=41), PEER
    )
    created = created_response.result
    assert isinstance(created, CreateJobResult)
    access = _access(created)
    operation_id = UUID("00000000-0000-4000-8000-000000000451")
    coordinator.start_result = StartSegmentResult(
        operation_id, False, access
    )

    response = await app.handle(_start_request(issuer, created), PEER)

    assert response.result == coordinator.start_result
    assert coordinator.start_calls == [
        (
            OWNER,
            StartSegmentPayload(
                created.ticket.job_id,
                0,
                "rpc-start-segment-01",
            ),
        )
    ]
    records = await audit.records_for_test()
    assert records[-1].decision.value == "allowed"
    assert records[-1].job_id == created.ticket.job_id
    assert records[-1].operation_id == operation_id


async def test_start_rejects_bad_capability_before_coordinator():
    coordinator = FakeSegmentCoordinator()
    app, issuer, _, _ = _fixture(coordinator=coordinator)
    created_response = await app.handle(
        _create_request(issuer, request_number=61), PEER
    )
    created = created_response.result
    assert isinstance(created, CreateJobResult)

    response = await app.handle(
        _start_request(
            issuer,
            created,
            request_number=62,
            capability=JobCapability("A" * 43),
        ),
        PEER,
    )

    assert response.failure.code is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    assert coordinator.start_calls == []


@pytest.mark.parametrize(
    ("coordinator_code", "rpc_code", "audit_reason"),
    [
        (
            SegmentCoordinatorCode.IDEMPOTENCY_CONFLICT,
            RpcErrorCode.IDEMPOTENCY_CONFLICT,
            "idempotency_conflict",
        ),
        (
            SegmentCoordinatorCode.STATE_CONFLICT,
            RpcErrorCode.STATE_CONFLICT,
            "state_conflict",
        ),
        (
            SegmentCoordinatorCode.INVALID_RECEIPT,
            RpcErrorCode.INVALID_RECEIPT,
            "invalid_receipt",
        ),
        (
            SegmentCoordinatorCode.RUNTIME_UNAVAILABLE,
            RpcErrorCode.RUNTIME_UNAVAILABLE,
            "runtime_unavailable",
        ),
        (
            SegmentCoordinatorCode.DEADLINE_EXCEEDED,
            RpcErrorCode.DEADLINE_EXCEEDED,
            "deadline_exceeded",
        ),
        (
            SegmentCoordinatorCode.INTERNAL,
            RpcErrorCode.INTERNAL,
            "internal",
        ),
    ],
)
async def test_start_maps_bounded_coordinator_failures(
    coordinator_code,
    rpc_code,
    audit_reason,
):
    coordinator = FakeSegmentCoordinator()
    app, issuer, _, audit = _fixture(coordinator=coordinator)
    created_response = await app.handle(
        _create_request(issuer, request_number=71), PEER
    )
    created = created_response.result
    assert isinstance(created, CreateJobResult)
    operation_id = UUID("00000000-0000-4000-8000-000000000471")
    coordinator.start_problem = SegmentCoordinatorProblem(
        coordinator_code,
        operation_id=operation_id,
    )

    response = await app.handle(
        _start_request(issuer, created, request_number=72), PEER
    )

    assert response.failure.code is rpc_code
    record = (await audit.records_for_test())[-1]
    assert record.reason.value == audit_reason
    assert record.job_id == created.ticket.job_id
    assert record.operation_id == operation_id


async def test_inspect_includes_coordinator_access_after_authorization():
    coordinator = FakeSegmentCoordinator()
    app, issuer, _, _ = _fixture(coordinator=coordinator)
    created_response = await app.handle(
        _create_request(issuer, request_number=81), PEER
    )
    created = created_response.result
    assert isinstance(created, CreateJobResult)
    coordinator.inspect_access = _access(created)
    request = RpcRequest(
        RPC_VERSION,
        UUID("00000000-0000-4000-8000-000000000082"),
        WorkloadIntent.INSPECT_JOB,
        _token(issuer, intents=frozenset({WorkloadIntent.INSPECT_JOB})),
        created.capability,
        InspectJobPayload(created.ticket.job_id),
    )

    response = await app.handle(request, PEER)

    assert isinstance(response.result, InspectJobResult)
    assert response.result.access == coordinator.inspect_access
    assert coordinator.inspect_calls == [(OWNER, created.ticket.job_id)]


async def test_start_audit_failure_returns_no_access_and_retry_can_replay():
    audit = FailOnceAuditSink()
    coordinator = FakeSegmentCoordinator()
    app, issuer, _, _ = _fixture(audit=audit, coordinator=coordinator)
    committed_create = await app.handle(
        _create_request(issuer, request_number=91), PEER
    )
    assert committed_create.failure.code is RpcErrorCode.INTERNAL
    created_response = await app.handle(
        _create_request(issuer, request_number=92), PEER
    )
    created = created_response.result
    assert isinstance(created, CreateJobResult)
    operation_id = UUID("00000000-0000-4000-8000-000000000491")
    coordinator.start_result = StartSegmentResult(
        operation_id,
        False,
        _access(created),
    )
    audit.failed = False

    hidden = await app.handle(
        _start_request(issuer, created, request_number=93), PEER
    )
    assert hidden.result is None
    assert hidden.failure.code is RpcErrorCode.INTERNAL

    coordinator.start_result = StartSegmentResult(
        operation_id,
        True,
        _access(created),
    )
    replay = await app.handle(
        _start_request(issuer, created, request_number=94), PEER
    )
    assert isinstance(replay.result, StartSegmentResult)
    assert replay.result.replayed is True


async def test_lifecycle_methods_authorize_dispatch_and_audit_operations():
    coordinator = FakeSegmentCoordinator()
    app, issuer, _, audit = _fixture(coordinator=coordinator)
    created_response = await app.handle(
        _create_request(issuer, request_number=101), PEER
    )
    created = created_response.result
    assert isinstance(created, CreateJobResult)
    running = _access(created)
    checkpoint = CheckpointGenerationAccess(
        job_id=running.job_id,
        conversation_id=running.conversation_id,
        generation=running.generation,
        deadline=running.deadline,
        socket_path=running.socket_path,
        relay_capability=running.relay_capability,
        session_api_key=running.session_api_key,
    )
    coordinator.quiesce_result = QuiesceSegmentResult(
        UUID("00000000-0000-4000-8000-000000000501"), False, checkpoint
    )
    coordinator.release_result = ReleaseSegmentResult(
        UUID("00000000-0000-4000-8000-000000000502"),
        False,
        JobState.PARKED,
        GenerationState.RELEASED,
    )
    coordinator.finalize_result = FinalizeJobResult(
        UUID("00000000-0000-4000-8000-000000000503"),
        False,
        JobState.TERMINAL,
    )
    receipt = SignedCheckpointReceipt("header.payload.signature")
    requests = (
        RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000000102"),
            WorkloadIntent.QUIESCE_SEGMENT,
            _token(
                issuer, intents=frozenset({WorkloadIntent.QUIESCE_SEGMENT})
            ),
            created.capability,
            QuiesceSegmentPayload(
                created.ticket.job_id,
                1,
                "rpc-quiesce-lifecycle",
                "barrier-lifecycle",
            ),
        ),
        RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000000103"),
            WorkloadIntent.RELEASE_SEGMENT,
            _token(
                issuer, intents=frozenset({WorkloadIntent.RELEASE_SEGMENT})
            ),
            created.capability,
            ReleaseSegmentPayload(
                created.ticket.job_id,
                1,
                "rpc-release-lifecycle",
                receipt,
                ReleaseTarget.PARKED,
            ),
        ),
        RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000000104"),
            WorkloadIntent.FINALIZE_JOB,
            _token(issuer, intents=frozenset({WorkloadIntent.FINALIZE_JOB})),
            created.capability,
            FinalizeJobPayload(
                created.ticket.job_id,
                1,
                "rpc-finalize-lifecycle",
                TerminalOutcome.SUCCESS,
            ),
        ),
    )

    responses = [await app.handle(request, PEER) for request in requests]

    assert isinstance(responses[0].result, QuiesceSegmentResult)
    assert isinstance(responses[1].result, ReleaseSegmentResult)
    assert isinstance(responses[2].result, FinalizeJobResult)
    assert coordinator.quiesce_calls == [(OWNER, requests[0].payload)]
    assert coordinator.release_calls == [(OWNER, requests[1].payload)]
    assert coordinator.finalize_calls == [(OWNER, requests[2].payload)]
    records = await audit.records_for_test()
    assert [record.operation_id for record in records[-3:]] == [
        coordinator.quiesce_result.operation_id,
        coordinator.release_result.operation_id,
        coordinator.finalize_result.operation_id,
    ]


async def test_release_invalid_receipt_is_bounded_and_secret_free():
    coordinator = FakeSegmentCoordinator()
    app, issuer, _, audit = _fixture(coordinator=coordinator)
    created_response = await app.handle(
        _create_request(issuer, request_number=111), PEER
    )
    created = created_response.result
    assert isinstance(created, CreateJobResult)
    operation_id = UUID("00000000-0000-4000-8000-000000000511")
    coordinator.release_problem = SegmentCoordinatorProblem(
        SegmentCoordinatorCode.INVALID_RECEIPT,
        operation_id=operation_id,
    )
    receipt = SignedCheckpointReceipt("header.payload.signature")
    request = RpcRequest(
        RPC_VERSION,
        UUID("00000000-0000-4000-8000-000000000112"),
        WorkloadIntent.RELEASE_SEGMENT,
        _token(issuer, intents=frozenset({WorkloadIntent.RELEASE_SEGMENT})),
        created.capability,
        ReleaseSegmentPayload(
            created.ticket.job_id,
            1,
            "rpc-release-invalid",
            receipt,
            ReleaseTarget.PARKED,
        ),
    )

    response = await app.handle(request, PEER)

    assert response.failure.code is RpcErrorCode.INVALID_RECEIPT
    record = (await audit.records_for_test())[-1]
    assert record.reason.value == "invalid_receipt"
    assert record.operation_id == operation_id
    assert receipt.value not in repr(record)
