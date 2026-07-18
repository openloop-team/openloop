from datetime import UTC, datetime
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openloop.broker.ledger import BrokerLedger
from openloop.broker.memory import InMemoryBrokerRepository
from openloop.broker.models import BrokerOwner, IsolationMode
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
from openloop.broker_rpc.errors import RpcErrorCode
from openloop.broker_rpc.identity import (
    WorkloadIdentityIssuer,
    WorkloadIdentityVerifier,
    WorkloadIntent,
)
from openloop.broker_rpc.models import (
    CreateJobPayload,
    CreateJobResult,
    InspectJobPayload,
    InspectJobResult,
    RpcRequest,
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


def _fixture(*, audit=None):
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
    app = BrokerRpcApplication(
        ledger=ledger,
        identity_verifier=verifier,
        capability_authority=capability,
        audit_sink=audit,
        policy=BrokerRpcPolicy("default", "docker", "postgres"),
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
        1,
        UUID(f"00000000-0000-4000-8000-{request_number:012d}"),
        WorkloadIntent.CREATE_JOB,
        _token(issuer, owner=owner, required=required),
        None,
        CreateJobPayload("rpc-create-key-01"),
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
        1,
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
            1,
            UUID("00000000-0000-4000-8000-000000000013"),
            WorkloadIntent.INSPECT_JOB,
            _token(issuer, owner=OTHER_OWNER),
            capability,
            InspectJobPayload(job_id),
        ),
        RpcRequest(
            1,
            UUID("00000000-0000-4000-8000-000000000014"),
            WorkloadIntent.INSPECT_JOB,
            _token(issuer),
            JobCapability("A" * 43),
            InspectJobPayload(job_id),
        ),
        RpcRequest(
            1,
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
        1,
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

