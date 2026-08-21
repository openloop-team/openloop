"""Authenticated broker RPC persistence and race proofs on real PostgreSQL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from openloop.broker.ledger import BrokerLedger
from openloop.broker.models import (
    BrokerOwner,
    IsolationMode,
    ReleaseTarget,
    SignedCheckpointReceipt,
    TerminalOutcome,
)
from openloop.broker.postgres import PostgresBrokerRepository
from openloop.broker_rpc.application import BrokerRpcApplication, BrokerRpcPolicy
from openloop.broker_rpc.audit import PeerCredentials, PostgresRpcAuditSink
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
    CreateJobPayload,
    CreateJobResult,
    FinalizeJobPayload,
    InspectJobPayload,
    QuiesceSegmentPayload,
    ReleaseSegmentPayload,
    RpcRequest,
    StartSegmentPayload,
)
from openloop.db import create_engine
from tests.support.broker_repository_contract import SequenceIds
from tests.support.postgres import postgres_dsn, require_postgres

DSN = postgres_dsn()


async def _admin(statement: str) -> None:
    """Run one statement outside the per-test schema, to create or drop it."""
    engine = await create_engine(DSN, min_size=1, max_size=1)
    try:
        async with engine.begin() as connection:
            # sql-text: DDL naming a schema this suite creates for itself.
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


class DisabledSegmentCoordinator:
    async def start_segment(self, owner, payload):
        raise SegmentCoordinatorProblem(SegmentCoordinatorCode.INTERNAL)

    async def inspect_running_access(self, owner, job_id):
        return None

    async def quiesce_segment(self, owner, payload):
        raise SegmentCoordinatorProblem(SegmentCoordinatorCode.INTERNAL)

    async def release_segment(self, owner, payload):
        raise SegmentCoordinatorProblem(SegmentCoordinatorCode.INTERNAL)

    async def finalize_job(self, owner, payload):
        raise SegmentCoordinatorProblem(SegmentCoordinatorCode.INTERNAL)


OWNER = BrokerOwner("tenant-a", "workload-a")
OTHER_OWNER = BrokerOwner("tenant-b", "workload-b")
PEER = PeerCredentials(4401, 1000, 1000)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@dataclass(frozen=True, slots=True)
class RpcPostgresFixture:
    app: BrokerRpcApplication
    issuer: WorkloadIdentityIssuer
    ledger: BrokerLedger
    capability: JobCapabilityAuthority
    audit: PostgresRpcAuditSink
    engine: object

    def token(
        self,
        *,
        owner: BrokerOwner = OWNER,
        intent: WorkloadIntent,
        isolation: IsolationMode = IsolationMode.DEDICATED,
        required: IsolationMode = IsolationMode.SHARED,
    ):
        return self.issuer.issue(
            owner=owner,
            worker_instance_id=uuid4(),
            assignment_id=uuid4(),
            isolation_mode=isolation,
            required_isolation=required,
            intents={intent},
        )

    def create_request(self, key: str) -> RpcRequest:
        return RpcRequest(
            RPC_VERSION,
            uuid4(),
            WorkloadIntent.CREATE_JOB,
            self.token(intent=WorkloadIntent.CREATE_JOB),
            None,
            CreateJobPayload(key),
        )


@pytest.fixture
async def rpc_postgres():
    await require_postgres(DSN)

    schema = f"broker_rpc_test_{uuid4().hex}"
    await _admin(f'CREATE SCHEMA "{schema}"')
    engine = await create_engine(
        DSN,
        min_size=2,
        max_size=10,
        server_settings={"search_path": schema},
    )
    repository = PostgresBrokerRepository()
    audit = PostgresRpcAuditSink()
    try:
        await repository.setup(engine)
        await audit.setup(engine)
        private_key = Ed25519PrivateKey.generate()
        issuer = WorkloadIdentityIssuer(
            private_key=private_key,
            key_id="issuer-v1",
            issuer="openloop-control",
            audience="openloop:broker-control",
            clock=lambda: NOW,
        )
        verifier = WorkloadIdentityVerifier(
            public_keys={"issuer-v1": private_key.public_key()},
            issuer="openloop-control",
            audience="openloop:broker-control",
            clock=lambda: NOW,
        )
        ledger = BrokerLedger(repository, id_factory=SequenceIds(start=20_000))
        capability = JobCapabilityAuthority(
            CapabilityRootRing({"cap-v1": bytes(range(32))}, current_version="cap-v1")
        )
        app = BrokerRpcApplication(
            ledger=ledger,
            identity_verifier=verifier,
            capability_authority=capability,
            audit_sink=audit,
            policy=BrokerRpcPolicy("default", "docker", "postgres", 300),
            segment_coordinator=DisabledSegmentCoordinator(),
        )
        yield RpcPostgresFixture(app, issuer, ledger, capability, audit, engine)
    finally:
        await audit.close()
        await repository.close()
        await engine.dispose()
        await _admin(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


async def test_concurrent_exact_create_persists_one_capability_digest_and_two_audits(
    rpc_postgres,
):
    fixture = rpc_postgres
    first, second = await asyncio.gather(
        fixture.app.handle(fixture.create_request("rpc-postgres-create-01"), PEER),
        fixture.app.handle(fixture.create_request("rpc-postgres-create-01"), PEER),
    )
    assert isinstance(first.result, CreateJobResult)
    assert isinstance(second.result, CreateJobResult)
    assert {first.result.ticket.replayed, second.result.ticket.replayed} == {
        False,
        True,
    }
    assert first.result.ticket.job_id == second.result.ticket.job_id
    assert first.result.capability == second.result.capability

    async with fixture.engine.connect() as connection:
        # sql-text: reads the authorization columns back, then the whole row as
        # JSON to prove no secret landed in any of them.
        job = (
            (
                await connection.execute(
                    text(
                        """
                        SELECT minimum_isolation, control_key_version,
                               control_epoch, control_capability_digest
                        FROM broker_jobs WHERE job_id = :job_id
                        """
                    ),
                    {"job_id": first.result.ticket.job_id},
                )
            )
            .mappings()
            .first()
        )
        encoded_job = await connection.scalar(
            text(
                "SELECT row_to_json(j)::text FROM broker_jobs j WHERE job_id = :job_id"
            ),
            {"job_id": first.result.ticket.job_id},
        )
        assert await connection.scalar(text("SELECT count(*) FROM broker_audit")) == 1
        assert (
            await connection.scalar(text("SELECT count(*) FROM broker_rpc_audit")) == 2
        )
    assert dict(job) == {
        "minimum_isolation": "shared",
        "control_key_version": "cap-v1",
        "control_epoch": 1,
        "control_capability_digest": fixture.capability.digest_for(
            OWNER, first.result.ticket.job_id, "cap-v1", 1
        ),
    }
    assert first.result.capability.value not in encoded_job


async def test_postgres_audit_accepts_every_reviewed_lifecycle_method(
    rpc_postgres,
):
    fixture = rpc_postgres
    created_response = await fixture.app.handle(
        fixture.create_request("rpc-postgres-lifecycle-create"), PEER
    )
    created = created_response.result
    assert isinstance(created, CreateJobResult)
    job_id = created.ticket.job_id
    receipt = SignedCheckpointReceipt("header.payload.signature")
    requests = (
        (
            WorkloadIntent.START_SEGMENT,
            StartSegmentPayload(job_id, 0, "rpc-postgres-start"),
        ),
        (
            WorkloadIntent.QUIESCE_SEGMENT,
            QuiesceSegmentPayload(
                job_id, 0, "rpc-postgres-quiesce", "barrier-postgres"
            ),
        ),
        (
            WorkloadIntent.RELEASE_SEGMENT,
            ReleaseSegmentPayload(
                job_id,
                0,
                "rpc-postgres-release",
                receipt,
                ReleaseTarget.PARKED,
            ),
        ),
        (
            WorkloadIntent.FINALIZE_JOB,
            FinalizeJobPayload(
                job_id,
                0,
                "rpc-postgres-finalize",
                TerminalOutcome.SUCCESS,
            ),
        ),
    )

    for method, payload in requests:
        response = await fixture.app.handle(
            RpcRequest(
                RPC_VERSION,
                uuid4(),
                method,
                fixture.token(intent=method),
                created.capability,
                payload,
            ),
            PEER,
        )
        assert response.failure.code is RpcErrorCode.INTERNAL

    async with fixture.engine.connect() as connection:
        # sql-text: reads the audit trail the sink appended, in order.
        methods = (
            (
                await connection.execute(
                    text("SELECT method FROM broker_rpc_audit ORDER BY sequence")
                )
            )
            .mappings()
            .all()
        )
    assert [row["method"] for row in methods] == [
        "CREATE_JOB",
        "START_SEGMENT",
        "QUIESCE_SEGMENT",
        "RELEASE_SEGMENT",
        "FINALIZE_JOB",
    ]


async def test_persisted_authorization_survives_restart_and_denials_are_generic(
    rpc_postgres,
):
    fixture = rpc_postgres
    created = await fixture.app.handle(
        fixture.create_request("rpc-postgres-create-02"), PEER
    )
    assert isinstance(created.result, CreateJobResult)
    job_id = created.result.ticket.job_id
    authorization = await fixture.ledger.inspect_job_authorization(OWNER, job_id)
    assert (
        fixture.capability.derive(OWNER, job_id, authorization.authorization)
        == created.result.capability
    )

    denials = (
        RpcRequest(
            RPC_VERSION,
            uuid4(),
            WorkloadIntent.INSPECT_JOB,
            fixture.token(owner=OTHER_OWNER, intent=WorkloadIntent.INSPECT_JOB),
            created.result.capability,
            InspectJobPayload(job_id),
        ),
        RpcRequest(
            RPC_VERSION,
            uuid4(),
            WorkloadIntent.INSPECT_JOB,
            fixture.token(intent=WorkloadIntent.INSPECT_JOB),
            JobCapability("A" * 43),
            InspectJobPayload(job_id),
        ),
    )
    for request in denials:
        response = await fixture.app.handle(request, PEER)
        assert response.failure.code is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    async with fixture.engine.connect() as connection:
        # sql-text: as above — the audit trail, in order.
        decisions = (
            (
                await connection.execute(
                    text(
                        """
                        SELECT decision, reason_code FROM broker_rpc_audit
                        ORDER BY sequence
                        """
                    )
                )
            )
            .mappings()
            .all()
        )
    assert [tuple(row.values()) for row in decisions] == [
        ("allowed", "allowed"),
        ("denied", "not_found_or_unauthorized"),
        ("denied", "not_found_or_unauthorized"),
    ]


async def test_legacy_rows_fail_closed_and_partial_authorization_is_rejected(
    rpc_postgres,
):
    fixture = rpc_postgres
    legacy = await fixture.ledger.create_job(
        OWNER,
        "rpc-postgres-legacy-01",
        "default",
        "docker",
        "postgres",
    )
    request = RpcRequest(
        RPC_VERSION,
        UUID("00000000-0000-4000-8000-000000030001"),
        WorkloadIntent.INSPECT_JOB,
        fixture.token(intent=WorkloadIntent.INSPECT_JOB),
        JobCapability("A" * 43),
        InspectJobPayload(legacy.job_id),
    )
    response = await fixture.app.handle(request, PEER)
    assert response.failure.code is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    async with fixture.engine.begin() as connection:
        # sql-text: sets one column on purpose to prove the table's
        # all-or-none authorization CHECK rejects a partial row.
        with pytest.raises(IntegrityError):
            await connection.execute(
                text(
                    "UPDATE broker_jobs SET minimum_isolation = 'shared' "
                    "WHERE job_id = :job_id"
                ),
                {"job_id": legacy.job_id},
            )
