"""Authenticated broker RPC persistence and race proofs on real PostgreSQL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from openloop.broker.ledger import BrokerLedger
from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker.postgres import PostgresBrokerRepository
from openloop.broker_rpc.application import BrokerRpcApplication, BrokerRpcPolicy
from openloop.broker_rpc.audit import PeerCredentials, PostgresRpcAuditSink
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
    RPC_VERSION,
    RpcRequest,
)
from tests.support.broker_repository_contract import SequenceIds


DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop",
)
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
OWNER = BrokerOwner("tenant-a", "workload-a")
OTHER_OWNER = BrokerOwner("tenant-b", "workload-b")
PEER = PeerCredentials(4401, 1000, 1000)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


async def _reachable() -> bool:
    try:
        import asyncpg

        connection = await asyncpg.connect(DSN, timeout=3)
        await connection.close()
        return True
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class RpcPostgresFixture:
    app: BrokerRpcApplication
    issuer: WorkloadIdentityIssuer
    ledger: BrokerLedger
    capability: JobCapabilityAuthority
    audit: PostgresRpcAuditSink
    pool: object

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
    if not await _reachable():
        pytest.skip(f"no PostgreSQL reachable at {DSN}")
    import asyncpg

    schema = f"broker_rpc_test_{uuid4().hex}"
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    await admin.close()
    pool = await asyncpg.create_pool(
        DSN,
        min_size=2,
        max_size=10,
        server_settings={"search_path": schema},
    )
    repository = PostgresBrokerRepository()
    audit = PostgresRpcAuditSink()
    try:
        await repository.setup(pool)
        await audit.setup(pool)
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
            CapabilityRootRing(
                {"cap-v1": bytes(range(32))}, current_version="cap-v1"
            )
        )
        app = BrokerRpcApplication(
            ledger=ledger,
            identity_verifier=verifier,
            capability_authority=capability,
            audit_sink=audit,
            policy=BrokerRpcPolicy("default", "docker", "postgres"),
        )
        yield RpcPostgresFixture(
            app, issuer, ledger, capability, audit, pool
        )
    finally:
        await audit.close()
        await repository.close()
        await pool.close()
        admin = await asyncpg.connect(DSN)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


async def test_concurrent_exact_create_persists_one_capability_digest_and_two_audits(
    rpc_postgres,
):
    fixture = rpc_postgres
    first, second = await asyncio.gather(
        fixture.app.handle(
            fixture.create_request("rpc-postgres-create-01"), PEER
        ),
        fixture.app.handle(
            fixture.create_request("rpc-postgres-create-01"), PEER
        ),
    )
    assert isinstance(first.result, CreateJobResult)
    assert isinstance(second.result, CreateJobResult)
    assert {first.result.ticket.replayed, second.result.ticket.replayed} == {
        False,
        True,
    }
    assert first.result.ticket.job_id == second.result.ticket.job_id
    assert first.result.capability == second.result.capability

    async with fixture.pool.acquire() as connection:
        job = await connection.fetchrow(
            """
            SELECT minimum_isolation, control_key_version, control_epoch,
                   control_capability_digest
            FROM broker_jobs WHERE job_id = $1
            """,
            first.result.ticket.job_id,
        )
        encoded_job = await connection.fetchval(
            "SELECT row_to_json(j)::text FROM broker_jobs j WHERE job_id = $1",
            first.result.ticket.job_id,
        )
        assert await connection.fetchval(
            "SELECT count(*) FROM broker_audit"
        ) == 1
        assert await connection.fetchval(
            "SELECT count(*) FROM broker_rpc_audit"
        ) == 2
    assert dict(job) == {
        "minimum_isolation": "shared",
        "control_key_version": "cap-v1",
        "control_epoch": 1,
        "control_capability_digest": fixture.capability.digest_for(
            OWNER, first.result.ticket.job_id, "cap-v1", 1
        ),
    }
    assert first.result.capability.value not in encoded_job


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
        fixture.capability.derive(
            OWNER, job_id, authorization.authorization
        )
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

    async with fixture.pool.acquire() as connection:
        decisions = await connection.fetch(
            """
            SELECT decision, reason_code FROM broker_rpc_audit
            ORDER BY sequence
            """
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

    import asyncpg

    async with fixture.pool.acquire() as connection:
        with pytest.raises(asyncpg.CheckViolationError):
            await connection.execute(
                "UPDATE broker_jobs SET minimum_isolation = 'shared' WHERE job_id = $1",
                legacy.job_id,
            )
