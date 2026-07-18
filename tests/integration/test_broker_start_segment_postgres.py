"""Restart and key-rotation proofs for composed starts on PostgreSQL."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
from uuid import uuid4

import pytest

from openloop.broker.ledger import BrokerLedger
from openloop.broker.models import BrokerOwner, GenerationState
from openloop.broker.postgres import PostgresBrokerRepository
from openloop.broker_control.coordinator import BrokerSegmentCoordinator
from openloop.broker_control.durable import LocalDurableStateAdapter
from openloop.broker_control.secrets import (
    RuntimeSecretAuthority,
    RuntimeSecretRootRing,
)
from openloop.broker_rpc.coordinator import (
    BrokerRpcPolicy,
    SegmentCoordinatorCode,
    SegmentCoordinatorProblem,
)
from openloop.broker_rpc.models import StartSegmentPayload
from openloop.broker_runtime.memory import InMemoryRuntimeDriver
from tests.support.broker_repository_contract import SequenceIds


DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop",
)
OWNER = BrokerOwner("tenant-start-postgres", "workload-start-postgres")
POLICY = BrokerRpcPolicy("default", "docker", "local", 300)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


async def _reachable() -> bool:
    try:
        import asyncpg

        connection = await asyncpg.connect(DSN, timeout=3)
        await connection.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def postgres_start(tmp_path: Path):
    if not await _reachable():
        pytest.skip(f"no PostgreSQL reachable at {DSN}")
    import asyncpg

    schema = f"broker_start_test_{uuid4().hex}"
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
    await repository.setup(pool)
    state_root = tmp_path / "state"
    state_root.mkdir(mode=0o700)
    os.chown(state_root, os.getuid(), os.getgid())
    state_root.chmod(0o700)
    try:
        yield repository, pool, state_root
    finally:
        await repository.close()
        await pool.close()
        admin = await asyncpg.connect(DSN)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


def _authority(*, current: str) -> RuntimeSecretAuthority:
    return RuntimeSecretAuthority(
        RuntimeSecretRootRing(
            {
                "runtime-v1": bytes(range(32)),
                "runtime-v2": bytes(range(32, 64)),
            },
            current_version=current,
        )
    )


def _coordinator(
    ledger: BrokerLedger,
    state_root: Path,
    authority: RuntimeSecretAuthority,
    runtime: InMemoryRuntimeDriver,
) -> BrokerSegmentCoordinator:
    return BrokerSegmentCoordinator(
        ledger=ledger,
        policy=POLICY,
        runtime_driver=runtime,
        secret_authority=authority,
        durable_state_adapter=LocalDurableStateAdapter(
            state_root=state_root,
            uid=os.getuid(),
            gid=os.getgid(),
        ),
        clock=lambda: datetime.now(UTC),
    )


async def _pin_starting_generation(
    ledger: BrokerLedger,
    authority: RuntimeSecretAuthority,
    job_id,
    conversation_id,
    state_root: Path,
    *,
    key: str,
):
    durable = LocalDurableStateAdapter(
        state_root=state_root,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    durable_ref = durable.reference(job_id)
    durable_digest = authority.durable_digest_for(
        OWNER,
        job_id,
        conversation_id,
        durable_ref,
        "runtime-v1",
    )
    return await ledger.begin_start(
        OWNER,
        key,
        job_id,
        0,
        POLICY.execution_lease_seconds,
        "runtime-v1",
        durable_ref,
        "runtime-v1",
        durable_digest,
    )


async def test_restart_replays_persisted_old_versions_without_secrets(
    postgres_start,
):
    repository, pool, state_root = postgres_start
    first_ledger = BrokerLedger(repository, id_factory=SequenceIds(start=30_000))
    authority_v1 = _authority(current="runtime-v1")
    created = await first_ledger.create_job(
        OWNER,
        "postgres-start-create-01",
        POLICY.profile,
        POLICY.runtime_driver,
        POLICY.durable_state_driver,
    )
    await _pin_starting_generation(
        first_ledger,
        authority_v1,
        created.job_id,
        created.conversation_id,
        state_root,
        key="postgres-start-key-01",
    )

    restarted_repository = PostgresBrokerRepository()
    await restarted_repository.setup(pool)
    restarted_ledger = BrokerLedger(
        restarted_repository, id_factory=SequenceIds(start=40_000)
    )
    runtime = InMemoryRuntimeDriver(
        clock=lambda: datetime.now(UTC), maximum_lifetime_seconds=600
    )
    authority_v2 = _authority(current="runtime-v2")
    coordinator = _coordinator(
        restarted_ledger, state_root, authority_v2, runtime
    )

    result = await coordinator.start_segment(
        OWNER,
        StartSegmentPayload(
            created.job_id,
            0,
            "postgres-start-key-01",
        ),
    )
    replay = await coordinator.start_segment(
        OWNER,
        StartSegmentPayload(
            created.job_id,
            0,
            "postgres-start-key-01",
        ),
    )
    recovery = await restarted_ledger.inspect_job_for_recovery(
        OWNER, created.job_id
    )
    generation = recovery.generation_record
    assert result.access == replay.access
    assert replay.replayed is True
    assert generation.runtime_key_version == "runtime-v1"
    assert generation.durable_key_version == "runtime-v1"

    derived = authority_v1.derive(
        OWNER,
        created.job_id,
        created.conversation_id,
        1,
        generation.durable_state_ref,
        runtime_key_version="runtime-v1",
        durable_key_version="runtime-v1",
    )
    async with pool.acquire() as connection:
        encoded = await connection.fetchval(
            """
            SELECT json_build_object(
                'job', row_to_json(j), 'generation', row_to_json(g)
            )::text
            FROM broker_jobs j
            JOIN broker_generations g USING (job_id)
            WHERE j.job_id = $1 AND g.generation = 1
            """,
            created.job_id,
        )
    for secret in (
        derived.relay_capability,
        derived.session_api_key,
        derived.conversation_secret,
        str(state_root),
    ):
        assert secret not in encoded
    await restarted_repository.close()


async def test_abandon_rotation_and_concurrent_replay_preserve_pins(
    postgres_start,
):
    repository, _, state_root = postgres_start
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=50_000))
    authority_v1 = _authority(current="runtime-v1")
    created = await ledger.create_job(
        OWNER,
        "postgres-rotate-create-01",
        POLICY.profile,
        POLICY.runtime_driver,
        POLICY.durable_state_driver,
    )
    await _pin_starting_generation(
        ledger,
        authority_v1,
        created.job_id,
        created.conversation_id,
        state_root,
        key="postgres-rotate-start-01",
    )
    await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.STARTING,
        SegmentCoordinatorCode.RUNTIME_UNAVAILABLE.value,
    )

    runtime = InMemoryRuntimeDriver(
        clock=lambda: datetime.now(UTC), maximum_lifetime_seconds=600
    )
    coordinator = _coordinator(
        ledger, state_root, _authority(current="runtime-v2"), runtime
    )
    with pytest.raises(SegmentCoordinatorProblem) as original:
        await coordinator.start_segment(
            OWNER,
            StartSegmentPayload(
                created.job_id,
                0,
                "postgres-rotate-start-01",
            ),
        )
    assert original.value.code is SegmentCoordinatorCode.RUNTIME_UNAVAILABLE

    payload = StartSegmentPayload(
        created.job_id,
        1,
        "postgres-rotate-start-02",
    )
    first, second = await asyncio.gather(
        coordinator.start_segment(OWNER, payload),
        coordinator.start_segment(OWNER, payload),
    )
    assert {first.replayed, second.replayed} == {False, True}
    assert first.operation_id == second.operation_id
    recovery = await ledger.inspect_job_for_recovery(OWNER, created.job_id)
    generation = recovery.generation_record
    assert generation.generation == 2
    assert generation.runtime_key_version == "runtime-v2"
    assert generation.durable_key_version == "runtime-v1"
