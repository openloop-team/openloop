"""Broker repository contract and concurrency tests against real PostgreSQL."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from openloop.broker.errors import (
    IdempotencyConflict,
    InvalidTransition,
    MigrationProblem,
    MigrationVersionError,
    ReceiptBindingMismatch,
    StaleGeneration,
)
from openloop.broker.ledger import BrokerLedger
from openloop.broker.models import (
    GenerationState,
    JobState,
    ReleaseTarget,
    TerminalOutcome,
)
from openloop.broker.postgres import (
    Migration,
    PostgresBrokerRepository,
    _load_packaged_migrations,
)
from tests.support.broker_repository_contract import (
    OWNER,
    SequenceIds,
    exercise_complete_lifecycle,
    mark_generation_running,
    quiesce_generation,
    receipt_for,
)


DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop",
)

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
async def postgres_repository():
    if not await _reachable():
        pytest.skip(f"no PostgreSQL reachable at {DSN}")
    import asyncpg

    schema = f"broker_test_{uuid4().hex}"
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    await admin.close()
    pool = await asyncpg.create_pool(
        DSN,
        min_size=1,
        max_size=10,
        server_settings={"search_path": schema},
    )
    repository = PostgresBrokerRepository()
    try:
        await repository.setup(pool)
        yield repository, pool
    finally:
        await repository.close()
        await pool.close()
        admin = await asyncpg.connect(DSN)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


async def _audit_count(pool) -> int:
    async with pool.acquire() as connection:
        return await connection.fetchval("SELECT count(*) FROM broker_audit")


async def test_postgres_create_start_running_and_abandon_contract(
    postgres_repository,
):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds())

    created = await ledger.create_job(
        OWNER, "postgres-create-0001", "default", "docker", "postgres"
    )
    replay = await ledger.create_job(
        OWNER, "postgres-create-0001", "default", "docker", "postgres"
    )
    assert replay.replayed is True
    assert replay.job_id == created.job_id

    start = await ledger.begin_start(
        OWNER, "postgres-start-00001", created.job_id, 0, 30
    )
    running = await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=start.operation_id,
        generation=1,
    )
    assert running.job_state is JobState.ACTIVE
    assert running.generation_state is GenerationState.RUNNING
    snapshot = await ledger.inspect_job(OWNER, created.job_id)
    assert snapshot.state is JobState.ACTIVE
    assert snapshot.current_generation == 1
    recovery = await ledger.inspect_job_for_recovery(OWNER, created.job_id)
    assert recovery.generation_record.runtime_ref == "runtime://generation-1"

    completion_replay = await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=start.operation_id,
        generation=1,
    )
    assert completion_replay.replayed is True

    abandoned = await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.RUNNING,
        "runtime_lost",
        TerminalOutcome.FAILED,
    )
    assert abandoned.generation_state is GenerationState.ABANDONED
    snapshot = await ledger.inspect_job(OWNER, created.job_id)
    assert snapshot.state is JobState.FINALIZING
    assert snapshot.terminal_outcome is TerminalOutcome.FAILED
    assert await _audit_count(pool) == 4


async def test_postgres_start_failure_allocates_next_generation(
    postgres_repository,
):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=100))
    created = await ledger.create_job(
        OWNER, "postgres-create-0002", "default", "docker", "postgres"
    )
    await ledger.begin_start(
        OWNER, "postgres-start-fail1", created.job_id, 0, 30
    )
    abandoned = await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.STARTING,
        "start_failed",
    )
    replay = await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.STARTING,
        "start_failed",
        replay_operation_id=abandoned.operation_id,
    )
    assert replay.replayed is True
    second = await ledger.begin_start(
        OWNER, "postgres-start-next1", created.job_id, 1, 30
    )
    assert second.generation == 2
    assert await _audit_count(pool) == 4


async def test_postgres_complete_lifecycle_shared_contract(postgres_repository):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=200))
    trace = await exercise_complete_lifecycle(ledger)
    assert trace.snapshots[-1].state is JobState.TERMINAL
    assert trace.snapshots[-1].generation == 2
    assert await _audit_count(pool) == 15


async def test_postgres_restart_preserves_inspection_and_exact_replay(
    postgres_repository,
):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=300))
    created = await ledger.create_job(
        OWNER, "postgres-restart-001", "default", "docker", "postgres"
    )
    start = await ledger.begin_start(
        OWNER, "postgres-restart-002", created.job_id, 0, 30
    )
    await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=start.operation_id,
        generation=1,
    )
    await repository.close()

    restarted = PostgresBrokerRepository()
    await restarted.setup(pool)
    try:
        restarted_ledger = BrokerLedger(restarted, id_factory=SequenceIds(start=400))
        snapshot = await restarted_ledger.inspect_job(OWNER, created.job_id)
        assert snapshot.state is JobState.ACTIVE
        replay = await restarted_ledger.begin_start(
            OWNER, "postgres-restart-002", created.job_id, 0, 30
        )
        assert replay.replayed is True
        assert replay.operation_id == start.operation_id
        completion = await mark_generation_running(
            restarted_ledger,
            job_id=created.job_id,
            operation_id=start.operation_id,
            generation=1,
        )
        assert completion.replayed is True
        assert await _audit_count(pool) == 3
    finally:
        await restarted.close()


async def test_postgres_concurrent_same_key_create_is_one_mutation_and_replay(
    postgres_repository,
):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=500))
    first, second = await asyncio.gather(
        ledger.create_job(
            OWNER, "postgres-race-create", "default", "docker", "postgres"
        ),
        ledger.create_job(
            OWNER, "postgres-race-create", "default", "docker", "postgres"
        ),
    )
    assert {first.replayed, second.replayed} == {False, True}
    assert first.operation_id == second.operation_id
    assert first.job_id == second.job_id
    assert await _audit_count(pool) == 1


async def test_postgres_concurrent_conflicting_key_has_one_winner(
    postgres_repository,
):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=600))
    results = await asyncio.gather(
        ledger.create_job(
            OWNER, "postgres-race-conflict", "default", "docker", "postgres"
        ),
        ledger.create_job(
            OWNER, "postgres-race-conflict", "gpu", "docker", "postgres"
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, IdempotencyConflict) for result in results) == 1
    assert await _audit_count(pool) == 1


async def test_postgres_concurrent_starts_preserve_one_live_generation(
    postgres_repository,
):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=700))
    created = await ledger.create_job(
        OWNER, "postgres-race-job001", "default", "docker", "postgres"
    )
    results = await asyncio.gather(
        ledger.begin_start(
            OWNER, "postgres-race-start-a", created.job_id, 0, 30
        ),
        ledger.begin_start(
            OWNER, "postgres-race-start-b", created.job_id, 0, 30
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(
        isinstance(result, (InvalidTransition, StaleGeneration)) for result in results
    ) == 1
    async with pool.acquire() as connection:
        live = await connection.fetchval(
            """
            SELECT count(*) FROM broker_generations
            WHERE job_id = $1
              AND state IN ('starting', 'running', 'quiescing', 'quiesced', 'releasing')
            """,
            created.job_id,
        )
    assert live == 1
    assert await _audit_count(pool) == 2


async def test_postgres_same_key_quiesce_release_and_completion_races_replay(
    postgres_repository,
):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=800))
    created = await ledger.create_job(
        OWNER, "postgres-race-flow01", "default", "docker", "postgres"
    )
    start = await ledger.begin_start(
        OWNER, "postgres-race-flow02", created.job_id, 0, 30
    )
    running_results = await asyncio.gather(
        mark_generation_running(
            ledger,
            job_id=created.job_id,
            operation_id=start.operation_id,
            generation=1,
        ),
        mark_generation_running(
            ledger,
            job_id=created.job_id,
            operation_id=start.operation_id,
            generation=1,
        ),
    )
    assert {item.replayed for item in running_results} == {False, True}

    quiesce_results = await asyncio.gather(
        ledger.begin_quiesce(
            OWNER,
            "postgres-race-quiesce",
            created.job_id,
            1,
            "barrier-race",
        ),
        ledger.begin_quiesce(
            OWNER,
            "postgres-race-quiesce",
            created.job_id,
            1,
            "barrier-race",
        ),
    )
    assert {item.replayed for item in quiesce_results} == {False, True}
    await ledger.mark_quiesced(
        OWNER, quiesce_results[0].operation_id, created.job_id, 1
    )
    receipt = receipt_for(
        job_id=created.job_id,
        conversation_id=created.conversation_id,
        generation=1,
        barrier_id="barrier-race",
        suffix="race",
    )
    release_results = await asyncio.gather(
        ledger.begin_release(
            OWNER,
            "postgres-race-release",
            created.job_id,
            1,
            receipt,
            ReleaseTarget.PARKED,
        ),
        ledger.begin_release(
            OWNER,
            "postgres-race-release",
            created.job_id,
            1,
            receipt,
            ReleaseTarget.PARKED,
        ),
    )
    assert {item.replayed for item in release_results} == {False, True}
    assert await _audit_count(pool) == 6


async def test_postgres_receipt_rejection_rolls_back_operation_and_audit(
    postgres_repository,
):
    repository, pool = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=900))
    created = await ledger.create_job(
        OWNER, "postgres-receipt-001", "default", "docker", "postgres"
    )
    start = await ledger.begin_start(
        OWNER, "postgres-receipt-002", created.job_id, 0, 30
    )
    await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=start.operation_id,
        generation=1,
    )
    _, _, barrier = await quiesce_generation(
        ledger, job_id=created.job_id, generation=1, suffix="reject"
    )
    wrong = receipt_for(
        job_id=uuid4(),
        conversation_id=created.conversation_id,
        generation=1,
        barrier_id=barrier,
        suffix="reject",
    )
    before_audit = await _audit_count(pool)
    async with pool.acquire() as connection:
        before_operations = await connection.fetchval(
            "SELECT count(*) FROM broker_operations"
        )
    with pytest.raises(ReceiptBindingMismatch):
        await ledger.begin_release(
            OWNER,
            "postgres-receipt-bad",
            created.job_id,
            1,
            wrong,
            ReleaseTarget.PARKED,
        )
    async with pool.acquire() as connection:
        after_operations = await connection.fetchval(
            "SELECT count(*) FROM broker_operations"
        )
    assert after_operations == before_operations
    assert await _audit_count(pool) == before_audit


async def test_postgres_concurrent_repeated_setup_is_idempotent(
    postgres_repository,
):
    repository, pool = postgres_repository
    second = PostgresBrokerRepository()
    third = PostgresBrokerRepository()
    try:
        await asyncio.gather(second.setup(pool), third.setup(pool))
        async with pool.acquire() as connection:
            assert await connection.fetchval(
                "SELECT count(*) FROM broker_schema_migrations"
            ) == 2
    finally:
        await second.close()
        await third.close()


async def test_postgres_concurrent_fresh_setup_serializes_bootstrap():
    if not await _reachable():
        pytest.skip(f"no PostgreSQL reachable at {DSN}")
    import asyncpg

    schema = f"broker_fresh_{uuid4().hex}"
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    await admin.close()
    pool = await asyncpg.create_pool(
        DSN,
        min_size=2,
        max_size=4,
        server_settings={"search_path": schema},
    )
    first = PostgresBrokerRepository()
    second = PostgresBrokerRepository()
    try:
        await asyncio.gather(first.setup(pool), second.setup(pool))
        async with pool.acquire() as connection:
            assert await connection.fetchval(
                "SELECT count(*) FROM broker_schema_migrations"
            ) == 2
            assert await connection.fetchval(
                """
                SELECT count(*) FROM information_schema.tables
                WHERE table_schema = current_schema() AND table_name LIKE 'broker_%'
                """
            ) == 6
    finally:
        await first.close()
        await second.close()
        await pool.close()
        admin = await asyncpg.connect(DSN)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


async def test_postgres_append_only_upgrade_records_checksum(
    postgres_repository, monkeypatch
):
    repository, pool = postgres_repository
    await repository.close()
    packaged = _load_packaged_migrations()
    upgrade = Migration.from_bytes(
        3,
        "contract_probe",
        b"CREATE TABLE broker_upgrade_probe (value INTEGER PRIMARY KEY);\n",
    )
    monkeypatch.setattr(
        "openloop.broker.postgres._load_packaged_migrations",
        lambda: (*packaged, upgrade),
    )
    upgraded = PostgresBrokerRepository()
    await upgraded.setup(pool)
    try:
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT name, checksum FROM broker_schema_migrations WHERE version = 3"
            )
            assert dict(row) == {
                "name": "contract_probe",
                "checksum": upgrade.checksum,
            }
    finally:
        await upgraded.close()


@pytest.mark.parametrize(
    ("mutation", "problem"),
    [
        (
            "UPDATE broker_schema_migrations SET checksum = repeat('0', 64) "
            "WHERE version = 1",
            MigrationProblem.CHECKSUM_DRIFT,
        ),
        (
            "INSERT INTO broker_schema_migrations (version, name, checksum) "
            "VALUES (3, 'future', repeat('a', 64))",
            MigrationProblem.FUTURE_VERSION,
        ),
    ],
)
async def test_postgres_setup_fails_closed_on_drift_or_future_version(
    postgres_repository, mutation, problem
):
    repository, pool = postgres_repository
    await repository.close()
    async with pool.acquire() as connection:
        await connection.execute(mutation)
    candidate = PostgresBrokerRepository()
    with pytest.raises(MigrationVersionError) as caught:
        await candidate.setup(pool)
    assert caught.value.problem is problem
    assert candidate._pool is None
    async with pool.acquire() as connection:
        assert await connection.fetchval("SELECT 1") == 1


async def test_postgres_failed_pending_migration_rolls_back_and_detaches(
    postgres_repository, monkeypatch
):
    repository, pool = postgres_repository
    await repository.close()
    packaged = _load_packaged_migrations()
    broken = Migration.from_bytes(
        3,
        "broken",
        (
            b"CREATE TABLE broker_should_rollback (value INTEGER);\n"
            b"SELECT * FROM broker_missing_relation;\n"
        ),
    )
    monkeypatch.setattr(
        "openloop.broker.postgres._load_packaged_migrations",
        lambda: (*packaged, broken),
    )
    candidate = PostgresBrokerRepository()
    with pytest.raises(Exception, match="broker_missing_relation"):
        await candidate.setup(pool)
    assert candidate._pool is None
    async with pool.acquire() as connection:
        assert await connection.fetchval(
            "SELECT to_regclass('broker_should_rollback')"
        ) is None
        assert await connection.fetchval(
            "SELECT max(version) FROM broker_schema_migrations"
        ) == 2
