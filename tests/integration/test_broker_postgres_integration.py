"""Broker repository contract and concurrency tests against real PostgreSQL."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import text

from openloop.broker.errors import (
    ConcurrentMutation,
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
    _generation_from_row,
    _job_from_row,
    _load_packaged_migrations,
)
from openloop.db import BorrowedEngineStore, create_engine
from tests.support.broker_repository_contract import (
    OWNER,
    SequenceIds,
    begin_generation_start,
    exercise_complete_lifecycle,
    mark_generation_running,
    quiesce_generation,
    receipt_for,
)
from tests.support.postgres import postgres_dsn, require_postgres

DSN = postgres_dsn()

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture
async def postgres_repository():
    await require_postgres(DSN)
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
    # Same schema, the other bind. Task 16 removes the pool and the isinstance.
    engine = await create_engine(
        DSN,
        min_size=1,
        max_size=10,
        server_settings={"search_path": schema},
    )
    repository = PostgresBrokerRepository()
    try:
        await repository.setup(
            engine if isinstance(repository, BorrowedEngineStore) else pool
        )
        yield repository, pool, engine
    finally:
        await repository.close()
        await engine.dispose()
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
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds())

    created = await ledger.create_job(
        OWNER, "postgres-create-0001", "default", "docker", "postgres"
    )
    replay = await ledger.create_job(
        OWNER, "postgres-create-0001", "default", "docker", "postgres"
    )
    assert replay.replayed is True
    assert replay.job_id == created.job_id

    start = await begin_generation_start(
        ledger,
        idempotency_key="postgres-start-00001",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
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
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=100))
    created = await ledger.create_job(
        OWNER, "postgres-create-0002", "default", "docker", "postgres"
    )
    await begin_generation_start(
        ledger,
        idempotency_key="postgres-start-fail1",
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
    replay = await ledger.abandon_generation(
        OWNER,
        created.job_id,
        1,
        GenerationState.STARTING,
        "start_failed",
        replay_operation_id=abandoned.operation_id,
    )
    assert replay.replayed is True
    second = await begin_generation_start(
        ledger,
        idempotency_key="postgres-start-next1",
        job_id=created.job_id,
        expected_generation=1,
        execution_lease_seconds=30,
    )
    assert second.generation == 2
    assert await _audit_count(pool) == 4


async def test_postgres_complete_lifecycle_shared_contract(postgres_repository):
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=200))
    trace = await exercise_complete_lifecycle(ledger)
    assert trace.snapshots[-1].state is JobState.TERMINAL
    assert trace.snapshots[-1].generation == 2
    assert await _audit_count(pool) == 15


async def test_postgres_restart_preserves_inspection_and_exact_replay(
    postgres_repository,
):
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=300))
    created = await ledger.create_job(
        OWNER, "postgres-restart-001", "default", "docker", "postgres"
    )
    start = await begin_generation_start(
        ledger,
        idempotency_key="postgres-restart-002",
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
    await repository.close()

    restarted = PostgresBrokerRepository()
    await restarted.setup(
        engine if isinstance(restarted, BorrowedEngineStore) else pool
    )
    try:
        restarted_ledger = BrokerLedger(restarted, id_factory=SequenceIds(start=400))
        snapshot = await restarted_ledger.inspect_job(OWNER, created.job_id)
        assert snapshot.state is JobState.ACTIVE
        replay = await begin_generation_start(
            restarted_ledger,
            idempotency_key="postgres-restart-002",
            job_id=created.job_id,
            expected_generation=0,
            execution_lease_seconds=30,
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
    repository, pool, engine = postgres_repository
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
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=600))
    results = await asyncio.gather(
        ledger.create_job(
            OWNER, "postgres-race-conflict", "default", "docker", "postgres"
        ),
        ledger.create_job(OWNER, "postgres-race-conflict", "gpu", "docker", "postgres"),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, IdempotencyConflict) for result in results) == 1
    assert await _audit_count(pool) == 1


async def test_postgres_concurrent_starts_preserve_one_live_generation(
    postgres_repository,
):
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=700))
    created = await ledger.create_job(
        OWNER, "postgres-race-job001", "default", "docker", "postgres"
    )
    results = await asyncio.gather(
        begin_generation_start(
            ledger,
            idempotency_key="postgres-race-start-a",
            job_id=created.job_id,
            expected_generation=0,
            execution_lease_seconds=30,
        ),
        begin_generation_start(
            ledger,
            idempotency_key="postgres-race-start-b",
            job_id=created.job_id,
            expected_generation=0,
            execution_lease_seconds=30,
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert (
        sum(
            isinstance(result, (InvalidTransition, StaleGeneration))
            for result in results
        )
        == 1
    )
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
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=800))
    created = await ledger.create_job(
        OWNER, "postgres-race-flow01", "default", "docker", "postgres"
    )
    start = await begin_generation_start(
        ledger,
        idempotency_key="postgres-race-flow02",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
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
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=900))
    created = await ledger.create_job(
        OWNER, "postgres-receipt-001", "default", "docker", "postgres"
    )
    start = await begin_generation_start(
        ledger,
        idempotency_key="postgres-receipt-002",
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


async def test_postgres_recovery_scan_and_internal_operations_are_durable(
    postgres_repository,
):
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=1000))
    created = await ledger.create_job(
        OWNER, "postgres-recovery-01", "default", "docker", "postgres"
    )
    start = await begin_generation_start(
        ledger,
        idempotency_key="postgres-recovery-02",
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
    _, _, barrier = await quiesce_generation(
        ledger, job_id=created.job_id, generation=1, suffix="recovery"
    )

    candidates = await ledger.scan_recovery_candidates(limit=1)
    assert [candidate.job_id for candidate in candidates] == [created.job_id]
    receipt = receipt_for(
        job_id=created.job_id,
        conversation_id=created.conversation_id,
        generation=1,
        barrier_id=barrier,
        suffix="recovery",
    )
    release = await ledger.begin_internal_release(
        OWNER, created.job_id, 1, receipt, ReleaseTarget.PARKED
    )
    await ledger.mark_released(OWNER, release.operation_id, created.job_id, 1)
    finalize = await ledger.begin_internal_finalize(
        OWNER, created.job_id, 1, TerminalOutcome.SUCCESS
    )
    await ledger.mark_terminal(OWNER, finalize.operation_id, created.job_id)

    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT source, idempotency_key, command_kind, status
            FROM broker_operations
            WHERE operation_id = ANY($1::uuid[])
            ORDER BY command_kind
            """,
            [release.operation_id, finalize.operation_id],
        )
    operation_metadata = {
        (row["command_kind"], row["source"], row["idempotency_key"]) for row in rows
    }
    assert operation_metadata == {
        ("begin_release", "internal", None),
        ("begin_finalize", "internal", None),
    }
    assert all(row["status"] == "completed" for row in rows)
    assert await ledger.scan_recovery_candidates() == ()


async def test_postgres_recovery_running_expiry_uses_database_time(
    postgres_repository,
):
    repository, pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=1100))
    created = await ledger.create_job(
        OWNER, "postgres-expiry-001", "default", "docker", "postgres"
    )
    start = await begin_generation_start(
        ledger,
        idempotency_key="postgres-expiry-002",
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
    assert await ledger.scan_recovery_candidates() == ()
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE broker_generations
            SET execution_lease_deadline = clock_timestamp() - interval '1 second'
            WHERE job_id = $1 AND generation = 1
            """,
            created.job_id,
        )

    candidates = await ledger.scan_recovery_candidates()

    assert [candidate.job_id for candidate in candidates] == [created.job_id]
    assert candidates[0].generation_state is GenerationState.RUNNING
    assert candidates[0].observed_at.tzinfo is not None


async def test_postgres_concurrent_repeated_setup_is_idempotent(
    postgres_repository,
):
    repository, pool, engine = postgres_repository
    second = PostgresBrokerRepository()
    third = PostgresBrokerRepository()
    try:
        await asyncio.gather(second.setup(engine), third.setup(engine))
        async with pool.acquire() as connection:
            assert (
                await connection.fetchval(
                    "SELECT count(*) FROM broker_schema_migrations"
                )
                == 4
            )
    finally:
        await second.close()
        await third.close()


async def test_postgres_concurrent_fresh_setup_serializes_bootstrap():
    await require_postgres(DSN)
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
    # Two engines on the same fresh schema: the advisory lock in setup() is
    # what has to serialize them, not a shared connection.
    engine = await create_engine(
        DSN,
        min_size=2,
        max_size=4,
        server_settings={"search_path": schema},
    )
    first = PostgresBrokerRepository()
    second = PostgresBrokerRepository()
    try:
        await asyncio.gather(first.setup(engine), second.setup(engine))
        async with pool.acquire() as connection:
            assert (
                await connection.fetchval(
                    "SELECT count(*) FROM broker_schema_migrations"
                )
                == 4
            )
            assert (
                await connection.fetchval(
                    """
                SELECT count(*) FROM information_schema.tables
                WHERE table_schema = current_schema() AND table_name LIKE 'broker_%'
                """
                )
                == 6
            )
    finally:
        await first.close()
        await second.close()
        await engine.dispose()
        await pool.close()
        admin = await asyncpg.connect(DSN)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


async def test_postgres_append_only_upgrade_records_checksum(
    postgres_repository, monkeypatch
):
    repository, pool, engine = postgres_repository
    await repository.close()
    packaged = _load_packaged_migrations()
    upgrade = Migration.from_bytes(
        5,
        "contract_probe",
        b"CREATE TABLE broker_upgrade_probe (value INTEGER PRIMARY KEY);\n",
    )
    monkeypatch.setattr(
        "openloop.broker.postgres._load_packaged_migrations",
        lambda: (*packaged, upgrade),
    )
    upgraded = PostgresBrokerRepository()
    await upgraded.setup(engine)
    try:
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT name, checksum FROM broker_schema_migrations WHERE version = 5"
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
            "VALUES (5, 'future', repeat('a', 64))",
            MigrationProblem.FUTURE_VERSION,
        ),
    ],
)
async def test_postgres_setup_fails_closed_on_drift_or_future_version(
    postgres_repository, mutation, problem
):
    repository, pool, engine = postgres_repository
    await repository.close()
    async with pool.acquire() as connection:
        await connection.execute(mutation)
    candidate = PostgresBrokerRepository()
    with pytest.raises(MigrationVersionError) as caught:
        await candidate.setup(engine)
    assert caught.value.problem is problem
    assert candidate._engine is None
    async with pool.acquire() as connection:
        assert await connection.fetchval("SELECT 1") == 1


async def test_postgres_failed_pending_migration_rolls_back_and_detaches(
    postgres_repository, monkeypatch
):
    repository, pool, engine = postgres_repository
    await repository.close()
    packaged = _load_packaged_migrations()
    broken = Migration.from_bytes(
        5,
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
        await candidate.setup(engine)
    assert candidate._engine is None
    async with pool.acquire() as connection:
        assert (
            await connection.fetchval("SELECT to_regclass('broker_should_rollback')")
            is None
        )
        assert (
            await connection.fetchval(
                "SELECT max(version) FROM broker_schema_migrations"
            )
            == 4
        )


async def test_migrations_apply_to_an_empty_schema_through_sqlalchemy(
    postgres_repository,
):
    """Every migration file holds several statements; they must all land.

    The fixture already hands back a schema created for this test alone and a
    repository whose `setup()` ran against it, so there is nothing to tear down
    first — which matters, because `execute(text(...))` takes asyncpg's
    prepared-statement path and cannot run `DROP SCHEMA …; CREATE SCHEMA …` as
    one statement.
    """
    _repository, _pool, engine = postgres_repository

    async with engine.connect() as conn:
        applied = (
            (
                await conn.execute(
                    text(
                        "SELECT version FROM broker_schema_migrations ORDER BY version"
                    )
                )
            )
            .scalars()
            .all()
        )
        tables = (
            (
                await conn.execute(
                    text(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = current_schema() ORDER BY tablename"
                    )
                )
            )
            .scalars()
            .all()
        )

    assert applied == [1, 2, 3, 4]
    # 0001 alone holds ten statements; a runner that stopped at the first would
    # still record the version.
    assert "broker_audit" in tables
    assert "broker_rpc_audit" in tables


async def test_update_job_refuses_a_stale_expected_revision(postgres_repository):
    """The optimistic predicate must stay inside the writing statement.

    Driving this through the ledger cannot work: `_job()` reads `FOR UPDATE`
    inside the same transaction that writes, and hands that freshly-read record
    straight to `_update_job` as `before` — so a bump made beforehand is simply
    read as the current revision and the guard passes. The race the guard exists
    for happens between another transaction's read and write, which a sequential
    test cannot stage. Calling `_update_job` with a `before` whose revision is
    deliberately stale reproduces exactly what that race presents to the
    statement, deterministically.
    """
    repository, _pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds())
    created = await ledger.create_job(
        OWNER, "postgres-fence-0001", "default", "docker", "postgres"
    )

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT * FROM broker_jobs WHERE job_id = :job_id"),
                    {"job_id": created.job_id},
                )
            )
            .mappings()
            .first()
        )
    current = _job_from_row(row)

    stale = replace(current, revision=current.revision + 1)
    after = replace(current, revision=current.revision + 1)

    with pytest.raises(ConcurrentMutation):
        async with engine.begin() as conn:
            await PostgresBrokerRepository._update_job(conn, stale, after)

    # And the matching revision still writes, so the guard is a guard and not a
    # statement that never matches.
    async with engine.begin() as conn:
        await PostgresBrokerRepository._update_job(conn, current, after)


async def test_update_generation_refuses_a_stale_expected_revision(
    postgres_repository,
):
    """The generation guard, tested the same way as the job guard.

    `begin_generation_start` is the contract helper that supplies `begin_start`'s
    nine arguments; it leaves generation 1 in state 'starting' with a pending
    operation, which is a row every CHECK on the table accepts. `after` changes
    only the revision, so nothing that held for the stored row stops holding.
    """
    repository, _pool, engine = postgres_repository
    ledger = BrokerLedger(repository, id_factory=SequenceIds())

    created = await ledger.create_job(
        OWNER, "postgres-genfence-0001", "default", "docker", "postgres"
    )
    await begin_generation_start(
        ledger,
        idempotency_key="postgres-genfence-start-1",
        job_id=created.job_id,
        expected_generation=0,
        execution_lease_seconds=30,
    )

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        "SELECT * FROM broker_generations "
                        "WHERE job_id = :job_id AND generation = :generation"
                    ),
                    {"job_id": created.job_id, "generation": 1},
                )
            )
            .mappings()
            .first()
        )
    current = _generation_from_row(row)

    stale = replace(current, revision=current.revision + 1)
    after = replace(current, revision=current.revision + 1)

    with pytest.raises(ConcurrentMutation):
        async with engine.begin() as conn:
            await PostgresBrokerRepository._update_generation(conn, stale, after)

    # The matching revision still writes, so the guard is a guard and not a
    # statement that never matches.
    async with engine.begin() as conn:
        await PostgresBrokerRepository._update_generation(conn, current, after)

    async with engine.connect() as conn:
        stored = await conn.scalar(
            text(
                "SELECT revision FROM broker_generations "
                "WHERE job_id = :job_id AND generation = :generation"
            ),
            {"job_id": created.job_id, "generation": 1},
        )
    assert stored == current.revision + 1
