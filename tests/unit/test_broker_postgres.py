from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from uuid import UUID

import pytest

from openloop.broker.errors import MigrationProblem, MigrationVersionError
from openloop.broker.models import (
    BrokerOwner,
    CommandKind,
    GenerationState,
    JobState,
    OperationResult,
    OperationSource,
    OperationStatus,
    OperationTicket,
    ReleaseTarget,
)
from openloop.broker.postgres import (
    _SCAN_RECOVERY_CANDIDATES,
    BROKER_MIGRATION_LOCK_ID,
    Migration,
    PostgresBrokerRepository,
    _decode_result,
    _decode_ticket,
    _encode_result,
    _encode_ticket,
    _generation_from_row,
    _job_from_row,
    _operation_from_row,
    discover_migrations,
)
from openloop.broker.repository import BrokerRepository


def _write(path: Path, name: str, sql: str) -> None:
    (path / name).write_text(sql, encoding="utf-8")


def test_migration_discovery_sorts_and_hashes_exact_bytes(tmp_path):
    _write(tmp_path, "0002_second.sql", "SELECT 2;\n")
    _write(tmp_path, "0001_initial.sql", "SELECT 1;\n")
    _write(tmp_path, "__init__.py", "")

    migrations = discover_migrations(tmp_path)

    assert [item.version for item in migrations] == [1, 2]
    assert [item.name for item in migrations] == ["initial", "second"]
    assert migrations[0].sql == "SELECT 1;\n"
    assert migrations[0].checksum == (
        "b4e0497804e46e0a0b0b8c31975b062152d551bac49c3c2e80932567b4085dcd"
    )


def test_migration_checksum_changes_with_whitespace(tmp_path):
    _write(tmp_path, "0001_initial.sql", "SELECT 1;\n")
    first = discover_migrations(tmp_path)[0]
    _write(tmp_path, "0001_initial.sql", "SELECT 1; \n")
    second = discover_migrations(tmp_path)[0]
    assert first.checksum != second.checksum


@pytest.mark.parametrize(
    ("names", "version", "problem"),
    [
        (["0000_zero.sql"], 0, MigrationProblem.MALFORMED_NAME),
        (["0001_initial.sql", "0003_gap.sql"], 2, MigrationProblem.NUMBERING_GAP),
        (["1_bad.sql"], 0, MigrationProblem.MALFORMED_NAME),
        (["0001-BAD.sql"], 0, MigrationProblem.MALFORMED_NAME),
        (["0001_initial.SQL"], 0, MigrationProblem.MALFORMED_NAME),
        (
            ["0001_initial.sql", "0001_duplicate.sql"],
            1,
            MigrationProblem.DUPLICATE_VERSION,
        ),
    ],
)
def test_migration_discovery_fails_closed_on_bad_sequences(
    tmp_path, names, version, problem
):
    for name in names:
        _write(tmp_path, name, "SELECT 1;\n")
    with pytest.raises(MigrationVersionError) as caught:
        discover_migrations(tmp_path)
    assert caught.value.version == version
    assert caught.value.problem is problem


class FakeResult:
    """The little of a SQLAlchemy Result that `setup()` reads back."""

    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeConnection:
    """Records what `setup()` sends, in order, as (kind, sql, parameters).

    `text()` statements are recorded by their string so the assertions below
    can keep matching on SQL fragments; a migration file arrives through
    `exec_driver_sql` instead, one statement per call.
    """

    def __init__(self, applied=(), *, fail_on=None):
        self.applied = list(applied)
        self.fail_on = fail_on
        self.events = []

    async def execute(self, statement, parameters=None):
        sql = str(statement)
        self.events.append(("execute", sql, parameters or {}))
        if self.fail_on is not None and self.fail_on in sql:
            raise RuntimeError("schema permission denied")
        if "FROM broker_schema_migrations" in sql:
            return FakeResult(self.applied)
        return FakeResult([])

    async def exec_driver_sql(self, statement):
        self.events.append(("exec_driver_sql", statement, {}))
        if self.fail_on is not None and self.fail_on in statement:
            raise RuntimeError("schema permission denied")
        return FakeResult([])


class Begin(AbstractAsyncContextManager):
    def __init__(self, engine):
        self.engine = engine

    async def __aenter__(self):
        self.engine.connection.events.append(("transaction_enter",))
        return self.engine.connection

    async def __aexit__(self, exc_type, exc, traceback):
        self.engine.connection.events.append(("transaction_exit", exc_type))
        return False


class FakeEngine:
    def __init__(self, connection):
        self.connection = connection
        self.disposed = False

    def begin(self):
        return Begin(self)

    def connect(self):
        return Begin(self)

    async def dispose(self):
        self.disposed = True


async def test_database_clock_is_truncated_for_runtime_identity_deadlines():
    class ClockConnection:
        def __init__(self):
            self.query = None

        async def scalar(self, query):
            self.query = str(query)
            return datetime(2026, 7, 18, 12, 0, tzinfo=UTC)

    connection = ClockConnection()
    value = await PostgresBrokerRepository._database_now(connection)

    assert value.microsecond == 0
    assert connection.query == "SELECT date_trunc('second', clock_timestamp())"


async def test_setup_uses_one_transaction_lock_and_applies_in_order(monkeypatch):
    migrations = (
        Migration.from_bytes(1, "initial", b"SELECT 1;\n"),
        Migration.from_bytes(2, "second", b"SELECT 2;\n"),
    )
    monkeypatch.setattr(
        "openloop.broker.postgres._load_packaged_migrations", lambda: migrations
    )
    connection = FakeConnection()
    engine = FakeEngine(connection)
    repository = PostgresBrokerRepository()

    await repository.setup(engine)

    assert repository._engine is engine
    assert not engine.disposed
    assert connection.events[0] == ("transaction_enter",)
    execute_events = [event for event in connection.events if event[0] == "execute"]
    assert "pg_advisory_xact_lock" in execute_events[0][1]
    assert execute_events[0][2] == {"lock_id": BROKER_MIGRATION_LOCK_ID}
    assert "CREATE TABLE IF NOT EXISTS broker_schema_migrations" in execute_events[1][1]
    read_index = next(
        index
        for index, event in enumerate(connection.events)
        if event[0] == "execute" and "FROM broker_schema_migrations" in event[1]
    )
    migration_indexes = [
        index
        for index, event in enumerate(connection.events)
        if event[0] == "exec_driver_sql" and event[1] in {"SELECT 1", "SELECT 2"}
    ]
    assert read_index < migration_indexes[0] < migration_indexes[1]
    inserts = [
        event
        for event in execute_events
        if "INSERT INTO broker_schema_migrations" in event[1]
    ]
    assert [
        (event[2]["version"], event[2]["name"], event[2]["checksum"])
        for event in inserts
    ] == [
        (1, "initial", migrations[0].checksum),
        (2, "second", migrations[1].checksum),
    ]
    assert connection.events[-1] == ("transaction_exit", None)


async def test_setup_skips_exact_applied_migrations(monkeypatch):
    migration = Migration.from_bytes(1, "initial", b"SELECT 1;\n")
    monkeypatch.setattr(
        "openloop.broker.postgres._load_packaged_migrations", lambda: (migration,)
    )
    connection = FakeConnection(
        applied=[
            {
                "version": 1,
                "name": "initial",
                "checksum": migration.checksum,
            }
        ]
    )
    repository = PostgresBrokerRepository()
    await repository.setup(FakeEngine(connection))
    assert not any(
        event[0] == "exec_driver_sql" and event[1] in migration.sql
        for event in connection.events
    )


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        (
            {"version": 2, "name": "future", "checksum": "a" * 64},
            MigrationProblem.FUTURE_VERSION,
        ),
        (
            {"version": 1, "name": "initial", "checksum": "0" * 64},
            MigrationProblem.CHECKSUM_DRIFT,
        ),
        (
            {"version": 1, "name": "renamed", "checksum": "a" * 64},
            MigrationProblem.CHECKSUM_DRIFT,
        ),
    ],
)
async def test_setup_rejects_future_version_or_applied_drift(monkeypatch, row, problem):
    migration = Migration.from_bytes(1, "initial", b"SELECT 1;\n")
    if row["version"] == 1 and row["name"] == "renamed":
        row = {**row, "checksum": migration.checksum}
    monkeypatch.setattr(
        "openloop.broker.postgres._load_packaged_migrations", lambda: (migration,)
    )
    repository = PostgresBrokerRepository()
    with pytest.raises(MigrationVersionError) as caught:
        await repository.setup(FakeEngine(FakeConnection(applied=[row])))
    assert caught.value.version == row["version"]
    assert caught.value.problem is problem
    assert repository._engine is None


async def test_setup_failure_detaches_without_disposing_borrowed_engine(monkeypatch):
    migration = Migration.from_bytes(1, "initial", b"SELECT fail_here;\n")
    monkeypatch.setattr(
        "openloop.broker.postgres._load_packaged_migrations", lambda: (migration,)
    )
    connection = FakeConnection(fail_on="fail_here")
    engine = FakeEngine(connection)
    repository = PostgresBrokerRepository()

    with pytest.raises(RuntimeError, match="schema permission denied"):
        await repository.setup(engine)

    assert repository._engine is None
    assert not engine.disposed
    assert connection.events[-1][0] == "transaction_exit"
    assert connection.events[-1][1] is RuntimeError


def test_initial_migration_is_packaged_and_contains_required_constraints():
    sql = (
        resources.files("openloop.broker.migrations")
        .joinpath("0001_initial.sql")
        .read_text(encoding="utf-8")
    )
    for table in (
        "broker_schema_migrations",
        "broker_jobs",
        "broker_generations",
        "broker_operations",
        "broker_audit",
    ):
        assert "CREATE TABLE" in sql and table in sql
    for fragment in (
        "PRIMARY KEY (job_id, generation)",
        "REFERENCES broker_jobs",
        "REFERENCES broker_generations",
        "REFERENCES broker_operations",
        "CREATE UNIQUE INDEX broker_one_live_generation_per_job",
        "WHERE state IN ('starting', 'running', 'quiescing', 'quiesced', 'releasing')",
        "CREATE UNIQUE INDEX broker_caller_idempotency",
        "WHERE idempotency_key IS NOT NULL",
        "CHECK (revision > 0)",
        "CHECK (generation >= 0)",
        "CHECK (octet_length(intent_ticket::text) <= 16384)",
        "UNIQUE (operation_id, command_kind)",
    ):
        assert fragment in sql
    assert "CREATE TRIGGER" not in sql.upper()
    assert "CREATE FUNCTION" not in sql.upper()


def test_rpc_authorization_migration_is_packaged_and_fail_closed_for_legacy_rows():
    sql = (
        resources.files("openloop.broker.migrations")
        .joinpath("0002_rpc_authorization.sql")
        .read_text(encoding="utf-8")
    )
    for fragment in (
        "minimum_isolation",
        "control_key_version",
        "control_epoch",
        "control_capability_digest",
        "broker_rpc_audit",
        "num_nonnulls",
        "shared",
        "dedicated",
    ):
        assert fragment in sql


def test_rpc_lifecycle_migration_allows_only_reviewed_intents():
    sql = (
        resources.files("openloop.broker.migrations")
        .joinpath("0003_rpc_lifecycle.sql")
        .read_text(encoding="utf-8")
    )
    for method in (
        "CREATE_JOB",
        "INSPECT_JOB",
        "START_SEGMENT",
        "QUIESCE_SEGMENT",
        "RELEASE_SEGMENT",
        "FINALIZE_JOB",
    ):
        assert f"'{method}'" in sql
    assert "DROP CONSTRAINT broker_rpc_audit_method_check" in sql
    assert "ADD CONSTRAINT broker_rpc_audit_method_check" in sql


def test_recovery_migration_adds_only_bounded_scan_indexes():
    sql = (
        resources.files("openloop.broker.migrations")
        .joinpath("0004_recovery_indexes.sql")
        .read_text(encoding="utf-8")
    )
    for fragment in (
        "broker_generation_checkpoint_recovery",
        "broker_generation_expired_running_recovery",
        "broker_job_finalizing_recovery",
        "WHERE state IN ('quiescing', 'quiesced', 'releasing')",
        "WHERE state = 'running'",
        "WHERE state = 'finalizing'",
    ):
        assert fragment in sql
    assert "ALTER TABLE" not in sql.upper()


def test_recovery_scan_has_separate_index_driven_arms_and_one_database_clock():
    assert _SCAN_RECOVERY_CANDIDATES.count("UNION ALL") == 2
    assert _SCAN_RECOVERY_CANDIDATES.count("clock_timestamp()") == 1
    assert "LEFT JOIN" not in _SCAN_RECOVERY_CANDIDATES
    assert "g.execution_lease_deadline <= recovery_clock.observed_at" in (
        _SCAN_RECOVERY_CANDIDATES
    )
    assert "ORDER BY job_id" in _SCAN_RECOVERY_CANDIDATES


def test_postgres_repository_implements_narrow_protocol():
    assert isinstance(PostgresBrokerRepository(), BrokerRepository)


def test_ticket_and_result_json_round_trip_uses_stable_safe_fields():
    operation_id = UUID("00000000-0000-4000-8000-000000000001")
    job_id = UUID("00000000-0000-4000-8000-000000000002")
    conversation_id = UUID("00000000-0000-4000-8000-000000000003")
    ticket = OperationTicket(
        operation_id=operation_id,
        command=CommandKind.BEGIN_START,
        job_id=job_id,
        conversation_id=conversation_id,
        generation=1,
        job_state=JobState.CREATED,
        generation_state=GenerationState.STARTING,
    )
    result = OperationResult(
        operation_id=operation_id,
        command=CommandKind.MARK_RUNNING,
        job_id=job_id,
        generation=1,
        job_state=JobState.ACTIVE,
        generation_state=GenerationState.RUNNING,
    )
    assert _decode_ticket(_encode_ticket(ticket)) == ticket
    assert _decode_result(_encode_result(result)) == result
    assert "runtime_ref" not in _encode_ticket(ticket)
    assert "capability_digest" not in _encode_result(result)


def test_explicit_row_mappers_round_trip_domain_types_and_redact_repr():
    now = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
    job_id = UUID("00000000-0000-4000-8000-000000000011")
    conversation_id = UUID("00000000-0000-4000-8000-000000000012")
    operation_id = UUID("00000000-0000-4000-8000-000000000013")
    ticket = OperationTicket(
        operation_id=operation_id,
        command=CommandKind.BEGIN_RELEASE,
        job_id=job_id,
        conversation_id=conversation_id,
        generation=1,
        job_state=JobState.ACTIVE,
        generation_state=GenerationState.RELEASING,
    )
    result = OperationResult(
        operation_id=operation_id,
        command=CommandKind.MARK_RELEASED,
        job_id=job_id,
        generation=1,
        job_state=JobState.PARKED,
        generation_state=GenerationState.RELEASED,
    )
    job = _job_from_row(
        {
            "job_id": job_id,
            "conversation_id": conversation_id,
            "tenant_id": "tenant-a",
            "workload_subject": "workload-a",
            "profile": "default",
            "runtime_driver": "docker",
            "durable_state_driver": "postgres",
            "state": "parked",
            "revision": 7,
            "generation": 1,
            "current_generation": None,
            "pending_operation_id": None,
            "durable_state_ref": "durable://protected",
            "durable_key_version": "key-v1",
            "durable_digest": "a" * 64,
            "terminal_outcome": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    generation = _generation_from_row(
        {
            "job_id": job_id,
            "generation": 1,
            "state": "released",
            "revision": 6,
            "previous_job_state": "created",
            "start_operation_id": operation_id,
            "pending_operation_id": None,
            "runtime_ref": "runtime://protected",
            "durable_state_ref": "durable://protected",
            "runtime_key_version": "runtime-key-v1",
            "durable_key_version": "durable-key-v1",
            "capability_digest": "b" * 64,
            "durable_digest": "a" * 64,
            "execution_lease_deadline": now,
            "barrier_id": "barrier-1",
            "receipt_issuer": "issuer-1",
            "receipt_id": "receipt-1",
            "receipt_tenant_id": "tenant-a",
            "receipt_job_id": job_id,
            "receipt_conversation_id": conversation_id,
            "receipt_generation": 1,
            "receipt_barrier_id": "barrier-1",
            "receipt_artifact_id": "artifact-1",
            "receipt_base_commit": "c" * 40,
            "receipt_ciphertext_sha256": "d" * 64,
            "receipt_plaintext_sha256": "e" * 64,
            "receipt_byte_count": 10,
            "receipt_store_version": "store-v1",
            "receipt_envelope_version": "envelope-v1",
            "receipt_key_version": "key-v1",
            "receipt_durable_write_sequence": 3,
            "release_target": "parked",
            "release_terminal_outcome": None,
            "failure_reason_code": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    operation = _operation_from_row(
        {
            "operation_id": operation_id,
            "tenant_id": "tenant-a",
            "workload_subject": "workload-a",
            "source": "caller",
            "idempotency_key": "postgres-key-0001",
            "command_kind": "begin_release",
            "request_digest": "f" * 64,
            "job_id": job_id,
            "generation": 1,
            "status": "completed",
            "intent_ticket": _encode_ticket(ticket),
            "completion_result": _encode_result(result),
            "created_at": now,
            "updated_at": now,
        }
    )
    assert job.owner == BrokerOwner("tenant-a", "workload-a")
    assert job.state is JobState.PARKED
    assert generation.state is GenerationState.RELEASED
    assert generation.release_target is ReleaseTarget.PARKED
    assert operation.source is OperationSource.CALLER
    assert operation.status is OperationStatus.COMPLETED
    assert operation.intent_ticket == ticket
    assert operation.completion_result == result
    rendered = repr((job, generation, operation))
    assert "runtime://protected" not in rendered
    assert "durable://protected" not in rendered
    assert "a" * 64 not in rendered
