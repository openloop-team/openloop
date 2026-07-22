"""Fail-closed retirement DDL for the removed dedicated analysis worker."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest


DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop_agents",
)
RETIREMENT_SQL = (
    Path(__file__).resolve().parents[2]
    / "ops"
    / "postgres"
    / "2026-07-22-retire-analysis-worker.sql"
)
DEDICATED_TABLES = (
    "analysis_staged_inputs",
    "analysis_uploads",
    "analysis_artifacts",
    "analysis_attempts",
    "analysis_inputs",
)

pytestmark = [pytest.mark.e2e, pytest.mark.postgres, pytest.mark.serial]


async def _reachable() -> bool:
    try:
        connection = await asyncpg.connect(DSN, timeout=3)
        await connection.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def retirement_db():
    if not await _reachable():
        pytest.skip(f"no PostgreSQL reachable at {DSN}")

    schema = f"analysis_retirement_{uuid4().hex}"
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    await admin.close()
    connection = await asyncpg.connect(
        DSN, server_settings={"search_path": f'"{schema}"'}
    )
    try:
        await connection.execute(
            """
            CREATE TABLE workflow_instances (workflow TEXT);
            CREATE TABLE approvals (action TEXT, tool TEXT);
            CREATE TABLE usage (task_kind TEXT);
            CREATE TABLE surface_sessions (result_artifact_ref TEXT);
            CREATE TABLE analysis_staged_inputs (id INTEGER);
            CREATE TABLE analysis_uploads (id INTEGER);
            CREATE TABLE analysis_artifacts (id INTEGER);
            CREATE TABLE analysis_attempts (id INTEGER);
            CREATE TABLE analysis_inputs (id INTEGER);
            """
        )
        yield connection
    finally:
        with contextlib.suppress(Exception):
            await connection.execute("ROLLBACK")
        await connection.close()
        admin = await asyncpg.connect(DSN)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


async def _run_retirement(connection) -> None:
    sql = RETIREMENT_SQL.read_text(encoding="utf-8")
    try:
        await connection.execute(sql)
    except BaseException:
        with contextlib.suppress(Exception):
            await connection.execute("ROLLBACK")
        raise


async def _table_exists(connection, name: str) -> bool:
    return (
        await connection.fetchval("SELECT to_regclass($1)::text", name)
        is not None
    )


@pytest.mark.parametrize(
    ("category", "insert_sql"),
    [
        ("analysis_staged_inputs", "INSERT INTO analysis_staged_inputs VALUES (1)"),
        ("analysis_uploads", "INSERT INTO analysis_uploads VALUES (1)"),
        ("analysis_artifacts", "INSERT INTO analysis_artifacts VALUES (1)"),
        ("analysis_attempts", "INSERT INTO analysis_attempts VALUES (1)"),
        ("analysis_inputs", "INSERT INTO analysis_inputs VALUES (1)"),
        (
            "workflow_instances.analysis_worker",
            "INSERT INTO workflow_instances VALUES ('analysis_worker')",
        ),
        (
            "approvals.analysis_action",
            "INSERT INTO approvals VALUES ('analysis.report:write', 'other')",
        ),
        (
            "approvals.analysis_tool",
            "INSERT INTO approvals VALUES ('other.action', 'analysis')",
        ),
        (
            "usage.analysis_worker",
            "INSERT INTO usage VALUES ('analysis_worker')",
        ),
        (
            "surface_sessions.analysis_artifact",
            "INSERT INTO surface_sessions VALUES ('analysis://job/report.md')",
        ),
    ],
)
async def test_nonempty_category_aborts_without_dropping_tables(
    retirement_db, category, insert_sql
):
    await retirement_db.execute(insert_sql)

    with pytest.raises(asyncpg.PostgresError, match=category):
        await _run_retirement(retirement_db)

    for table in DEDICATED_TABLES:
        assert await _table_exists(retirement_db, table), table


async def test_failure_reports_every_nonempty_category(retirement_db):
    await retirement_db.execute("INSERT INTO analysis_uploads VALUES (1)")
    await retirement_db.execute("INSERT INTO usage VALUES ('analysis_worker')")

    with pytest.raises(asyncpg.PostgresError) as raised:
        await _run_retirement(retirement_db)

    assert "analysis_uploads" in str(raised.value)
    assert "usage.analysis_worker" in str(raised.value)


async def test_missing_shared_table_aborts_audit(retirement_db):
    await retirement_db.execute("DROP TABLE usage")

    with pytest.raises(
        asyncpg.PostgresError, match=r"missing required shared tables: usage"
    ):
        await _run_retirement(retirement_db)

    for table in DEDICATED_TABLES:
        assert await _table_exists(retirement_db, table), table


async def test_empty_retirement_preserves_shared_rows_and_is_idempotent(
    retirement_db,
):
    await retirement_db.execute(
        """
        INSERT INTO workflow_instances VALUES ('coding_worker');
        INSERT INTO approvals VALUES ('github.issues:write', 'github');
        INSERT INTO usage VALUES ('coding_worker');
        INSERT INTO surface_sessions VALUES ('artifact://workspace/result');
        """
    )

    await _run_retirement(retirement_db)

    for table in DEDICATED_TABLES:
        assert not await _table_exists(retirement_db, table), table
    assert await retirement_db.fetchval("SELECT count(*) FROM workflow_instances") == 1
    assert await retirement_db.fetchval("SELECT count(*) FROM approvals") == 1
    assert await retirement_db.fetchval("SELECT count(*) FROM usage") == 1
    assert await retirement_db.fetchval("SELECT count(*) FROM surface_sessions") == 1

    await _run_retirement(retirement_db)


async def test_absent_dedicated_tables_count_as_empty(retirement_db):
    for table in DEDICATED_TABLES:
        await retirement_db.execute(f'DROP TABLE "{table}"')

    await _run_retirement(retirement_db)
