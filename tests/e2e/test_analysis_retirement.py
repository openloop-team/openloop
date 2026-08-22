"""Fail-closed retirement DDL for the removed dedicated analysis worker."""

from __future__ import annotations

import contextlib
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from openloop.broker.postgres import split_sql_statements
from openloop.db import create_engine
from tests.support.postgres import postgres_dsn, require_postgres

DSN = postgres_dsn()
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


async def _admin(statement: str) -> None:
    """Run one statement outside the test's schema, to create or drop it."""
    engine = await create_engine(DSN, min_size=1, max_size=1)
    try:
        async with engine.begin() as connection:
            # sql-text: DDL naming a schema this suite creates for itself.
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


@pytest.fixture
async def retirement_db():
    await require_postgres(DSN)

    schema = f"analysis_retirement_{uuid4().hex}"
    await _admin(f'CREATE SCHEMA "{schema}"')
    engine = await create_engine(
        DSN, min_size=1, max_size=1, server_settings={"search_path": f'"{schema}"'}
    )
    # One connection in AUTOCOMMIT, held for the test: the retirement script
    # opens and closes its own transaction, so nothing may wrap it in another.
    connection = await engine.connect()
    connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
    try:
        for statement in (
            "CREATE TABLE workflow_instances (workflow TEXT)",
            "CREATE TABLE approvals (action TEXT, tool TEXT)",
            "CREATE TABLE usage (task_kind TEXT)",
            "CREATE TABLE surface_sessions (result_artifact_ref TEXT)",
            "CREATE TABLE analysis_staged_inputs (id INTEGER)",
            "CREATE TABLE analysis_uploads (id INTEGER)",
            "CREATE TABLE analysis_artifacts (id INTEGER)",
            "CREATE TABLE analysis_attempts (id INTEGER)",
            "CREATE TABLE analysis_inputs (id INTEGER)",
        ):
            # sql-text: the fixture's own scratch schema — nothing this project
            # describes as metadata.
            await connection.exec_driver_sql(statement)
        yield connection
    finally:
        with contextlib.suppress(Exception):
            await connection.exec_driver_sql("ROLLBACK")
        await connection.close()
        await engine.dispose()
        await _admin(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


async def _run_retirement(connection) -> None:
    """Apply the retirement script, statement by statement.

    The dialect prepares everything it sends, so a multi-statement file cannot
    go over as one command; the script's own `BEGIN`/`COMMIT` still bracket it,
    which is why the fixture's connection is in AUTOCOMMIT.
    """
    sql = RETIREMENT_SQL.read_text(encoding="utf-8")
    try:
        for statement in split_sql_statements(sql):
            # sql-text: an operator-facing migration file, applied verbatim.
            await connection.exec_driver_sql(statement)
    except BaseException:
        with contextlib.suppress(Exception):
            await connection.exec_driver_sql("ROLLBACK")
        raise


async def _table_exists(connection, name: str) -> bool:
    value = await connection.scalar(
        # sql-text: a presence probe over the catalogue, not over a table.
        text("SELECT to_regclass(:name)::text"),
        {"name": name},
    )
    return value is not None


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
    await retirement_db.exec_driver_sql(insert_sql)

    with pytest.raises(DBAPIError, match=category):
        await _run_retirement(retirement_db)

    for table in DEDICATED_TABLES:
        assert await _table_exists(retirement_db, table), table


async def test_failure_reports_every_nonempty_category(retirement_db):
    await retirement_db.exec_driver_sql("INSERT INTO analysis_uploads VALUES (1)")
    await retirement_db.exec_driver_sql("INSERT INTO usage VALUES ('analysis_worker')")

    with pytest.raises(DBAPIError) as raised:
        await _run_retirement(retirement_db)

    assert "analysis_uploads" in str(raised.value)
    assert "usage.analysis_worker" in str(raised.value)


async def test_missing_shared_table_aborts_audit(retirement_db):
    await retirement_db.exec_driver_sql("DROP TABLE usage")

    with pytest.raises(DBAPIError, match=r"missing required shared tables: usage"):
        await _run_retirement(retirement_db)

    for table in DEDICATED_TABLES:
        assert await _table_exists(retirement_db, table), table


async def test_empty_retirement_preserves_shared_rows_and_is_idempotent(
    retirement_db,
):
    for statement in (
        "INSERT INTO workflow_instances VALUES ('coding_worker')",
        "INSERT INTO approvals VALUES ('github.issues:write', 'github')",
        "INSERT INTO usage VALUES ('coding_worker')",
        "INSERT INTO surface_sessions VALUES ('artifact://workspace/result')",
    ):
        # sql-text: seeding the fixture's own scratch schema.
        await retirement_db.exec_driver_sql(statement)

    await _run_retirement(retirement_db)

    for table in DEDICATED_TABLES:
        assert not await _table_exists(retirement_db, table), table
    assert (
        await retirement_db.scalar(text("SELECT count(*) FROM workflow_instances")) == 1
    )
    assert await retirement_db.scalar(text("SELECT count(*) FROM approvals")) == 1
    assert await retirement_db.scalar(text("SELECT count(*) FROM usage")) == 1
    assert (
        await retirement_db.scalar(text("SELECT count(*) FROM surface_sessions")) == 1
    )

    await _run_retirement(retirement_db)


async def test_absent_dedicated_tables_count_as_empty(retirement_db):
    for table in DEDICATED_TABLES:
        await retirement_db.exec_driver_sql(f'DROP TABLE "{table}"')

    await _run_retirement(retirement_db)
