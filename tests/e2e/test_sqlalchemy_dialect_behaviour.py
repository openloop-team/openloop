"""What the asyncpg *dialect* does differently from the asyncpg *driver*.

Every statement in this project rests on these answers, and each one fails at
run time rather than at construction. If one of these fails, stop and report it:
the statements written against it are wrong, not the test.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from openloop.db import create_engine
from tests.support.postgres import postgres_dsn, require_postgres

pytestmark = [pytest.mark.e2e, pytest.mark.postgres, pytest.mark.serial]


@pytest.fixture
async def engine():
    dsn = postgres_dsn()
    await require_postgres(dsn)
    eng = await create_engine(dsn, min_size=1, max_size=4)
    try:
        yield eng
    finally:
        await eng.dispose()


async def test_jsonb_comes_back_decoded_not_as_text(engine):
    """The dialect registers json/jsonb codecs, so `json.loads(row[...])` breaks."""
    async with engine.connect() as conn:
        row = (
            (await conn.execute(text("SELECT '{\"a\": 1}'::jsonb AS doc")))
            .mappings()
            .first()
        )
    assert row["doc"] == {"a": 1}


async def test_named_parameters_replace_positional_ones(engine):
    async with engine.connect() as conn:
        value = await conn.scalar(text("SELECT CAST(:n AS int) + 1"), {"n": 41})
    assert value == 42


def test_a_postfix_cast_swallows_the_bind_parameter():
    """Why every cast in this codebase is written CAST(x AS t).

    text()'s bind regex refuses a name followed by a colon, so `:n::int` is not
    a parameter at all — it reaches Postgres as literal text.
    """
    assert list(text("SELECT :n::int")._bindparams.keys()) == []
    assert list(text("SELECT CAST(:n AS int)")._bindparams.keys()) == ["n"]


async def test_rowcount_replaces_the_status_string(engine):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TEMP TABLE t (id int)"))
        result = await conn.execute(text("INSERT INTO t (id) VALUES (1), (2), (3)"))
        assert result.rowcount == 3


async def test_a_multi_statement_script_must_be_split(engine):
    """The broker's migration files hold several statements per file.

    Pinned in the negative, and this is the answer the conversion has to build
    on: the dialect routes every statement through `prepare`, including
    `exec_driver_sql`, and a prepared statement carries exactly one command.
    There is no simple-query escape hatch. Multi-statement scripts are split on
    statement boundaries and applied one execute at a time.
    """
    script = "CREATE TEMP TABLE a (id int);\nCREATE TEMP TABLE b (id int);"

    async with engine.begin() as conn:
        with pytest.raises(ProgrammingError):
            await conn.exec_driver_sql(script)

    async with engine.begin() as conn:
        for statement in (part for part in script.split(";") if part.strip()):
            await conn.exec_driver_sql(statement)
        present = await conn.scalar(
            text("SELECT count(*) FROM pg_class WHERE relname IN ('a', 'b')")
        )
    assert present == 2


async def test_a_pgvector_parameter_needs_an_explicit_cast(engine):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        distance = await conn.scalar(
            text("SELECT CAST(:a AS vector(3)) <=> CAST(:b AS vector(3))"),
            {"a": "[1.0,0.0,0.0]", "b": "[0.0,1.0,0.0]"},
        )
    assert distance == pytest.approx(1.0)


async def test_a_session_advisory_lock_survives_across_statements(engine):
    """The connection is the lease — it must not return to the pool mid-hold."""
    conn = await engine.connect()
    conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
    try:
        assert await conn.scalar(text("SELECT pg_try_advisory_lock(:k)"), {"k": 4242})
        async with engine.connect() as other:
            other = await other.execution_options(isolation_level="AUTOCOMMIT")
            assert not await other.scalar(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": 4242}
            )
        assert await conn.scalar(text("SELECT pg_advisory_unlock(:k)"), {"k": 4242})
    finally:
        await conn.close()
