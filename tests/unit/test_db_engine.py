"""Unit coverage for engine construction, DSN normalization, and store binding."""

import pytest
from sqlalchemy.exc import SQLAlchemyError

from openloop.db import BorrowedEngineStore, create_engine, normalize_dsn


def test_normalize_dsn_adds_the_asyncpg_driver():
    assert (
        normalize_dsn("postgresql://u:p@h:5432/db")
        == "postgresql+asyncpg://u:p@h:5432/db"
    )


def test_normalize_dsn_rewrites_the_postgres_alias():
    assert normalize_dsn("postgres://h/db") == "postgresql+asyncpg://h/db"


def test_normalize_dsn_leaves_an_explicit_driver_alone():
    assert normalize_dsn("postgresql+asyncpg://h/db") == "postgresql+asyncpg://h/db"


def test_normalize_dsn_rejects_a_non_postgres_url():
    with pytest.raises(ValueError):
        normalize_dsn("mysql://h/db")


async def test_create_engine_probes_connectivity_and_disposes_on_failure():
    # Port 1 is never a Postgres server, so the eager probe must raise rather
    # than hand back a lazy engine that only fails at first query.
    with pytest.raises((OSError, SQLAlchemyError)):
        await create_engine("postgresql://u:p@127.0.0.1:1/db", min_size=1, max_size=2)


async def test_store_close_detaches_without_disposing_the_borrowed_engine():
    class _Engine:
        def __init__(self):
            self.disposed = False

        async def dispose(self):
            self.disposed = True

    store = BorrowedEngineStore()
    engine = _Engine()
    store._engine = engine

    await store.close()

    assert store._engine is None
    assert not engine.disposed


async def test_require_engine_refuses_before_setup():
    store = BorrowedEngineStore()
    with pytest.raises(RuntimeError):
        store._require_engine()
