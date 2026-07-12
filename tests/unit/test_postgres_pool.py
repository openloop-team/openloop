"""Unit coverage for shared PostgreSQL pool ownership and sizing."""

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from openloop.config import Settings
from openloop.postgres import BorrowedPostgresStore, create_pool
from openloop.usage.postgres import PostgresUsageStore


class _Pool:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


async def test_store_close_detaches_without_closing_borrowed_pool():
    store = BorrowedPostgresStore()
    pool = _Pool()
    store._pool = pool

    await store.close()

    assert store._pool is None
    assert not pool.closed


async def test_create_pool_forwards_explicit_bounds(monkeypatch):
    pool = _Pool()

    async def asyncpg_create_pool(dsn, *, min_size, max_size):
        assert dsn == "postgresql://test"
        assert (min_size, max_size) == (2, 9)
        return pool

    monkeypatch.setitem(
        sys.modules,
        "asyncpg",
        SimpleNamespace(create_pool=asyncpg_create_pool),
    )

    assert await create_pool("postgresql://test", min_size=2, max_size=9) is pool


def test_stores_no_longer_accept_a_dsn():
    with pytest.raises(TypeError):
        PostgresUsageStore("postgresql://test")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"postgres_pool_min_size": -1},
        {"postgres_pool_max_size": 0},
        {"postgres_pool_min_size": 11, "postgres_pool_max_size": 10},
    ],
)
def test_invalid_pool_sizes_fail_during_settings_construction(kwargs):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)
