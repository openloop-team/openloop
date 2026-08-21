"""Unit coverage for shared PostgreSQL pool ownership and sizing."""

import pytest
from pydantic import ValidationError

from openloop.postgres import BorrowedPostgresStore
from openloop.usage.postgres import PostgresUsageStore
from tests.support.settings import IsolatedSettings as Settings


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
        Settings(**kwargs)
