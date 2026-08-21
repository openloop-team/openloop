"""Unit coverage for the shared PostgreSQL connection pool's sizing.

Ownership and binding moved to `test_db_engine.py` with the engine seam
(ADR 0007); the pool bounds are still configuration, and still validated here.
"""

import pytest
from pydantic import ValidationError

from openloop.usage.postgres import PostgresUsageStore
from tests.support.settings import IsolatedSettings as Settings


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
