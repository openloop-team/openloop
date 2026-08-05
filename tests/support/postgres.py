"""Reachability gate shared by the Postgres-backed suites.

Each suite keeps its own default DSN so a developer with the compose stack up
just runs them, and skips cleanly when no database is around. CI instead sets
``OPENLOOP_REQUIRE_POSTGRES=1``, which turns that skip into a failure: a
database that never came up must break the build rather than quietly reduce the
Postgres suites to a no-op. The skip is a convenience for laptops — never a way
for the gate to disappear without anyone noticing.
"""

from __future__ import annotations

import os

import pytest

DSN_ENV = "OPENLOOP_TEST_DATABASE_URL"
REQUIRE_ENV = "OPENLOOP_REQUIRE_POSTGRES"

# The disposable database `mise run test-postgres-up` provisions: a distinct
# port and database from the compose stack on 5432, because these suites are
# destructive (CREATE/DROP SCHEMA, and the e2e suite writes to shared tables
# with no schema isolation at all). Pointing them at the development database
# would delete real rows, which is the whole reason tests read their own
# variable instead of the runtime's DATABASE_URL.
DEFAULT_DSN = "postgresql://openloop:openloop-test@localhost:5433/openloop_test"


def postgres_dsn() -> str:
    """Return the DSN for the Postgres suites — the test database, never the app's."""
    return os.environ.get(DSN_ENV, DEFAULT_DSN)


async def require_postgres(dsn: str) -> None:
    """Skip the calling test unless Postgres answers — or fail if CI demands it."""
    try:
        import asyncpg

        connection = await asyncpg.connect(dsn, timeout=3)
        await connection.close()
        return
    except Exception as exc:  # noqa: BLE001 — any failure to connect gates alike
        reason = f"no Postgres reachable at {dsn}: {exc}"

    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{reason} (set {REQUIRE_ENV}=1, so this is a failure)")
    pytest.skip(reason)
