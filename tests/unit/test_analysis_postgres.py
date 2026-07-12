"""Unit coverage for sealed-analysis Postgres failure boundaries."""

import pytest

from openloop.analysis.postgres import PostgresAnalysisAttemptStore


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Pool:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False

    def acquire(self):
        return _Acquire(self.connection)

    async def close(self):
        self.closed = True


async def test_setup_detaches_without_closing_borrowed_pool_on_schema_failure():
    class _FailingConnection:
        async def execute(self, query):
            raise RuntimeError("schema permission denied")

    pool = _Pool(_FailingConnection())
    store = PostgresAnalysisAttemptStore()

    with pytest.raises(RuntimeError, match="schema permission denied"):
        await store.setup(pool)

    assert not pool.closed
    assert store._pool is None


async def test_unknown_attempt_cannot_be_charged():
    class _UnknownAttemptConnection:
        async def fetchrow(self, query, *args):
            if "UPDATE analysis_attempts" in query:
                return None
            return {
                "attempt_id": "attempt-unknown",
                "job_id": "job-1",
                "status": "unknown",
                "cost_usd": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "error": "interrupted before telemetry",
                "created_at": None,
                "charged_at": None,
                "settled_at": None,
                "updated_at": None,
            }

    store = PostgresAnalysisAttemptStore()
    store._pool = _Pool(_UnknownAttemptConnection())

    with pytest.raises(RuntimeError, match="is unknown; cannot charge"):
        await store.charge(
            "attempt-unknown",
            cost_usd=0.42,
            prompt_tokens=120,
            completion_tokens=30,
        )
