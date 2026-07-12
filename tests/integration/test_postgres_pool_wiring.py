"""Integration coverage for the application-owned PostgreSQL pool."""

import pytest
from fastapi.testclient import TestClient

from openloop import app as appmod
from openloop.analysis import (
    InMemoryAnalysisAttemptStore,
    InMemoryArtifactStore,
    InMemoryInputStore,
    InMemoryUploadStore,
)
from openloop.config import Settings
from openloop.memory import InMemoryStore
from openloop.sessions import InMemorySurfaceSessionStore, InMemoryThreadRecordStore
from openloop.usage import InMemoryUsageStore


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Connection:
    async def execute(self, query, *args):
        if query.lstrip().startswith("UPDATE"):
            return "UPDATE 0"
        return "OK"

    async def fetch(self, query, *args):
        return []

    async def fetchrow(self, query, *args):
        return None

    async def fetchval(self, query, *args):
        return 0


class _Pool:
    def __init__(self):
        self.connection = _Connection()
        self.close_calls = 0

    def acquire(self):
        return _Acquire(self.connection)

    async def close(self):
        self.close_calls += 1


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        memory_backend="postgres",
        lock_backend="memory",
        agents_dir=str(tmp_path),
        embeddings_enabled=False,
        recovery_interval_seconds=0,
        postgres_pool_min_size=2,
        postgres_pool_max_size=7,
    )


def test_lifespan_creates_and_closes_one_shared_pool(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    pool = _Pool()
    create_calls = []

    async def create_pool(dsn, *, min_size, max_size):
        create_calls.append((dsn, min_size, max_size))
        return pool

    monkeypatch.setattr(appmod, "get_settings", lambda: settings)
    monkeypatch.setattr(appmod, "create_pool", create_pool)

    app = appmod.create_app()
    with pytest.raises(RuntimeError, match="exceptional shutdown"):
        with TestClient(app):
            assert create_calls == [(settings.database_url, 2, 7)]
            assert app.state.postgres_pool is pool
            ordinary_stores = [
                app.state.memory,
                app.state.usage,
                app.state.sessions,
                app.state.threads,
                app.state.analysis_inputs,
                app.state.analysis_artifacts,
                app.state.analysis_attempts,
                app.state.analysis_uploads,
                app.state.tools.approvals,
                app.state.tools.engine.store,
            ]
            assert all(store._pool is pool for store in ordinary_stores)
            assert pool.close_calls == 0
            raise RuntimeError("exceptional shutdown")

    assert pool.close_calls == 1
    assert app.state.postgres_pool is None
    assert all(store._pool is None for store in ordinary_stores)


def test_pool_creation_failure_uses_fallbacks_without_store_pool_attempts(
    monkeypatch, tmp_path
):
    settings = _settings(tmp_path)
    calls = 0

    async def create_pool(dsn, *, min_size, max_size):
        nonlocal calls
        calls += 1
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(appmod, "get_settings", lambda: settings)
    monkeypatch.setattr(appmod, "create_pool", create_pool)

    app = appmod.create_app()
    with TestClient(app):
        assert calls == 1
        assert app.state.postgres_pool is None
        assert isinstance(app.state.memory, InMemoryStore)
        assert isinstance(app.state.usage, InMemoryUsageStore)
        assert isinstance(app.state.sessions, InMemorySurfaceSessionStore)
        assert isinstance(app.state.threads, InMemoryThreadRecordStore)
        assert isinstance(app.state.analysis_inputs, InMemoryInputStore)
        assert isinstance(app.state.analysis_artifacts, InMemoryArtifactStore)
        assert isinstance(
            app.state.analysis_attempts, InMemoryAnalysisAttemptStore
        )
        assert isinstance(app.state.analysis_uploads, InMemoryUploadStore)
