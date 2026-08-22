"""Integration coverage for the application-owned PostgreSQL engine."""

import importlib

import pytest
from fastapi.testclient import TestClient

from openloop import app as appmod
from openloop.approvals import InMemoryApprovalStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.db import BorrowedEngineStore
from openloop.memory import InMemoryStore
from openloop.sessions import InMemorySurfaceSessionStore, InMemoryThreadRecordStore
from openloop.usage import InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore
from tests.support.settings import IsolatedSettings as Settings

composemod = importlib.import_module("openloop.wiring.compose")


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _EngineResult:
    """The little of a SQLAlchemy Result that a store's setup() path reads."""

    rowcount = 0

    def first(self):
        return None

    def mappings(self):
        return self

    def all(self):
        return []


class _EngineConnection:
    """Accepts any statement a converted store's setup() issues."""

    async def execute(self, statement, parameters=None):
        return _EngineResult()

    async def scalar(self, statement, parameters=None):
        return None


class _Engine:
    """Stands in for an AsyncEngine — begin()/connect(), released by dispose()."""

    def __init__(self):
        self.connection = _EngineConnection()
        self.dispose_calls = 0

    def begin(self):
        return _Acquire(self.connection)

    def connect(self):
        return _Acquire(self.connection)

    async def dispose(self):
        self.dispose_calls += 1


def _handle_of(store):
    """The engine a settled store is bound to."""
    assert isinstance(store, BorrowedEngineStore)
    return store._engine


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "memory_backend": "postgres",
        "lock_backend": "memory",
        "agents_dir": str(tmp_path),
        "embeddings_enabled": False,
        "recovery_interval_seconds": 0,
        "postgres_pool_min_size": 2,
        "postgres_pool_max_size": 7,
    }
    values.update(overrides)
    return Settings(**values)


def test_lifespan_creates_and_disposes_one_shared_engine(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    engine = _Engine()
    engine_calls = []

    async def create_engine(dsn, *, min_size, max_size):
        engine_calls.append((dsn, min_size, max_size))
        return engine

    monkeypatch.setattr(composemod, "create_engine", create_engine)

    app = appmod.create_app(settings=settings)
    with pytest.raises(RuntimeError, match="exceptional shutdown"):
        with TestClient(app):
            assert engine_calls == [(settings.database_url, 2, 7)]
            ctx = app.state.ctx
            assert ctx.postgres_engine is engine
            ordinary_stores = [
                ctx.memory,
                ctx.usage,
                ctx.approvals,
                ctx.checkpoints,
                ctx.sessions,
                ctx.threads,
                ctx.engine.store,
            ]
            for store in ordinary_stores:
                assert _handle_of(store) is engine
            assert engine.dispose_calls == 0
            raise RuntimeError("exceptional shutdown")

    assert engine.dispose_calls == 1
    assert all(_handle_of(store) is None for store in ordinary_stores)


def test_engine_creation_failure_uses_fallbacks_without_store_setup_attempts(
    monkeypatch, tmp_path
):
    settings = _settings(tmp_path)
    calls = 0

    async def create_engine(dsn, *, min_size, max_size):
        nonlocal calls
        calls += 1
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(composemod, "create_engine", create_engine)

    app = appmod.create_app(settings=settings)
    with TestClient(app):
        assert calls == 1
        ctx = app.state.ctx
        assert ctx.postgres_engine is None
        assert isinstance(ctx.memory, InMemoryStore)
        assert isinstance(ctx.usage, InMemoryUsageStore)
        assert isinstance(ctx.approvals, InMemoryApprovalStore)
        assert isinstance(ctx.checkpoints, InMemoryCheckpointStore)
        assert isinstance(ctx.workflows, InMemoryWorkflowStore)
        assert isinstance(ctx.sessions, InMemorySurfaceSessionStore)
        assert isinstance(ctx.threads, InMemoryThreadRecordStore)


def test_lifespan_passes_the_mounted_postgres_password(monkeypatch, tmp_path):
    settings = _settings(tmp_path, postgres_password="mounted-db-secret")
    engine_calls = []

    async def create_engine(dsn, *, min_size, max_size, password):
        engine_calls.append((dsn, min_size, max_size, password))
        return _Engine()

    monkeypatch.setattr(composemod, "create_engine", create_engine)

    app = appmod.create_app(settings=settings)
    with TestClient(app):
        assert engine_calls == [
            (
                settings.database_url,
                2,
                7,
                "mounted-db-secret",
            )
        ]
