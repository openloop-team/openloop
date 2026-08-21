"""Regression coverage for settle-before-wire application composition."""

import base64
import importlib
import os
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.broker_config import BrokerServiceConfig
from openloop.broker_runtime.memory import InMemoryRuntimeDriver
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.coordination import InProcessLock, PostgresLock
from openloop.memory import InMemoryStore
from openloop.postgres import BorrowedPostgresStore
from openloop.sessions import (
    InMemorySurfaceSessionStore,
    InMemoryThreadRecordStore,
)
from openloop.tools import ToolGateway
from openloop.tools.openhands_broker_workspace import BrokerWorkspaceAdapter
from openloop.wiring import builders, compose
from openloop.wiring.broker import _derive_receipt_key, build_broker_service
from openloop.workflows import InMemoryWorkflowStore
from tests.support.settings import (
    IsolatedBrokerSettings,
)
from tests.support.settings import (
    IsolatedSettings as Settings,
)

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"
composemod = importlib.import_module("openloop.wiring.compose")


def test_effective_storage_mode_precedence_and_legacy_mapping():
    assert Settings().effective_storage_mode == "memory"
    assert Settings(memory_backend="postgres").effective_storage_mode == "auto"
    assert (
        Settings(
            storage_mode="memory",
            memory_backend="postgres",
        ).effective_storage_mode
        == "memory"
    )
    assert isinstance(
        builders.build_lock(Settings(storage_mode="postgres", lock_backend="auto")),
        PostgresLock,
    )
    assert isinstance(
        builders.build_lock(Settings(storage_mode="memory", lock_backend="auto")),
        InProcessLock,
    )


async def test_precomposed_gateway_override_is_rejected():
    with pytest.raises(TypeError, match="tools_factory"):
        async with compose(
            Settings(storage_mode="memory"),
            {},
            overrides={"tools": ToolGateway()},
        ):
            pass


async def test_broker_handle_is_a_recognized_override():
    # broker_handle is a leaf override; supplying it skips build_broker and
    # threads through to build_tool_gateway without an "unknown override" error.
    async with compose(
        Settings(
            storage_mode="memory",
            coding_worker_openhands_broker_enabled=True,
            broker_mode="external",
        ),
        {},
        overrides={"broker_handle": object()},
    ) as ctx:
        assert ctx is not None


async def test_app_composition_refuses_embedded_broker_mode(monkeypatch, caplog):
    calls = 0

    async def forbidden_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(composemod, "build_broker", forbidden_build)
    with caplog.at_level("ERROR"):
        async with compose(
            Settings(
                storage_mode="memory",
                coding_worker_openhands_broker_enabled=True,
                broker_mode="embedded",
            ),
            {},
        ):
            pass

    assert calls == 0
    assert "BROKER_MODE=external" in caplog.text


def _broker_root(seed: int) -> str:
    return base64.b64encode(bytes([seed]) * 32).decode()


async def test_external_broker_mode_wires_adapter_without_app_reconciler(
    tmp_path, short_socket_root, monkeypatch
):
    identity_seed = bytes(range(1, 33))
    identity_private = Ed25519PrivateKey.from_private_bytes(identity_seed)
    receipt_root = bytes([4]) * 32
    receipt_public = (
        _derive_receipt_key(receipt_root, "broker-receipt", "receipt-key-v1")
        .public_key()
        .public_bytes_raw()
    )

    broker_state = tmp_path / "broker-state"
    broker_runtime = tmp_path / "broker-runtime"
    app_ingress = tmp_path / "app-ingress"
    app_receipts = tmp_path / "app-receipts"
    for path in (broker_state, broker_runtime):
        path.mkdir()
        path.chmod(0o700)
    for path in (app_ingress, app_receipts):
        path.mkdir()
        os.chown(path, -1, os.getgid())
        path.chmod(0o2750)

    settings = Settings(
        storage_mode="memory",
        embeddings_enabled=False,
        recovery_interval_seconds=0,
        coding_worker_enabled=True,
        coding_worker_backend="openhands",
        coding_worker_sandbox="docker",
        coding_worker_openhands_broker_enabled=True,
        coding_worker_openhands_state_dir=str(tmp_path / "openhands-state"),
        coding_worker_openhands_state_master_key=_broker_root(9),
        github_token="test-token",
        broker_mode="external",
        broker_control_socket_dir=str(short_socket_root),
        broker_ingress_root=str(app_ingress),
        broker_checkpoint_receipt_root=str(app_receipts),
        broker_shared_data_gid=os.getgid(),
        broker_identity_private_key=SecretStr(base64.b64encode(identity_seed).decode()),
        broker_receipt_roots={
            "receipt-key-v1": base64.b64encode(receipt_root).decode()
        },
    )
    broker_settings = IsolatedBrokerSettings(
        broker_control_socket_dir=str(short_socket_root),
        broker_state_root=str(broker_state),
        broker_runtime_root=str(broker_runtime),
        broker_ingress_root=str(app_ingress),
        broker_checkpoint_receipt_root=str(app_receipts),
        broker_shared_data_gid=os.getgid(),
        broker_expected_app_uid=os.getuid(),
        broker_identity_public_keys={
            "identity-v1": base64.b64encode(
                identity_private.public_key().public_bytes_raw()
            ).decode()
        },
        broker_capability_roots={"cap-key-v1": _broker_root(1)},
        broker_runtime_roots={"runtime-key-v1": _broker_root(2)},
        broker_receipt_public_keys={
            "receipt-key-v1": base64.b64encode(receipt_public).decode()
        },
        broker_execution_lease_seconds=300,
    )
    agent = load_agent(AGENT_YAML)

    monkeypatch.setattr(builders.OpenHandsCodingWorker, "probe", lambda self: None)
    recovery_reconcilers = []
    real_run_recovery_pass = builders.run_recovery_pass

    async def capture_reconciler(*args, **kwargs):
        recovery_reconcilers.append(kwargs.get("broker_reconciler"))
        return await real_run_recovery_pass(*args, **kwargs)

    monkeypatch.setattr(builders, "run_recovery_pass", capture_reconciler)

    async with AsyncExitStack() as broker_stack:
        service = await build_broker_service(
            BrokerServiceConfig.from_broker_settings(broker_settings),
            broker_stack,
            runtime_driver=InMemoryRuntimeDriver(),
        )
        await service.bind()
        broker_stack.push_async_callback(service.server.stop)

        async with compose(settings, {agent.metadata.name: agent}) as ctx:
            worker = ctx.tools._tools["workspace_task"].orchestrator.worker
            assert isinstance(worker._docker_adapter, BrokerWorkspaceAdapter)

    assert recovery_reconcilers == [None]


async def test_auto_fallback_settles_every_capture_before_wiring(monkeypatch):
    async def unavailable_pool(*args, **kwargs):
        raise RuntimeError("postgres unavailable")

    agent = load_agent(AGENT_YAML)
    settings = Settings(
        storage_mode="auto",
        lock_backend="memory",
        embeddings_enabled=False,
        recovery_interval_seconds=0,
        coding_worker_enabled=True,
        coding_worker_warm_context=False,
        github_token="test-token",
        slack_bot_token="xoxb-test",
    )

    async with compose(
        settings,
        {agent.metadata.name: agent},
        overrides={"pool_factory": unavailable_pool},
    ) as ctx:
        runtime = ctx.agents.slack_runtime()
        assert runtime is not None
        assert runtime.memory is ctx.memory
        assert runtime.usage is ctx.usage
        assert runtime.tools is ctx.tools
        assert runtime.engine is ctx.engine
        assert ctx.engine.store is ctx.workflows
        assert ctx.tools.approvals is ctx.approvals

        coding = ctx.tools._tools["workspace_task"]
        assert coding.checkpoints is ctx.checkpoints
        assert isinstance(coding.checkpoints, InMemoryCheckpointStore)
        assert coding.orchestrator._ledger.usage is ctx.usage

        runner = ctx.session_runner
        assert runner is not None
        assert runner.sessions is ctx.sessions
        assert runner.threads is ctx.threads


class _ClosingPool:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class _DisposingEngine:
    """Stands in for an AsyncEngine — released through dispose(), not close()."""

    def __init__(self):
        self.dispose_calls = 0

    async def dispose(self):
        self.dispose_calls += 1


class _FailingPostgresStore(BorrowedPostgresStore):
    async def setup(self, pool):
        raise RuntimeError("schema setup failed")


async def test_postgres_store_setup_failure_unwinds_pool_once():
    pool = _ClosingPool()
    engine = _DisposingEngine()

    async def pool_factory(*args, **kwargs):
        return pool

    async def engine_factory(*args, **kwargs):
        return engine

    with pytest.raises(RuntimeError, match="schema setup failed"):
        async with compose(
            Settings(
                storage_mode="postgres",
                embeddings_enabled=False,
                recovery_interval_seconds=0,
            ),
            {},
            overrides={
                "pool_factory": pool_factory,
                "engine_factory": engine_factory,
                "memory": _FailingPostgresStore(),
            },
        ):
            pass

    assert pool.close_calls == 1
    assert engine.dispose_calls == 1


def test_create_app_composes_with_the_settings_it_is_given(tmp_path):
    """The composition root takes its configuration rather than loading its
    own. Pointing it at an empty agents directory must yield no agents, where
    ambient configuration would have loaded the repo's own."""
    from fastapi.testclient import TestClient

    from openloop import app as appmod

    created = appmod.create_app(
        settings=Settings(agents_dir=str(tmp_path), storage_mode="memory")
    )

    with TestClient(created):
        assert list(created.state.ctx.agents.loaded) == []


async def test_engine_backed_stores_bind_to_the_engine_not_the_pool():
    """During the ADR 0007 transition both binds exist; a store gets its own."""
    from openloop.db import BorrowedEngineStore

    bound: dict[str, object] = {}

    class _EngineStore(BorrowedEngineStore):
        async def setup(self, engine):
            bound["engine"] = engine

    class _Engine:
        async def dispose(self):
            bound["disposed"] = True

    async def engine_factory(*args, **kwargs):
        return _Engine()

    async def pool_factory(*args, **kwargs):
        return _ClosingPool()

    settings = Settings(
        storage_mode="postgres",
        embeddings_enabled=False,
        recovery_interval_seconds=0,
    )
    async with compose(
        settings,
        agents={},
        overrides={
            "pool_factory": pool_factory,
            "engine_factory": engine_factory,
            "usage": _EngineStore(),
            # Only `usage` is durable here: the stub pool has no acquire(), so
            # every other store has to stay process-local.
            "memory": InMemoryStore(),
            "approvals": InMemoryApprovalStore(),
            "checkpoints": InMemoryCheckpointStore(),
            "workflows": InMemoryWorkflowStore(),
            "sessions": InMemorySurfaceSessionStore(),
            "threads": InMemoryThreadRecordStore(),
        },
    ):
        pass

    assert isinstance(bound["engine"], _Engine)
    assert bound.get("disposed") is True
