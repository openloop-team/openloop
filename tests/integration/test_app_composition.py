"""Regression coverage for settle-before-wire application composition."""

from pathlib import Path

import pytest

from openloop.agents import load_agent
from openloop.agents.schema import Tool
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.config import Settings
from openloop.coordination import InProcessLock, PostgresLock
from openloop.postgres import BorrowedPostgresStore
from openloop.sandbox import DockerSandbox
from openloop.tools import ToolGateway
from openloop.wiring import compose
from openloop.wiring import builders

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def test_effective_storage_mode_precedence_and_legacy_mapping():
    assert Settings(_env_file=None).effective_storage_mode == "memory"
    assert (
        Settings(_env_file=None, memory_backend="postgres").effective_storage_mode
        == "auto"
    )
    assert (
        Settings(
            _env_file=None,
            storage_mode="memory",
            memory_backend="postgres",
        ).effective_storage_mode
        == "memory"
    )
    assert isinstance(
        builders.build_lock(
            Settings(_env_file=None, storage_mode="postgres", lock_backend="auto")
        ),
        PostgresLock,
    )
    assert isinstance(
        builders.build_lock(
            Settings(_env_file=None, storage_mode="memory", lock_backend="auto")
        ),
        InProcessLock,
    )


async def test_precomposed_gateway_override_is_rejected():
    with pytest.raises(TypeError, match="tools_factory"):
        async with compose(
            Settings(_env_file=None, storage_mode="memory"),
            {},
            overrides={"tools": ToolGateway()},
        ):
            pass


async def test_auto_fallback_settles_every_capture_before_wiring(monkeypatch):
    async def unavailable_pool(*args, **kwargs):
        raise RuntimeError("postgres unavailable")

    monkeypatch.setattr(
        DockerSandbox,
        "probe_sealed",
        lambda self, workspace_root=None: None,
    )
    agent = load_agent(AGENT_YAML)
    agent.spec.tools.append(
        Tool(name="analysis", type="native", permissions=["report:write"])
    )
    agent.spec.approvals.require_for.append("analysis.report:write")
    settings = Settings(
        _env_file=None,
        storage_mode="auto",
        lock_backend="memory",
        embeddings_enabled=False,
        recovery_interval_seconds=0,
        coding_worker_enabled=True,
        coding_worker_warm_context=False,
        github_token="test-token",
        analysis_worker_enabled=True,
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

        coding = ctx.tools._tools["coding_worker"]
        assert coding.checkpoints is ctx.checkpoints
        assert isinstance(coding.checkpoints, InMemoryCheckpointStore)
        assert coding.orchestrator._ledger.usage is ctx.usage

        analysis = ctx.tools._tools["analysis"]
        orchestrator = analysis.orchestrator
        assert orchestrator._ledger.usage is ctx.usage
        assert orchestrator._artifacts is ctx.analysis_artifacts
        assert orchestrator._attempts is ctx.analysis_attempts
        assert orchestrator._provisioners["staged"].inputs is ctx.analysis_inputs
        assert orchestrator._provisioners["upload"].uploads is ctx.analysis_uploads
        assert analysis.uploads is ctx.analysis_uploads

        runner = ctx.session_runner
        assert runner is not None
        assert runner.sessions is ctx.sessions
        assert runner.threads is ctx.threads
        assert runner.artifacts is ctx.analysis_artifacts
        assert runner.uploads is ctx.analysis_uploads


class _ClosingPool:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class _FailingPostgresStore(BorrowedPostgresStore):
    async def setup(self, pool):
        raise RuntimeError("schema setup failed")


async def test_postgres_store_setup_failure_unwinds_pool_once():
    pool = _ClosingPool()

    async def pool_factory(*args, **kwargs):
        return pool

    with pytest.raises(RuntimeError, match="schema setup failed"):
        async with compose(
            Settings(
                _env_file=None,
                storage_mode="postgres",
                embeddings_enabled=False,
                recovery_interval_seconds=0,
            ),
            {},
            overrides={
                "pool_factory": pool_factory,
                "memory": _FailingPostgresStore(),
            },
        ):
            pass

    assert pool.close_calls == 1
