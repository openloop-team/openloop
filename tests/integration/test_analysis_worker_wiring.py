"""Fail-closed analysis-worker application wiring (Phases 1 + 3)."""

from pathlib import Path

import pytest

import openloop.app as appmod
from openloop.agents import load_agent
from openloop.agents.schema import Tool
from openloop.approvals import InMemoryApprovalStore
from openloop.analysis import InMemoryAnalysisAttemptStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.config import DEFAULT_ANALYSIS_SANDBOX_IMAGE, Settings
from openloop.sandbox import DockerSandbox
from openloop.tools.analysis_worker import AnalysisWorkerConnector, BuiltinAnalysisWorker
from openloop.usage import InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"
IMAGE = "registry.example/openloop-analysis@sha256:" + "a" * 64


def _agent():
    agent = load_agent(AGENT_YAML)
    agent.spec.tools.append(Tool(name="analysis", type="native", permissions=["report:write"]))
    agent.spec.approvals.require_for.append("analysis.report:write")
    return agent


def _gateway(settings, engine=None, agent=None):
    return appmod.build_tool_gateway(
        settings,
        {"dev-platform": agent or _agent()},
        InMemoryApprovalStore(),
        InMemoryCheckpointStore(),
        engine or WorkflowEngine(InMemoryWorkflowStore()),
        usage=InMemoryUsageStore(),
    )


def test_analysis_registers_without_github_when_digest_and_sealed_probe_work(monkeypatch):
    monkeypatch.setattr(DockerSandbox, "probe_sealed", lambda self, workspace_root=None: None)
    monkeypatch.setattr(appmod, "build_github_credentials", lambda settings: None)
    engine = WorkflowEngine(InMemoryWorkflowStore())
    gateway = _gateway(Settings(
        analysis_worker_enabled=True,
        analysis_worker_sandbox_image=IMAGE,
    ), engine=engine)

    connector = gateway._tools["analysis"]
    assert isinstance(connector, AnalysisWorkerConnector)
    assert isinstance(connector.orchestrator.worker, BuiltinAnalysisWorker)
    # Iterative is the product default, and the hard cap gate is opt-in —
    # spend stays bounded by max_iterations, capped feedback growth, human
    # approval per run, and any caps the invoking agent does carry.
    assert connector.orchestrator.worker.strategy == "iterative"
    assert isinstance(connector.orchestrator._attempts, InMemoryAnalysisAttemptStore)
    assert connector.orchestrator._ledger.task_kind == "analysis_worker"
    assert connector.orchestrator._ledger.require_per_task_cap is False
    assert "github" not in gateway._tools
    # Phase 2: the connector declares the durable workflow, and the engine has
    # it registered — an approval parks/wakes the sealed run instead of a
    # direct execute().
    assert connector.workflow == "analysis_worker"
    assert "analysis_worker" in engine.workflows


def test_analysis_uses_an_immutable_python_smoke_image_by_default(monkeypatch):
    monkeypatch.setattr(
        DockerSandbox, "probe_sealed", lambda self, workspace_root=None: None
    )
    monkeypatch.setattr(appmod, "build_github_credentials", lambda settings: None)

    gateway = _gateway(Settings(analysis_worker_enabled=True))

    worker = gateway._tools["analysis"].orchestrator.worker
    assert worker.sandbox.image == DEFAULT_ANALYSIS_SANDBOX_IMAGE
    assert DEFAULT_ANALYSIS_SANDBOX_IMAGE.startswith("python@sha256:")


@pytest.mark.parametrize(
    ("settings", "needle"),
    [
        (Settings(analysis_worker_enabled=True, analysis_worker_sandbox="host"), "host is forbidden"),
        (Settings(analysis_worker_enabled=True, analysis_worker_sandbox_image="openloop:latest"), "digest-pinned"),
        (
            Settings(
                analysis_worker_enabled=True,
                analysis_worker_sandbox_image=IMAGE,
                analysis_worker_sandbox_network="egress-proxy",
            ),
            "network=none",
        ),
        (
            Settings(
                analysis_worker_enabled=True,
                analysis_worker_sandbox_image=IMAGE,
                analysis_worker_strategy="both",
            ),
            "ANALYSIS_WORKER_STRATEGY",
        ),
    ],
)
def test_analysis_unsafe_sandbox_configuration_fails_closed(settings, needle, caplog):
    gateway = _gateway(settings)

    assert "analysis" not in gateway._tools
    assert needle in caplog.text


def test_uncapped_agents_register_by_default_without_the_cap_gate(monkeypatch):
    # The hard cap requirement is an opt-in posture: with the default config,
    # capless agents may run iterative analysis (bounded by max_iterations,
    # capped feedback growth, and per-run human approval).
    monkeypatch.setattr(DockerSandbox, "probe_sealed", lambda self, workspace_root=None: None)
    monkeypatch.setattr(appmod, "build_github_credentials", lambda settings: None)
    agent = _agent()
    agent.spec.budget.per_task_usd = None

    gateway = _gateway(
        Settings(
            analysis_worker_enabled=True,
            analysis_worker_sandbox_image=IMAGE,
        ),
        agent=agent,
    )

    connector = gateway._tools["analysis"]
    assert connector.orchestrator.worker.strategy == "iterative"
    assert connector.orchestrator._ledger.require_per_task_cap is False


def test_opt_in_cap_requirement_without_caps_fails_closed(monkeypatch, caplog):
    # ANALYSIS_WORKER_REQUIRE_PER_TASK_CAP is the openhands-style boot gate:
    # once the operator demands caps, a capless exposing agent disables the
    # tool rather than running uncapped.
    monkeypatch.setattr(DockerSandbox, "probe_sealed", lambda self, workspace_root=None: None)
    monkeypatch.setattr(appmod, "build_github_credentials", lambda settings: None)
    agent = _agent()
    agent.spec.budget.per_task_usd = None

    gateway = _gateway(
        Settings(
            analysis_worker_enabled=True,
            analysis_worker_sandbox_image=IMAGE,
            analysis_worker_require_per_task_cap=True,
        ),
        agent=agent,
    )

    assert "analysis" not in gateway._tools
    assert "ANALYSIS_WORKER_REQUIRE_PER_TASK_CAP" in caplog.text
    assert "dev-platform" in caplog.text


def test_opt_in_cap_requirement_with_caps_registers_and_hardens_the_ledger(monkeypatch):
    monkeypatch.setattr(DockerSandbox, "probe_sealed", lambda self, workspace_root=None: None)
    monkeypatch.setattr(appmod, "build_github_credentials", lambda settings: None)

    gateway = _gateway(Settings(
        analysis_worker_enabled=True,
        analysis_worker_sandbox_image=IMAGE,
        analysis_worker_require_per_task_cap=True,
        analysis_worker_max_iterations=6,
    ))

    connector = gateway._tools["analysis"]
    worker = connector.orchestrator.worker
    assert isinstance(worker, BuiltinAnalysisWorker)
    assert worker.strategy == "iterative"
    assert worker.max_iterations == 6
    # Stale approved jobs stay fail-closed if caps drift after approval.
    assert connector.orchestrator._ledger.require_per_task_cap is True
