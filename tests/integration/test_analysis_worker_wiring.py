"""Fail-closed Phase 1 analysis-worker application wiring."""

from pathlib import Path

import pytest

import openloop.app as appmod
from openloop.agents import load_agent
from openloop.agents.schema import Tool
from openloop.approvals import InMemoryApprovalStore
from openloop.analysis import InMemoryAnalysisAttemptStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.config import Settings
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


def _gateway(settings):
    return appmod.build_tool_gateway(
        settings,
        {"dev-platform": _agent()},
        InMemoryApprovalStore(),
        InMemoryCheckpointStore(),
        WorkflowEngine(InMemoryWorkflowStore()),
        usage=InMemoryUsageStore(),
    )


def test_analysis_registers_without_github_when_digest_and_sealed_probe_work(monkeypatch):
    monkeypatch.setattr(DockerSandbox, "probe_sealed", lambda self, workspace_root=None: None)
    monkeypatch.setattr(appmod, "build_github_credentials", lambda settings: None)
    gateway = _gateway(Settings(
        analysis_worker_enabled=True,
        analysis_worker_sandbox_image=IMAGE,
    ))

    connector = gateway._tools["analysis"]
    assert isinstance(connector, AnalysisWorkerConnector)
    assert isinstance(connector.orchestrator.worker, BuiltinAnalysisWorker)
    assert isinstance(connector.orchestrator._attempts, InMemoryAnalysisAttemptStore)
    assert connector.orchestrator._ledger.task_kind == "analysis_worker"
    assert "github" not in gateway._tools


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
    ],
)
def test_analysis_unsafe_sandbox_configuration_fails_closed(settings, needle, caplog):
    gateway = _gateway(settings)

    assert "analysis" not in gateway._tools
    assert needle in caplog.text
