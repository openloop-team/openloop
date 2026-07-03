"""Integration: worker-backend selection wiring — fail-closed, never weaker.

Phase 4 policy mirrors the Phase 3 sandbox wiring: a requested backend that
can't run safely (missing extra, unusable docker, typo'd value, or — for the
agentic backend — no fail-closed spend cap) disables the coding worker loudly
instead of degrading to a different worker or a weaker boundary.
"""

import openloop.app as appmod
from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.config import Settings
from openloop.tools.coding_worker import GitCodingWorker
from openloop.tools.openhands_worker import (
    OpenHandsCodingWorker,
    OpenHandsUnavailable,
)
from openloop.usage import InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.testing import EXAMPLE_AGENT


def _settings(**kwargs):
    return Settings(
        coding_worker_enabled=True, github_token="t", **kwargs
    )


def _gateway(settings, agents=None, usage=None):
    return appmod.build_tool_gateway(
        settings,
        agents if agents is not None else {"dev-platform": load_agent(EXAMPLE_AGENT)},
        InMemoryApprovalStore(),
        InMemoryCheckpointStore(),
        WorkflowEngine(InMemoryWorkflowStore()),
        usage=usage if usage is not None else InMemoryUsageStore(),
    )


def test_default_backend_is_git_with_ledger_attached():
    gateway = _gateway(_settings())

    connector = gateway._tools["coding_worker"]
    orchestrator = connector.orchestrator
    assert isinstance(orchestrator.worker, GitCodingWorker)
    # The Phase 4 ledger rides along on the default backend too: spend is
    # recorded and the example agent's per-task cap enforced.
    assert orchestrator._ledger is not None
    assert orchestrator._ledger.per_task_usd == 0.50
    assert orchestrator._ledger.agent == "dev-platform"


def test_unknown_backend_fails_closed(caplog):
    with caplog.at_level("ERROR"):
        gateway = _gateway(_settings(coding_worker_backend="opnhands"))
    assert "coding_worker" not in gateway._tools
    assert "unknown CODING_WORKER_BACKEND" in caplog.text
    assert "CODING WORKER DISABLED" in caplog.text


def test_openhands_registers_with_cap_and_probe(monkeypatch):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    gateway = _gateway(
        _settings(
            coding_worker_backend="openhands",
            coding_worker_sandbox="docker",
            coding_worker_openhands_network="egress-proxy",
            anthropic_api_key="sk-test",
        )
    )

    worker = gateway._tools["coding_worker"].orchestrator.worker
    assert isinstance(worker, OpenHandsCodingWorker)
    # CODING_WORKER_SANDBOX=docker maps to the mounted DockerWorkspace mode.
    assert worker.docker is True
    assert worker.network == "egress-proxy"
    # The provider key for the worker model is threaded from settings.
    assert worker._api_key == "sk-test"


def test_openhands_without_per_task_cap_fails_closed(monkeypatch, caplog):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    agent = load_agent(EXAMPLE_AGENT)
    agent.spec.budget.per_task_usd = None

    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(coding_worker_backend="openhands"),
            agents={"dev-platform": agent},
        )

    assert "coding_worker" not in gateway._tools
    assert "CODING WORKER DISABLED" in caplog.text
    assert "per_task_usd" in caplog.text


def test_openhands_without_usage_store_fails_closed(monkeypatch, caplog):
    # A deploy that passes no usage store cannot build a ledger — the agentic
    # backend must not run uncapped and unrecorded.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = appmod.build_tool_gateway(
            _settings(coding_worker_backend="openhands"),
            {"dev-platform": load_agent(EXAMPLE_AGENT)},
            InMemoryApprovalStore(),
            InMemoryCheckpointStore(),
            WorkflowEngine(InMemoryWorkflowStore()),
            usage=None,
        )
    assert "coding_worker" not in gateway._tools
    assert "CODING WORKER DISABLED" in caplog.text


def test_openhands_probe_failure_fails_closed(monkeypatch, caplog):
    def boom(self):
        raise OpenHandsUnavailable("openhands extra not installed")

    monkeypatch.setattr(OpenHandsCodingWorker, "probe", boom)
    with caplog.at_level("ERROR"):
        gateway = _gateway(_settings(coding_worker_backend="openhands"))
    assert "coding_worker" not in gateway._tools
    assert "openhands backend probe failed" in caplog.text
    assert "CODING WORKER DISABLED" in caplog.text


def test_openhands_with_sandbox_typo_fails_closed(monkeypatch, caplog):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands", coding_worker_sandbox="dokcer"
            )
        )
    assert "coding_worker" not in gateway._tools
    assert "unknown CODING_WORKER_SANDBOX" in caplog.text
