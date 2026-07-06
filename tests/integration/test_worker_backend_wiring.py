"""Integration: worker-backend selection wiring — fail-closed, never weaker.

Phase 4 policy mirrors the Phase 3 sandbox wiring: a requested backend that
can't run safely (missing extra, unusable docker, typo'd value, or — for the
agentic backend — no fail-closed spend cap) disables the coding worker loudly
instead of degrading to a different worker or a weaker boundary.
"""

from pathlib import Path
import pytest

import openloop.app as appmod
from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.config import Settings
from openloop.tools.claude_worker import ClaudeCodeCodingWorker
from openloop.tools.coding_worker import BuiltinCodingWorker
from openloop.tools.openhands_worker import (
    OpenHandsCodingWorker,
    OpenHandsUnavailable,
)
from openloop.usage import InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _settings(**kwargs):
    return Settings(
        coding_worker_enabled=True, github_token="t", **kwargs
    )


def _gateway(settings, agents=None, usage=None):
    return appmod.build_tool_gateway(
        settings,
        agents if agents is not None else {"dev-platform": load_agent(AGENT_YAML)},
        InMemoryApprovalStore(),
        InMemoryCheckpointStore(),
        WorkflowEngine(InMemoryWorkflowStore()),
        usage=usage if usage is not None else InMemoryUsageStore(),
    )


def test_default_backend_is_the_builtin_diff_worker_with_ledger_attached():
    gateway = _gateway(_settings())

    connector = gateway._tools["coding_worker"]
    orchestrator = connector.orchestrator
    assert isinstance(orchestrator.worker, BuiltinCodingWorker)
    # The ledger rides along on the default backend too: spend is recorded
    # and the invoking agent's per-task cap enforced (the example agent is
    # the attribution fallback).
    assert orchestrator._ledger is not None
    assert orchestrator._ledger.default_agent == "dev-platform"
    assert orchestrator._ledger.per_task_usd_for(None) == 0.50


@pytest.mark.parametrize("retired", ["git", "diff"])
def test_retired_backend_names_fail_closed(retired, caplog):
    # Both pre-release names of the builtin backend are dead values now: a
    # stale config must disable the worker loudly, never guess a mapping.
    with caplog.at_level("ERROR"):
        gateway = _gateway(_settings(coding_worker_backend=retired))

    assert "coding_worker" not in gateway._tools
    assert "unknown CODING_WORKER_BACKEND" in caplog.text
    assert "expected builtin|openhands|claude" in caplog.text


def test_unknown_backend_fails_closed(caplog):
    with caplog.at_level("ERROR"):
        gateway = _gateway(_settings(coding_worker_backend="opnhands"))
    assert "coding_worker" not in gateway._tools
    assert "unknown CODING_WORKER_BACKEND" in caplog.text
    assert "expected builtin|openhands|claude" in caplog.text
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
    agent = load_agent(AGENT_YAML)
    agent.spec.budget.per_task_usd = None

    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(coding_worker_backend="openhands"),
            agents={"dev-platform": agent},
        )

    assert "coding_worker" not in gateway._tools
    assert "CODING WORKER DISABLED" in caplog.text
    assert "per_task_usd" in caplog.text


def test_openhands_requires_a_cap_on_every_worker_agent(monkeypatch, caplog):
    # Phase 5 attribution enforces the *invoking* agent's cap, so the gate
    # must hold for every agent exposing the tool — one capped owner is no
    # longer enough.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    capped = load_agent(AGENT_YAML)
    uncapped = load_agent(AGENT_YAML)
    uncapped.metadata.name = "docs-bot"
    uncapped.spec.budget.per_task_usd = None

    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(coding_worker_backend="openhands"),
            agents={"dev-platform": capped, "docs-bot": uncapped},
        )

    assert "coding_worker" not in gateway._tools
    assert "CODING WORKER DISABLED" in caplog.text
    assert "docs-bot" in caplog.text  # the gate names the offender


def test_openhands_ignores_uncapped_agent_without_worker_action(monkeypatch):
    # Tool name alone is not enough: only agents that can invoke
    # coding_worker.pr:write need a cap and can become the fallback owner.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    capped = load_agent(AGENT_YAML)
    observer = load_agent(AGENT_YAML)
    observer.metadata.name = "docs-bot"
    observer.spec.budget.per_task_usd = None
    for tool in observer.spec.tools:
        if tool.name == "coding_worker":
            tool.permissions = []

    gateway = _gateway(
        _settings(coding_worker_backend="openhands"),
        agents={"docs-bot": observer, "dev-platform": capped},
    )

    connector = gateway._tools["coding_worker"]
    assert connector.orchestrator._ledger.default_agent == "dev-platform"


def test_openhands_without_usage_store_fails_closed(monkeypatch, caplog):
    # A deploy that passes no usage store cannot build a ledger — the agentic
    # backend must not run uncapped and unrecorded.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = appmod.build_tool_gateway(
            _settings(coding_worker_backend="openhands"),
            {"dev-platform": load_agent(AGENT_YAML)},
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


def test_claude_registers_in_host_mode(monkeypatch):
    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", lambda self: None)
    gateway = _gateway(_settings(coding_worker_backend="claude"))

    worker = gateway._tools["coding_worker"].orchestrator.worker
    assert isinstance(worker, ClaudeCodeCodingWorker)
    # --max-turns + the deadline are threaded from settings as the fail-closed
    # bound; the deadline default (600) is passed through, not disabled.
    assert worker.max_turns == 100
    assert worker.deadline_seconds == 600.0


def test_claude_docker_mode_fails_closed(monkeypatch, caplog):
    # Docker isolation for the claude backend is not implemented: requesting it
    # must disable the worker loudly, never silently run on the host.
    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(coding_worker_backend="claude", coding_worker_sandbox="docker")
        )
    assert "coding_worker" not in gateway._tools
    assert "supports only CODING_WORKER_SANDBOX=host" in caplog.text
    assert "CODING WORKER DISABLED" in caplog.text


def test_claude_registers_without_a_per_task_dollar_cap(monkeypatch):
    # Unlike openhands, the claude backend's fail-closed bound is turns +
    # deadline (the subscription dollar signal is unreliable), so it does NOT
    # require a per-task dollar cap to register. The ledger still rides along.
    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", lambda self: None)
    uncapped = load_agent(AGENT_YAML)
    uncapped.spec.budget.per_task_usd = None

    gateway = _gateway(
        _settings(coding_worker_backend="claude"),
        agents={"dev-platform": uncapped},
    )

    assert "coding_worker" in gateway._tools
    assert gateway._tools["coding_worker"].orchestrator._ledger is not None


def test_claude_probe_failure_fails_closed(monkeypatch, caplog):
    def boom(self):
        from openloop.tools.claude_worker import ClaudeCodeUnavailable

        raise ClaudeCodeUnavailable("claude CLI not found")

    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", boom)
    with caplog.at_level("ERROR"):
        gateway = _gateway(_settings(coding_worker_backend="claude"))
    assert "coding_worker" not in gateway._tools
    assert "claude backend probe failed" in caplog.text
    assert "CODING WORKER DISABLED" in caplog.text
