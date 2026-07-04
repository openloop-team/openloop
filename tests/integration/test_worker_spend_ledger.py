"""Integration: the spend ledger fails closed on BOTH durable paths.

Per the roadmap's leading invariant there is no single "coding-worker path":
the checkpoint-connector fallback and the workflow-backed path must both run
attempts through the one shared :class:`GitWorkspaceOrchestrator`, so wiring
the ledger there caps them both — including an engine-less deploy. These tests
drive each path end to end with a real orchestrator (git subprocesses faked)
and assert the over-budget attempt is recorded, failed, and never becomes a
push or a PR — and (Phase 5) that spend follows the *invoking* agent through
the gateway's approval hop on both paths.
"""

import pytest

from openloop.agents import load_agent
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.credentials import EnvCredentialResolver
from openloop.tools import ToolGateway
from openloop.tools.coding_worker import (
    CodingWorkerConnector,
    GitWorkspaceOrchestrator,
)
from openloop.usage import (
    InMemoryUsageStore,
    UsageRecord,
    WorkerSpendLedger,
)
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.coding_worker import build_coding_worker_workflow
from openloop.testing import EXAMPLE_AGENT, FakeCodingWorker, FakeGitHub


def _agent(per_task_usd=0.50, monthly_usd=None):
    agent = load_agent(EXAMPLE_AGENT)  # name=dev-platform, workspace=acme
    agent.spec.budget.per_task_usd = per_task_usd
    agent.spec.budget.monthly_usd = monthly_usd
    return agent


def _orchestrator(monkeypatch, usage, *, cost_usd, agent=None):
    agent = agent or _agent()
    ledger = WorkerSpendLedger(
        usage=usage,
        model="m",
        agents={agent.metadata.name: agent},
        default_agent=agent.metadata.name,
    )
    orch = GitWorkspaceOrchestrator(
        FakeCodingWorker(title="t", body="b", cost_usd=cost_usd),
        EnvCredentialResolver({"github": "tok"}),
        ledger=ledger,
    )

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    return orch


async def test_connector_path_fails_closed_over_cap(monkeypatch):
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    connector = CodingWorkerConnector(
        _orchestrator(monkeypatch, usage, cost_usd=0.75),
        github,
        checkpoints=InMemoryCheckpointStore(),
    )

    result = await connector.execute(
        "pr:write", {"repo": "acme/x", "instruction": "x", "job_id": "j1"}
    )

    assert not result.ok
    assert result.data["status"] == "failed"
    assert "per-task budget" in result.data["error"]
    assert github.pulls == []  # fail-closed: no PR from an over-budget run
    assert usage.records[0].outcome == "over_task_budget"
    # "failed" is terminal — the startup reconciler must NOT re-drive the job
    # and spend the budget again on every recovery pass.
    assert await connector.resume_incomplete() == []
    assert len(usage.records) == 1


async def test_workflow_path_fails_closed_over_cap(monkeypatch):
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    store = InMemoryWorkflowStore()
    engine = WorkflowEngine(store)
    engine.register(
        build_coding_worker_workflow(
            _orchestrator(monkeypatch, usage, cost_usd=0.75), github
        )
    )

    await engine.start(
        "coding_worker", "j1",
        {"job_id": "j1", "repo": "acme/x", "instruction": "x"},
    )
    instance = await engine.send_event("j1", "await_approval", {})

    assert instance.status == "failed"
    assert "per-task budget" in instance.error
    assert github.pulls == []
    assert usage.records[0].outcome == "over_task_budget"


async def test_connector_path_monthly_gate_fails_closed(monkeypatch):
    # Phase 5: a spent monthly budget refuses the attempt outright.
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    connector = CodingWorkerConnector(
        _orchestrator(
            monkeypatch, usage, cost_usd=0.25, agent=_agent(monthly_usd=50.0)
        ),
        github,
    )
    await usage.record(UsageRecord(
        scope_key="ws:acme:agent:dev-platform", workspace="acme",
        agent="dev-platform", model="m", cost_usd=50.0,
    ))

    result = await connector.execute(
        "pr:write", {"repo": "acme/x", "instruction": "x", "job_id": "j1"}
    )

    assert not result.ok
    assert "monthly budget" in result.data["error"]
    assert github.pulls == []
    assert usage.records[-1].outcome == "blocked"


async def test_workflow_path_monthly_gate_fails_closed(monkeypatch):
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    engine = WorkflowEngine(InMemoryWorkflowStore())
    engine.register(
        build_coding_worker_workflow(
            _orchestrator(
                monkeypatch, usage, cost_usd=0.25, agent=_agent(monthly_usd=50.0)
            ),
            github,
        )
    )
    await usage.record(UsageRecord(
        scope_key="ws:acme:agent:dev-platform", workspace="acme",
        agent="dev-platform", model="m", cost_usd=50.0,
    ))

    await engine.start(
        "coding_worker", "j1",
        {"job_id": "j1", "repo": "acme/x", "instruction": "x"},
    )
    instance = await engine.send_event("j1", "await_approval", {})

    assert instance.status == "failed"
    assert "monthly budget" in instance.error
    assert github.pulls == []
    assert usage.records[-1].outcome == "blocked"


async def test_within_budget_run_is_recorded_and_ships(monkeypatch):
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    connector = CodingWorkerConnector(
        _orchestrator(monkeypatch, usage, cost_usd=0.25), github
    )

    result = await connector.execute(
        "pr:write", {"repo": "acme/x", "instruction": "x", "job_id": "j2"}
    )

    assert result.ok
    assert len(github.pulls) == 1
    (record,) = usage.records
    assert record.outcome == "ok"
    assert record.task_kind == "coding_worker"
    # Worker spend now counts against the same monthly scope /usage reports.
    assert await usage.monthly_total("ws:acme:agent:dev-platform") == 0.25


async def test_spend_follows_the_invoking_agent_through_the_approval_hop(
    monkeypatch,
):
    """Phase 5 attribution end to end: the gateway stamps the invoking agent
    into the approval args; the workflow's attempt settles under *that*
    agent's scope and cap — not the ledger's default — fixing the deferred
    Phase 4 multi-agent finding."""
    usage = InMemoryUsageStore()
    github = FakeGitHub()

    invoking = _agent(per_task_usd=5.0)
    invoking.metadata.name = "docs-bot"
    default = _agent(per_task_usd=0.10)  # would refuse this run
    ledger = WorkerSpendLedger(
        usage=usage,
        model="m",
        agents={"dev-platform": default, "docs-bot": invoking},
        default_agent="dev-platform",
    )
    orch = GitWorkspaceOrchestrator(
        FakeCodingWorker(title="t", body="b", cost_usd=0.75),
        EnvCredentialResolver({"github": "tok"}),
        ledger=ledger,
    )

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)

    engine = WorkflowEngine(InMemoryWorkflowStore())
    engine.register(build_coding_worker_workflow(orch, github))
    gateway = ToolGateway(
        tools=[CodingWorkerConnector(orch, github)], engine=engine
    )

    pending = await gateway.invoke(
        invoking, "coding_worker.pr:write",
        {"repo": "acme/x", "instruction": "x"},
    )
    assert pending.approval.args["agent"] == "docs-bot"

    resolved = await gateway.resolve(
        pending.approval.id, "@maciag.artur", approve=True
    )

    assert resolved.result.ok  # within docs-bot's cap, over the default's
    assert len(github.pulls) == 1
    (record,) = usage.records
    assert record.agent == "docs-bot"
    assert record.scope_key == "ws:acme:agent:docs-bot"
    assert record.outcome == "ok"
