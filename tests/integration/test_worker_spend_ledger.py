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

from pathlib import Path
import pytest

from openloop.agents import load_agent
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.credentials import EnvCredentialResolver
from openloop.tools import ToolGateway
from openloop.tools.coding_worker import (
    CodingWorkerConnector,
    GitWorkspaceOrchestrator,
    WorkerRunAborted,
)
from openloop.usage import (
    InMemoryUsageStore,
    UsageRecord,
    WorkerSpendLedger,
    budget_scope_key,
)
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.coding_worker import build_coding_worker_workflow
from openloop.testing import FakeCodingWorker, FakeGitHub

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _agent(per_task_usd=0.50, monthly_usd=None):
    agent = load_agent(AGENT_YAML)  # name=dev-platform, workspace=acme
    agent.spec.budget.per_task_usd = per_task_usd
    agent.spec.budget.monthly_usd = monthly_usd
    return agent


# Every _agent() load carries the fixture's one stamped id, so the scope is
# stable across instances.
_SCOPE = budget_scope_key(load_agent(AGENT_YAML))


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


async def test_in_run_abort_records_partial_spend_and_fails_closed(monkeypatch):
    # A worker that stops itself at the ceiling: the orchestrator records the
    # spend that already happened and fails the attempt closed — no push, no PR.
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    agent = _agent(per_task_usd=0.50)
    ledger = WorkerSpendLedger(
        usage=usage, model="m",
        agents={agent.metadata.name: agent}, default_agent=agent.metadata.name,
    )

    class AbortingWorker:
        async def run(self, workspace, state, on_step=None):
            # The orchestrator must have stamped this agent's cap for the guard.
            assert state.budget_usd == 0.50
            raise WorkerRunAborted(
                "in-run spend $0.55 reached the $0.50 per-task cap",
                cost_usd=0.55, prompt_tokens=120, completion_tokens=30,
            )

    orch = GitWorkspaceOrchestrator(
        AbortingWorker(), EnvCredentialResolver({"github": "tok"}), ledger=ledger,
    )

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    connector = CodingWorkerConnector(orch, github, checkpoints=InMemoryCheckpointStore())

    result = await connector.execute(
        "pr:write", {"repo": "acme/x", "instruction": "x", "job_id": "j1"}
    )

    assert not result.ok
    assert result.data["status"] == "failed"
    assert github.pulls == []  # fail closed: no PR from the aborted run
    # The partial spend is on the audit trail, marked over the cap.
    assert usage.records[0].cost_usd == 0.55
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
        scope_key=_SCOPE, workspace="acme",
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
        scope_key=_SCOPE, workspace="acme",
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
        "pr:write",
        {
            "repo": "acme/x", "instruction": "x", "job_id": "j2",
            # The engine-less path receives session_id via _args_for_execute the
            # same way the gateway stamps it into the approval args (step 5).
            "session_id": "sess-direct",
        },
    )

    assert result.ok
    assert len(github.pulls) == 1
    (record,) = usage.records
    assert record.outcome == "ok"
    assert record.task_kind == "coding_worker"
    # Step 5: the direct execute → WorkerState → settle leg carries session_id.
    assert record.session_id == "sess-direct"
    # Worker spend now counts against the same monthly scope /usage reports.
    assert await usage.monthly_total(_SCOPE) == 0.25


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
    # A distinct principal needs a distinct identity (both helpers load the
    # same fixture file, which carries one id).
    invoking.metadata.id = "b1f2a7c92f3d4f45a51f2f8f31c9dd42"
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
        session_id="sess-abc",
    )
    assert pending.approval.args["agent"] == "docs-bot"
    # The durable identity is pinned at approval alongside the name, so the
    # settle-time reuse guard can verify the principal, not just the handle.
    assert pending.approval.args["agent_id"] == invoking.metadata.id
    # Step 5: the gateway stamps the originating session id into the approval
    # args, so it survives into the durable workflow state.
    assert pending.approval.args["session_id"] == "sess-abc"

    resolved = await gateway.resolve(
        pending.approval.id, "@maciag.artur", approve=True
    )
    await engine.wait_background(pending.approval.args["job_id"])

    assert resolved.result.ok  # within docs-bot's cap, over the default's
    assert len(github.pulls) == 1
    (record,) = usage.records
    assert record.agent == "docs-bot"
    # The scope is the pinned identity's — spend followed the id end to end.
    assert record.scope_key == budget_scope_key(invoking)
    assert record.outcome == "ok"
    # Attribution envelope (finding 4): the workflow path carries the job, the
    # approval, and the approver all the way to the spend record.
    assert record.job_id == pending.approval.args["job_id"]
    assert record.approval_id == pending.approval.id
    assert record.approver == "maciag.artur"
    # Step 5: session attribution flows gateway → durable workflow → settle,
    # landing on the spend record (not derived from warm_key, which is absent).
    assert record.session_id == "sess-abc"


async def test_recreated_same_name_agent_fails_the_approved_job_closed(
    monkeypatch,
):
    """The name-reuse hole, closed end to end: spend approved under one
    docs-bot must never settle under a recreated docs-bot (same name, fresh
    identity). The pinned id no longer resolves at run time, so the workflow
    fails — no push, no PR, and nothing attributed to the new principal."""
    usage = InMemoryUsageStore()
    github = FakeGitHub()

    invoking = _agent(per_task_usd=5.0)
    invoking.metadata.name = "docs-bot"
    invoking.metadata.id = "b1f2a7c92f3d4f45a51f2f8f31c9dd42"
    recreated = _agent(per_task_usd=5.0)
    recreated.metadata.name = "docs-bot"  # same handle, different principal
    ledger = WorkerSpendLedger(
        usage=usage,
        model="m",
        agents={"docs-bot": recreated},
        default_agent="docs-bot",
    )
    orch = GitWorkspaceOrchestrator(
        FakeCodingWorker(title="t", body="b", cost_usd=0.75),
        EnvCredentialResolver({"github": "tok"}),
        ledger=ledger,
    )

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)

    store = InMemoryWorkflowStore()
    engine = WorkflowEngine(store)
    engine.register(build_coding_worker_workflow(orch, github))
    gateway = ToolGateway(
        tools=[CodingWorkerConnector(orch, github)], engine=engine
    )

    pending = await gateway.invoke(
        invoking, "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"}
    )
    assert pending.approval.args["agent_id"] == invoking.metadata.id

    await gateway.resolve(pending.approval.id, "@maciag.artur", approve=True)
    await engine.wait_background(pending.approval.args["job_id"])

    (instance,) = await store.recent()
    assert instance.status == "failed"
    assert "unknown agent identity" in instance.error
    assert github.pulls == []  # fail closed: no PR
    assert usage.records == []  # no spend on the recreated principal's scope
