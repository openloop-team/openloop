"""Unit tests for the worker-spend ledger (Phase 4 gate + Phase 5 promotion).

Covers the three enforcement legs — record, fail-closed per-task cap,
fail-closed monthly gate — and the Phase 5 attribution rule: spend lands on
the *invoking* agent (threaded through ``WorkerState.agent``), falling back
to the boot-time default only when no identity was carried.
"""

import pytest

from openloop.agents.schema import (
    Agent,
    AgentMetadata,
    AgentSpec,
    Budget,
    ModelPolicy,
)
from openloop.credentials import EnvCredentialResolver
from openloop.tools.coding_worker import (
    GitWorkspaceOrchestrator,
    WorkerState,
)
from openloop.usage import (
    InMemoryUsageStore,
    UsageRecord,
    WorkerBudgetExceeded,
    WorkerSpendLedger,
)
from openloop.testing import FakeCodingWorker


def _agent(
    name="dev-platform",
    workspace="acme",
    *,
    monthly_usd=None,
    per_task_usd=None,
    on_exceeded="block",
) -> Agent:
    return Agent(
        metadata=AgentMetadata(name=name, workspace=workspace),
        spec=AgentSpec(
            model_policy=ModelPolicy(default="m"),
            budget=Budget(
                monthly_usd=monthly_usd,
                per_task_usd=per_task_usd,
                on_exceeded=on_exceeded,
            ),
        ),
    )


def _ledger(
    usage=None,
    agents=None,
    default="dev-platform",
    require_per_task_cap=False,
    **budget,
):
    agents = agents or {"dev-platform": _agent(**budget)}
    return WorkerSpendLedger(
        usage=usage or InMemoryUsageStore(),
        model="m",
        agents=agents,
        default_agent=default,
        require_per_task_cap=require_per_task_cap,
    )


def _state(job_id="j1", agent=None):
    return WorkerState(
        job_id=job_id, repo="a/b", instruction="x", base="main",
        branch=f"openloop/job-{job_id}", agent=agent,
    )


async def _spent(usage, scope_key, cost_usd):
    """Seed accumulated monthly spend for a scope."""
    await usage.record(UsageRecord(
        scope_key=scope_key, workspace="acme", agent="seed", model="m",
        cost_usd=cost_usd,
    ))


# --- settle: record + per-task cap ---


async def test_settle_records_spend_under_the_agent_scope():
    usage = InMemoryUsageStore()
    await _ledger(usage).settle(
        job_id="j1", cost_usd=0.12, prompt_tokens=100, completion_tokens=50
    )

    (record,) = usage.records
    assert record.scope_key == "ws:acme:agent:dev-platform"
    assert record.task_kind == "coding_worker"
    assert record.cost_usd == 0.12
    assert record.prompt_tokens == 100
    assert record.completion_tokens == 50
    assert record.outcome == "ok"
    # Worker spend lands in the same monthly total /usage reads.
    assert await usage.monthly_total("ws:acme:agent:dev-platform") == 0.12


async def test_settle_with_idempotency_key_records_once():
    usage = InMemoryUsageStore()
    ledger = _ledger(usage)

    await ledger.settle(
        job_id="j1",
        idempotency_key="analysis-attempt-1",
        cost_usd=0.12,
        prompt_tokens=100,
        completion_tokens=50,
    )
    # A checkpoint crash can repeat settlement; the stable attempt key keeps
    # the audit row and monthly total exactly-once.
    await ledger.settle(
        job_id="j1",
        idempotency_key="analysis-attempt-1",
        cost_usd=0.12,
        prompt_tokens=100,
        completion_tokens=50,
    )

    assert len(usage.records) == 1
    assert usage.records[0].idempotency_key == "analysis-attempt-1"
    assert await usage.monthly_total("ws:acme:agent:dev-platform") == 0.12


async def test_settle_attributes_to_the_invoking_agent():
    usage = InMemoryUsageStore()
    agents = {
        "dev-platform": _agent(per_task_usd=0.50),
        "docs-bot": _agent("docs-bot", per_task_usd=5.0),
    }
    ledger = _ledger(usage, agents)

    # Within docs-bot's cap even though it would blow dev-platform's — the
    # invoking agent's budget is the one enforced.
    await ledger.settle(agent="docs-bot", job_id="j1", cost_usd=0.75)

    (record,) = usage.records
    assert record.agent == "docs-bot"
    assert record.scope_key == "ws:acme:agent:docs-bot"
    assert record.outcome == "ok"


async def test_settle_without_agent_falls_back_to_the_default():
    usage = InMemoryUsageStore()
    # Pre-Phase 5 approvals/checkpoints carry no agent identity.
    await _ledger(usage).settle(agent=None, job_id="j1", cost_usd=0.10)
    assert usage.records[0].agent == "dev-platform"


async def test_settle_unknown_agent_falls_back_capped(caplog):
    # An agent removed from config between approval and run must not dodge
    # the cap — spend is attributed to the default owner, loudly.
    usage = InMemoryUsageStore()
    ledger = _ledger(usage, per_task_usd=0.50)
    with caplog.at_level("WARNING"):
        with pytest.raises(WorkerBudgetExceeded):
            await ledger.settle(agent="ghost", job_id="j1", cost_usd=0.75)
    assert "unknown agent 'ghost'" in caplog.text
    assert usage.records[0].agent == "dev-platform"
    assert usage.records[0].outcome == "over_task_budget"


async def test_settle_over_cap_fails_closed_and_still_records():
    usage = InMemoryUsageStore()
    with pytest.raises(WorkerBudgetExceeded, match="j1"):
        await _ledger(usage, per_task_usd=0.50).settle(job_id="j1", cost_usd=0.51)

    # The spend already happened — it must stay visible in the audit trail.
    (record,) = usage.records
    assert record.outcome == "over_task_budget"
    assert record.cost_usd == 0.51


async def test_no_cap_records_without_ever_raising():
    usage = InMemoryUsageStore()
    await _ledger(usage, per_task_usd=None).settle(job_id="j1", cost_usd=999.0)
    assert usage.records[0].outcome == "ok"


async def test_required_cap_settle_fails_closed_without_cap():
    usage = InMemoryUsageStore()
    ledger = _ledger(usage, per_task_usd=None, require_per_task_cap=True)

    with pytest.raises(WorkerBudgetExceeded, match="no per-task spend cap"):
        await ledger.settle(job_id="j1", cost_usd=0.25)

    assert usage.records[0].cost_usd == 0.25
    assert usage.records[0].outcome == "blocked"


async def test_spend_at_the_cap_is_allowed():
    await _ledger(per_task_usd=0.50).settle(job_id="j1", cost_usd=0.50)


async def test_record_failure_propagates_fail_closed():
    # A run that cannot be accounted for must not proceed to push.
    class BrokenStore(InMemoryUsageStore):
        async def record(self, usage):
            raise RuntimeError("store down")

    with pytest.raises(RuntimeError, match="store down"):
        await _ledger(BrokenStore()).settle(job_id="j1", cost_usd=0.01)


# --- check_monthly: the Phase 5 pre-attempt gate ---


async def test_check_monthly_passes_under_budget():
    usage = InMemoryUsageStore()
    ledger = _ledger(usage, monthly_usd=50.0)
    await _spent(usage, "ws:acme:agent:dev-platform", 49.0)
    await ledger.check_monthly("dev-platform", job_id="j1")  # no raise


async def test_check_monthly_blocks_and_records_the_refusal():
    usage = InMemoryUsageStore()
    ledger = _ledger(usage, monthly_usd=50.0, on_exceeded="block")
    await _spent(usage, "ws:acme:agent:dev-platform", 50.0)

    with pytest.raises(WorkerBudgetExceeded, match="j1"):
        await ledger.check_monthly("dev-platform", job_id="j1")

    # The refusal is visible in /audit: zero cost, outcome=blocked.
    blocked = usage.records[-1]
    assert blocked.outcome == "blocked"
    assert blocked.cost_usd == 0.0
    assert blocked.task_kind == "coding_worker"


async def test_check_monthly_warn_mode_proceeds():
    # Budget's block|warn semantics are preserved: warn logs and proceeds.
    usage = InMemoryUsageStore()
    ledger = _ledger(usage, monthly_usd=50.0, on_exceeded="warn")
    await _spent(usage, "ws:acme:agent:dev-platform", 50.0)
    await ledger.check_monthly("dev-platform", job_id="j1")  # no raise


async def test_check_monthly_no_budget_always_passes():
    await _ledger(monthly_usd=None).check_monthly(None, job_id="j1")


async def test_check_monthly_required_cap_refuses_before_work():
    usage = InMemoryUsageStore()
    ledger = _ledger(usage, per_task_usd=None, require_per_task_cap=True)

    with pytest.raises(WorkerBudgetExceeded, match="no per-task spend cap"):
        await ledger.check_monthly("dev-platform", job_id="j1")

    assert usage.records[0].cost_usd == 0.0
    assert usage.records[0].outcome == "blocked"


async def test_check_monthly_gates_on_the_invoking_agents_budget():
    usage = InMemoryUsageStore()
    agents = {
        "dev-platform": _agent(monthly_usd=50.0),
        "docs-bot": _agent("docs-bot", monthly_usd=1.0),
    }
    ledger = _ledger(usage, agents)
    await _spent(usage, "ws:acme:agent:docs-bot", 1.0)

    with pytest.raises(WorkerBudgetExceeded, match="docs-bot"):
        await ledger.check_monthly("docs-bot", job_id="j1")
    # The other agent's budget is untouched by docs-bot's exhaustion.
    await ledger.check_monthly("dev-platform", job_id="j2")


# --- the orchestrator wiring: gates around the attempt boundary ---


def _orchestrator(monkeypatch, ledger, *, cost_usd=0.0):
    orch = GitWorkspaceOrchestrator(
        FakeCodingWorker(cost_usd=cost_usd, prompt_tokens=10, completion_tokens=5),
        EnvCredentialResolver({"github": "t"}),
        ledger=ledger,
    )
    commands = []

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        commands.append(cmd)
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    return orch, commands


async def test_orchestrator_settles_before_the_push_boundary(monkeypatch):
    """Over-budget attempt: spend recorded, exception raised, and — the
    fail-closed part — the branch is never committed or pushed."""
    usage = InMemoryUsageStore()
    orch, commands = _orchestrator(
        monkeypatch, _ledger(usage, per_task_usd=0.50), cost_usd=0.75
    )
    with pytest.raises(WorkerBudgetExceeded):
        await orch.run_attempt(_state())

    assert usage.records[0].outcome == "over_task_budget"
    flat = [arg for cmd in commands for arg in cmd]
    assert "clone" in flat  # the attempt provisioned…
    assert "commit" not in flat and "push" not in flat  # …but never shipped


async def test_orchestrator_monthly_gate_refuses_before_any_work(monkeypatch):
    """A spent monthly budget refuses the attempt before clone — no git
    command runs at all, and the refusal is recorded."""
    usage = InMemoryUsageStore()
    ledger = _ledger(usage, monthly_usd=50.0)
    await _spent(usage, "ws:acme:agent:dev-platform", 50.0)
    orch, commands = _orchestrator(monkeypatch, ledger)

    with pytest.raises(WorkerBudgetExceeded):
        await orch.run_attempt(_state())

    assert commands == []  # not even a clone
    assert usage.records[-1].outcome == "blocked"


async def test_orchestrator_required_cap_refuses_before_any_work(monkeypatch):
    """The agentic-backend backstop catches config drift before clone."""
    usage = InMemoryUsageStore()
    ledger = _ledger(usage, per_task_usd=None, require_per_task_cap=True)
    orch, commands = _orchestrator(monkeypatch, ledger)

    with pytest.raises(WorkerBudgetExceeded, match="no per-task spend cap"):
        await orch.run_attempt(_state())

    assert commands == []  # not even a clone
    assert usage.records[-1].outcome == "blocked"


async def test_orchestrator_attributes_spend_to_state_agent(monkeypatch):
    usage = InMemoryUsageStore()
    agents = {
        "dev-platform": _agent(per_task_usd=0.50),
        "docs-bot": _agent("docs-bot", per_task_usd=5.0),
    }
    orch, commands = _orchestrator(
        monkeypatch, _ledger(usage, agents), cost_usd=0.75
    )

    await orch.run_attempt(_state(agent="docs-bot"))

    # docs-bot's cap (not the default agent's) governed the attempt.
    assert usage.records[0].agent == "docs-bot"
    assert usage.records[0].outcome == "ok"
    assert any("push" in cmd for cmd in commands)


async def test_orchestrator_records_within_budget_spend_and_pushes(monkeypatch):
    usage = InMemoryUsageStore()
    orch, commands = _orchestrator(
        monkeypatch, _ledger(usage, per_task_usd=0.50), cost_usd=0.25
    )
    outcome = await orch.run_attempt(_state())

    assert outcome.cost_usd == 0.25
    assert usage.records[0].outcome == "ok"
    assert any("push" in cmd for cmd in commands)
