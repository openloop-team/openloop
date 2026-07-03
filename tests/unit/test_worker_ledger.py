"""Unit tests for the Phase 4 worker-spend ledger (record + fail-closed cap)."""

import pytest

from openloop.credentials import EnvCredentialResolver
from openloop.tools.coding_worker import (
    GitWorkspaceOrchestrator,
    WorkerState,
)
from openloop.usage import (
    InMemoryUsageStore,
    WorkerBudgetExceeded,
    WorkerSpendLedger,
)
from openloop.testing import FakeCodingWorker


def _ledger(usage=None, per_task_usd=None):
    return WorkerSpendLedger(
        usage=usage or InMemoryUsageStore(),
        scope_key="ws:acme:agent:dev-platform",
        workspace="acme",
        agent="dev-platform",
        model="m",
        per_task_usd=per_task_usd,
    )


def _state(job_id="j1"):
    return WorkerState(
        job_id=job_id, repo="a/b", instruction="x", base="main",
        branch=f"openloop/job-{job_id}",
    )


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


async def test_spend_at_the_cap_is_allowed():
    await _ledger(per_task_usd=0.50).settle(job_id="j1", cost_usd=0.50)


async def test_record_failure_propagates_fail_closed():
    # A run that cannot be accounted for must not proceed to push.
    class BrokenStore(InMemoryUsageStore):
        async def record(self, usage):
            raise RuntimeError("store down")

    with pytest.raises(RuntimeError, match="store down"):
        await _ledger(BrokenStore()).settle(job_id="j1", cost_usd=0.01)


async def test_orchestrator_settles_before_the_push_boundary(monkeypatch):
    """Over-budget attempt: spend recorded, exception raised, and — the
    fail-closed part — the branch is never committed or pushed."""
    usage = InMemoryUsageStore()
    orch = GitWorkspaceOrchestrator(
        FakeCodingWorker(cost_usd=0.75, prompt_tokens=10, completion_tokens=5),
        EnvCredentialResolver({"github": "t"}),
        ledger=_ledger(usage, per_task_usd=0.50),
    )
    commands = []

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        commands.append(cmd)
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    with pytest.raises(WorkerBudgetExceeded):
        await orch.run_attempt(_state())

    assert usage.records[0].outcome == "over_task_budget"
    flat = [arg for cmd in commands for arg in cmd]
    assert "clone" in flat  # the attempt provisioned…
    assert "commit" not in flat and "push" not in flat  # …but never shipped


async def test_orchestrator_records_within_budget_spend_and_pushes(monkeypatch):
    usage = InMemoryUsageStore()
    orch = GitWorkspaceOrchestrator(
        FakeCodingWorker(cost_usd=0.25),
        EnvCredentialResolver({"github": "t"}),
        ledger=_ledger(usage, per_task_usd=0.50),
    )
    commands = []

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        commands.append(cmd)
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    outcome = await orch.run_attempt(_state())

    assert outcome.cost_usd == 0.25
    assert usage.records[0].outcome == "ok"
    assert any("push" in cmd for cmd in commands)
