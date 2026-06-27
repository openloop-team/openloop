"""Tests for the usage audit trail and budget enforcement."""

from datetime import datetime, timedelta, timezone

from openloop.agents import load_agent
from openloop.agents.schema import Agent
from openloop.runtime import Runtime, Task
from openloop.usage import (
    InMemoryUsageStore,
    UsageRecord,
    budget_scope_key,
    check_budget,
)
from openloop.testing import EXAMPLE_AGENT, FakeGateway


def _agent() -> Agent:
    return load_agent(EXAMPLE_AGENT)  # budget: monthly_usd=50, per_task=0.5, block


async def test_monthly_total_sums_current_month_only():
    store = InMemoryUsageStore()
    now = datetime.now(timezone.utc)
    last_month = (now.replace(day=1) - timedelta(days=2))
    await store.record(UsageRecord(scope_key="s", workspace="w", agent="a",
                                   model="m", cost_usd=3.0))
    await store.record(UsageRecord(scope_key="s", workspace="w", agent="a",
                                   model="m", cost_usd=1.5,
                                   created_at=last_month))
    assert await store.monthly_total("s") == 3.0


async def test_check_budget_blocks_when_monthly_exceeded():
    agent = _agent()
    store = InMemoryUsageStore()
    await store.record(UsageRecord(scope_key=budget_scope_key(agent),
                                   workspace="acme", agent="dev-platform",
                                   model="m", cost_usd=50.0))
    decision = await check_budget(agent, store)
    assert not decision.allowed
    assert "monthly budget reached" in decision.reason


async def test_handle_records_usage():
    agent = _agent()
    usage = InMemoryUsageStore()
    runtime = Runtime(agent, gateway=FakeGateway(), usage=usage)
    await runtime.handle(Task(text="hi", surface="slack", channel="#dev-platform",
                              user="U1", kind="summarize"))
    assert len(usage.records) == 1
    rec = usage.records[0]
    assert rec.agent == "dev-platform"
    assert rec.channel == "#dev-platform"
    assert rec.task_kind == "summarize"
    assert rec.outcome == "ok"


async def test_budget_block_short_circuits_model_call():
    agent = _agent()
    usage = InMemoryUsageStore()
    await usage.record(UsageRecord(scope_key=budget_scope_key(agent),
                                   workspace="acme", agent="dev-platform",
                                   model="m", cost_usd=50.0))
    gateway = FakeGateway(reply="should not run")
    runtime = Runtime(agent, gateway=gateway, usage=usage)

    response = await runtime.handle(
        Task(text="do work", surface="slack", channel="#dev-platform")
    )
    assert response.model == "budget-guard"
    assert "blocked" in response.text.lower()
    assert gateway.last_messages is None  # model never called
    assert usage.records[-1].outcome == "blocked"


async def test_per_task_overage_is_flagged():
    agent = _agent()  # per_task_usd = 0.50
    usage = InMemoryUsageStore()
    gateway = FakeGateway()

    async def expensive(model, messages, **kwargs):
        from openloop.models.gateway import ModelResponse
        return ModelResponse(text="ok", model=model, cost_usd=0.75)

    gateway.complete = expensive  # type: ignore[assignment]
    runtime = Runtime(agent, gateway=gateway, usage=usage)
    await runtime.handle(Task(text="hi", surface="slack", channel="#dev-platform"))
    assert usage.records[0].outcome == "over_task_budget"
