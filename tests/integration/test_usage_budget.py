"""Tests for the usage audit trail, budget enforcement, and throughput limits."""

from pathlib import Path
import asyncio
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
from openloop.testing import FakeGateway, in_memory_workflow_engine
from tests.support.agents import make_agent

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _agent() -> Agent:
    return load_agent(AGENT_YAML)  # budget: monthly_usd=50, per_task=0.5, block


def test_scope_key_is_the_id_alone():
    # The id is globally unique, so workspace/name would only re-split billing
    # history when edited.
    agent = _agent()
    assert budget_scope_key(agent) == f"agent:{agent.metadata.id}"


def test_scope_key_distinguishes_same_spec_different_identity():
    # A delete-and-recreate under the same name (even a byte-identical spec)
    # gets a fresh minted id — a different principal, a different scope.
    assert budget_scope_key(make_agent("twin", "acme")) != budget_scope_key(
        make_agent("twin", "acme")
    )


def test_scope_key_survives_a_rename():
    # Rename/billing continuity: the scope follows the id, not the handle.
    agent = _agent()
    before = budget_scope_key(agent)
    agent.metadata.name = "renamed"
    agent.metadata.workspace = "moved"
    assert budget_scope_key(agent) == before


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
    runtime = Runtime(
        agent,
        gateway=FakeGateway(),
        usage=usage,
        engine=in_memory_workflow_engine(),
    )
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
    runtime = Runtime(
        agent, gateway=gateway, usage=usage, engine=in_memory_workflow_engine()
    )

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
    runtime = Runtime(
        agent, gateway=gateway, usage=usage, engine=in_memory_workflow_engine()
    )
    await runtime.handle(Task(text="hi", surface="slack", channel="#dev-platform"))
    assert usage.records[0].outcome == "over_task_budget"


# --- Phase 5 throughput limits at the runtime entry ---


def _task(text="hi"):
    return Task(text=text, surface="slack", channel="#dev-platform", user="U1")


async def test_rate_limited_task_is_refused_before_the_model():
    agent = _agent()
    agent.spec.limits.tasks_per_minute = 1
    usage = InMemoryUsageStore()
    gateway = FakeGateway(reply="second call")
    runtime = Runtime(
        agent, gateway=gateway, usage=usage, engine=in_memory_workflow_engine()
    )

    first = await runtime.handle(_task("one"))
    assert first.model != "throughput-guard"

    second = await runtime.handle(_task("two"))
    assert second.model == "throughput-guard"
    assert "rate limit" in second.text.lower()
    # The model was called for the admitted task ("one") and never for the refused
    # one ("two"). (The loop appends the final answer to the log, so the last
    # message is no longer the user turn — assert on membership, not position.)
    seen = [m.get("content") for m in gateway.last_messages]
    assert "one" in seen
    assert "two" not in seen
    # The refusal is visible in the audit trail.
    assert usage.records[-1].outcome == "rate_limited"
    assert usage.records[-1].cost_usd == 0.0


async def test_concurrency_limit_refuses_overlapping_tasks_then_recovers():
    agent = _agent()
    agent.spec.limits.max_concurrent_tasks = 1
    gate = asyncio.Event()

    class SlowGateway(FakeGateway):
        async def complete(self, model, messages, **kwargs):
            await gate.wait()
            return await super().complete(model, messages, **kwargs)

    runtime = Runtime(
        agent,
        gateway=SlowGateway(),
        usage=InMemoryUsageStore(),
        engine=in_memory_workflow_engine(),
    )

    in_flight = asyncio.create_task(runtime.handle(_task("slow")))
    await asyncio.sleep(0)  # let the first task acquire its slot
    refused = await runtime.handle(_task("overlap"))
    assert refused.model == "throughput-guard"
    assert "concurrent" in refused.text

    gate.set()
    assert (await in_flight).model != "throughput-guard"
    # The slot was released on completion — the next task is admitted.
    assert (await runtime.handle(_task("after"))).model != "throughput-guard"


async def test_unset_limits_change_nothing():
    agent = _agent()  # example agent's limits are ignored: fresh Limits below
    agent.spec.limits.max_concurrent_tasks = None
    agent.spec.limits.tasks_per_minute = None
    runtime = Runtime(
        agent,
        gateway=FakeGateway(),
        usage=InMemoryUsageStore(),
        engine=in_memory_workflow_engine(),
    )
    for _ in range(5):
        assert (await runtime.handle(_task())).model != "throughput-guard"
