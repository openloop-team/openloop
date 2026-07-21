"""Unit tests for the Phase 5 per-scope throughput limits."""

from openloop.agents.schema import Agent, Limits
from openloop.usage import InMemoryTaskLimiter, limit_scope_key
from tests.support.agents import make_agent


def _agent(name="dev-platform", workspace="acme") -> Agent:
    return make_agent(name, workspace)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_limit_scope_key_is_tenant_shaped():
    # One tenant today, but the key carries the tenant dimension so Phase 7
    # swaps key contents, not the seam.
    assert (
        limit_scope_key(_agent())
        == "tenant:default:ws:acme:agent:dev-platform"
    )
    assert limit_scope_key(_agent(), tenant="beta").startswith("tenant:beta:")


async def test_unset_limits_admit_everything():
    limiter = InMemoryTaskLimiter()
    for _ in range(100):
        assert (await limiter.acquire("s", Limits())).allowed


async def test_concurrency_cap_refuses_then_recovers_on_release():
    limiter = InMemoryTaskLimiter()
    limits = Limits(max_concurrent_tasks=2)

    assert (await limiter.acquire("s", limits)).allowed
    assert (await limiter.acquire("s", limits)).allowed
    refused = await limiter.acquire("s", limits)
    assert not refused.allowed
    assert "max 2 concurrent" in refused.reason

    await limiter.release("s")
    assert (await limiter.acquire("s", limits)).allowed


async def test_refused_task_consumes_no_slot_or_window_entry():
    limiter = InMemoryTaskLimiter()
    limits = Limits(max_concurrent_tasks=1, tasks_per_minute=2)

    assert (await limiter.acquire("s", limits)).allowed
    assert not (await limiter.acquire("s", limits)).allowed  # concurrency
    await limiter.release("s")
    # The refusal didn't burn a rate-window entry: one admission so far,
    # so one more fits under tasks_per_minute=2.
    assert (await limiter.acquire("s", limits)).allowed


async def test_rate_cap_refuses_within_window_and_frees_after():
    clock = FakeClock()
    limiter = InMemoryTaskLimiter(clock=clock)
    limits = Limits(tasks_per_minute=2)

    assert (await limiter.acquire("s", limits)).allowed
    await limiter.release("s")
    assert (await limiter.acquire("s", limits)).allowed
    await limiter.release("s")
    refused = await limiter.acquire("s", limits)
    assert not refused.allowed
    assert "2/minute" in refused.reason

    clock.now += 60.0  # the window slides
    assert (await limiter.acquire("s", limits)).allowed


async def test_scopes_are_isolated():
    limiter = InMemoryTaskLimiter()
    limits = Limits(max_concurrent_tasks=1)

    assert (await limiter.acquire("tenant:default:a", limits)).allowed
    assert not (await limiter.acquire("tenant:default:a", limits)).allowed
    # Another (tenant, agent) scope is unaffected — the noisy neighbor
    # can't starve it.
    assert (await limiter.acquire("tenant:default:b", limits)).allowed


async def test_release_never_goes_negative():
    limiter = InMemoryTaskLimiter()
    await limiter.release("s")  # spurious release
    limits = Limits(max_concurrent_tasks=1)
    assert (await limiter.acquire("s", limits)).allowed
    assert not (await limiter.acquire("s", limits)).allowed
