"""Per-scope throughput limits (Phase 5) — noisy-neighbor controls.

Budgets cap *spend*; these cap *throughput*: how many tasks a scope may have
in flight at once (``max_concurrent_tasks``) and how many it may start per
minute (``tasks_per_minute``). The runtime acquires a slot before doing any
work — a refused task never hits the budget check, the memory store, or a
model — and releases it when the turn finishes.

The scope key is tenant-shaped (``tenant:<t>:ws:<w>:agent:<a>``) even though
there is one tenant today, so multi-tenant activation (roadmap Phase 7) swaps
the key contents, not the seam. Limits come from the agent's config
(``spec.limits``); both knobs default to unset = unlimited, so existing
configs behave identically.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from openloop.agents.schema import Agent, Limits

# The rate window. "per minute" is the config's unit; keep them in lockstep.
WINDOW_SECONDS = 60.0


@dataclass(slots=True)
class LimitDecision:
    allowed: bool
    reason: str | None = None


def limit_scope_key(agent: Agent, tenant: str = "default") -> str:
    """Throughput limits are per (tenant, agent) — tenant-shaped from day one."""
    return (
        f"tenant:{tenant}:ws:{agent.metadata.workspace}"
        f":agent:{agent.metadata.name}"
    )


@runtime_checkable
class TaskLimiter(Protocol):
    async def acquire(self, scope_key: str, limits: Limits) -> LimitDecision:
        """Admit a task or refuse it. An admitted task holds a concurrency
        slot (and a rate-window entry) until :meth:`release`; a refused task
        consumes neither."""
        ...

    async def release(self, scope_key: str) -> None:
        """Return an admitted task's concurrency slot."""
        ...


class InMemoryTaskLimiter:
    """Process-local limiter — same posture as :class:`InMemoryUsageStore`.

    Counters live in this process only: a multi-replica deploy limits per
    replica until a shared backend exists (Phase 7 territory, alongside the
    per-tenant quotas). A crash mid-task drops the in-flight counter with the
    process, so restarts can't strand slots. Single-threaded asyncio keeps
    ``acquire`` atomic (no awaits between check and increment).
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._in_flight: dict[str, int] = {}
        self._admissions: dict[str, deque[float]] = {}

    async def acquire(self, scope_key: str, limits: Limits) -> LimitDecision:
        in_flight = self._in_flight.get(scope_key, 0)
        if (
            limits.max_concurrent_tasks is not None
            and in_flight >= limits.max_concurrent_tasks
        ):
            return LimitDecision(
                allowed=False,
                reason=(
                    f"{in_flight} task(s) already running "
                    f"(max {limits.max_concurrent_tasks} concurrent)"
                ),
            )

        now = self._clock()
        window = self._admissions.setdefault(scope_key, deque())
        while window and now - window[0] >= WINDOW_SECONDS:
            window.popleft()
        if (
            limits.tasks_per_minute is not None
            and len(window) >= limits.tasks_per_minute
        ):
            return LimitDecision(
                allowed=False,
                reason=(
                    f"task rate limit reached "
                    f"({limits.tasks_per_minute}/minute)"
                ),
            )

        window.append(now)
        self._in_flight[scope_key] = in_flight + 1
        return LimitDecision(allowed=True)

    async def release(self, scope_key: str) -> None:
        in_flight = self._in_flight.get(scope_key, 0)
        if in_flight > 0:
            self._in_flight[scope_key] = in_flight - 1
