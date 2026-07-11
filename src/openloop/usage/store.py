"""Usage records, the store protocol, and an in-memory backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class UsageRecord:
    """One unit of spend — the audit trail's row."""

    scope_key: str  # budget scope, e.g. "ws:acme:agent:dev-platform"
    workspace: str
    agent: str
    model: str
    channel: str | None = None
    surface: str | None = None
    user: str | None = None
    task_kind: str | None = None
    # A non-null key makes a charge write idempotent across a checkpoint crash.
    # Ordinary chat/worker records leave it unset and remain append-only.
    idempotency_key: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    outcome: str = "ok"  # ok | blocked | rate_limited | over_task_budget | error
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def _month_start(now: datetime) -> datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


@runtime_checkable
class UsageStore(Protocol):
    async def record(self, usage: UsageRecord) -> bool: ...

    async def monthly_total(
        self, scope_key: str, now: datetime | None = None
    ) -> float:
        """Total USD spent against `scope_key` in the current calendar month."""
        ...

    async def recent(self, limit: int = 50) -> list[UsageRecord]:
        """Most recent usage records (the audit trail), newest first."""
        ...


class InMemoryUsageStore:
    """Process-local audit trail — good for dev and tests, lost on restart."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []
        self._idempotency_keys: set[str] = set()

    async def record(self, usage: UsageRecord) -> bool:
        if usage.idempotency_key:
            if usage.idempotency_key in self._idempotency_keys:
                return False
            self._idempotency_keys.add(usage.idempotency_key)
        self.records.append(usage)
        return True

    async def monthly_total(
        self, scope_key: str, now: datetime | None = None
    ) -> float:
        now = now or datetime.now(timezone.utc)
        start = _month_start(now)
        return sum(
            r.cost_usd
            for r in self.records
            if r.scope_key == scope_key and r.created_at >= start
        )

    async def recent(self, limit: int = 50) -> list[UsageRecord]:
        return sorted(self.records, key=lambda r: r.created_at, reverse=True)[:limit]
