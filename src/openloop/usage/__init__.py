"""Token/cost tracking and the audit trail.

Every handled task records a :class:`UsageRecord` (tokens, cost, outcome) to a
:class:`UsageStore`. The same store answers budget questions — accumulated
monthly spend per agent — so the runtime can enforce spend guardrails.
"""

from openloop.usage.budget import BudgetDecision, budget_scope_key, check_budget
from openloop.usage.ledger import WorkerBudgetExceeded, WorkerSpendLedger
from openloop.usage.limits import (
    InMemoryTaskLimiter,
    LimitDecision,
    TaskLimiter,
    limit_scope_key,
)
from openloop.usage.store import (
    InMemoryUsageStore,
    UsageRecord,
    UsageStore,
)

__all__ = [
    "BudgetDecision",
    "budget_scope_key",
    "check_budget",
    "InMemoryTaskLimiter",
    "InMemoryUsageStore",
    "LimitDecision",
    "limit_scope_key",
    "TaskLimiter",
    "UsageRecord",
    "UsageStore",
    "WorkerBudgetExceeded",
    "WorkerSpendLedger",
]
