"""Budget guardrails — enforce per-agent monthly spend before a task runs.

Per-task cost can only be known after the model responds, so this gates on
*accumulated* monthly spend: if the agent has already hit its monthly cap and
`on_exceeded: block`, the task is refused before any spend. `warn` logs and
proceeds. Per-task overages are flagged on the usage record after the fact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openloop.agents.schema import Agent
from openloop.usage.store import UsageStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BudgetDecision:
    allowed: bool
    reason: str | None = None


def budget_scope_key(agent: Agent) -> str:
    """Budgets are per-agent (per-channel scoping can extend this later).

    Keyed on the durable id — the identity of record — so billing/audit
    lineage survives renames and workspace moves, and a delete-and-recreate
    under the same name is a different principal with a fresh scope.
    """
    return f"agent:{agent.metadata.id}"


async def check_budget(agent: Agent, usage: UsageStore) -> BudgetDecision:
    budget = agent.spec.budget
    if budget.monthly_usd is None:
        return BudgetDecision(allowed=True)

    spent = await usage.monthly_total(budget_scope_key(agent))
    if spent < budget.monthly_usd:
        return BudgetDecision(allowed=True)

    detail = f"${spent:.2f} / ${budget.monthly_usd:.2f} this month"
    if budget.on_exceeded == "block":
        return BudgetDecision(
            allowed=False, reason=f"monthly budget reached ({detail})"
        )
    logger.warning("agent %s over monthly budget (%s) — warn mode, proceeding",
                   agent.metadata.name, detail)
    return BudgetDecision(allowed=True)
