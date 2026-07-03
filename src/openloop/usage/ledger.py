"""The minimal worker-spend ledger — the Phase 4 gate.

Heavy agentic workers (OpenHands) can burn real money per run, so before one
may be registered every worker attempt must be *recorded* and *capped*:

- **Recorded:** each attempt's model spend lands in the :class:`UsageStore`
  under the owning agent's budget scope key, so it shows up in ``/usage``
  month-to-date and the ``/audit`` trail alongside chat spend.
- **Fail-closed cap:** an attempt whose spend exceeds the agent's per-task
  budget raises :class:`WorkerBudgetExceeded` *before* any push or PR — the
  job fails loudly instead of shipping an over-budget change. This is the
  OpenLoop-side backstop to whatever in-run limits the worker itself has.

The ledger lives inside :class:`~openloop.tools.coding_worker.
GitWorkspaceOrchestrator` — the one attempt boundary **both** durable paths
(the connector's checkpoint fallback and the workflow) run through — so an
engine-less deploy is covered by construction, per the roadmap's leading
invariant.

A store failure while recording also fails the attempt (the exception
propagates out of the attempt): a worker run that cannot be accounted for
does not get to push. Full budget unification (monthly enforcement, rate
limits) is Phase 5; this is deliberately just record + per-task cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openloop.usage.store import UsageRecord, UsageStore

logger = logging.getLogger(__name__)

# The audit trail's task_kind for worker attempts, so worker spend is
# distinguishable from chat spend in /audit.
WORKER_TASK_KIND = "coding_worker"


class WorkerBudgetExceeded(RuntimeError):
    """A worker attempt spent more than the per-task budget — fail closed."""


@dataclass(slots=True)
class WorkerSpendLedger:
    """Records one worker attempt's spend and enforces the per-task cap."""

    usage: UsageStore
    scope_key: str  # the owning agent's budget scope (budget_scope_key)
    workspace: str
    agent: str
    model: str
    per_task_usd: float | None = None

    async def settle(
        self,
        *,
        job_id: str,
        cost_usd: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record the attempt's spend; raise if it blew the per-task cap.

        Always records first — over-budget spend already happened and must be
        visible in the audit trail — then fails the attempt. Callers run this
        *before* the push/PR boundary so an over-budget change never ships.
        """
        over = self.per_task_usd is not None and cost_usd > self.per_task_usd
        await self.usage.record(
            UsageRecord(
                scope_key=self.scope_key,
                workspace=self.workspace,
                agent=self.agent,
                model=self.model,
                task_kind=WORKER_TASK_KIND,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                outcome="over_task_budget" if over else "ok",
            )
        )
        if over:
            raise WorkerBudgetExceeded(
                f"worker job {job_id} spent ${cost_usd:.4f}, over the "
                f"${self.per_task_usd:.2f} per-task budget — failing closed "
                "(no push, no PR)"
            )
        logger.debug(
            "worker job %s spend recorded: $%.4f against %s",
            job_id, cost_usd, self.scope_key,
        )
