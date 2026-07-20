"""The worker-spend ledger — Phase 4's gate, promoted to full accounting in
Phase 5.

Heavy agentic workers (OpenHands) can burn real money per run, so before one
may be registered every worker attempt must be *recorded* and *capped*:

- **Recorded:** each attempt's model spend lands in the :class:`UsageStore`
  under the **invoking agent's** budget scope key, so it shows up in
  ``/usage`` month-to-date and the ``/audit`` trail alongside chat spend.
  The invoking agent's name is threaded from the tool gateway through the
  approval args into :class:`~openloop.tools.coding_worker.WorkerState`
  (Phase 5) — attribution follows whoever asked, not a boot-time owner.
- **Monthly gate (fail-closed):** before an attempt does any work, the
  invoking agent's accumulated monthly spend is checked with the same
  :func:`~openloop.usage.budget.check_budget` the chat runtime uses —
  preserving ``Budget``'s ``block | warn`` semantics. A blocked attempt
  raises :class:`WorkerBudgetExceeded` before a credential is even resolved.
- **Per-task cap (fail-closed):** an attempt whose spend exceeds the invoking
  agent's ``per_task_usd`` raises :class:`WorkerBudgetExceeded` *before* any
  push or PR — the job fails loudly instead of shipping an over-budget
  change. This is the OpenLoop-side backstop to whatever in-run limits the
  worker itself has.

The ledger lives inside :class:`~openloop.tools.coding_worker.
GitWorkspaceOrchestrator` — the one attempt boundary **both** durable paths
(the connector's checkpoint fallback and the workflow) run through — so an
engine-less deploy is covered by construction, per the roadmap's leading
invariant.

A store failure while recording also fails the attempt (the exception
propagates out of the attempt): a worker run that cannot be accounted for
does not get to push.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openloop.agents.schema import Agent
from openloop.usage.budget import budget_scope_key, check_budget
from openloop.usage.store import UsageRecord, UsageStore

logger = logging.getLogger(__name__)

# The audit trail's task_kind for worker attempts, so worker spend is
# distinguishable from chat spend in /audit.
WORKER_TASK_KIND = "coding_worker"


class WorkerBudgetExceeded(RuntimeError):
    """A worker attempt hit the invoking agent's budget — fail closed."""


@dataclass(slots=True)
class WorkerSpendLedger:
    """Records worker-attempt spend and enforces the invoking agent's budget.

    ``agents`` is the live agent-config map; ``default_agent`` is the
    attribution fallback for attempts that carry **no** agent identity
    (approvals or checkpoints created before Phase 5) — the boot-time owner
    heuristic Phase 4 used for everything. An *asserted* name missing from
    the map (agent removed or renamed since approval) fails closed instead:
    it is never attributed to the default.

    ``require_per_task_cap`` is set for agentic worker backends whose boot gate
    requires a cap. It keeps stale approved jobs fail-closed if config drifts
    after approval but before recovery.
    """

    usage: UsageStore
    model: str
    agents: dict[str, Agent]
    default_agent: str
    require_per_task_cap: bool = False
    # Keep the coding-worker default for compatibility, while allowing another
    # worker type to remain distinguishable in the shared usage/audit trail.
    task_kind: str = WORKER_TASK_KIND

    def _agent_for(self, agent_name: str | None) -> Agent:
        if agent_name is None:
            # Only an identity-less record (pre-Phase 5 approval/checkpoint)
            # may fall back to the boot-time owner.
            return self.agents[self.default_agent]
        agent = self.agents.get(agent_name)
        if agent is None:
            # An asserted identity that no longer resolves (agent removed or
            # renamed since approval) must not inherit the default agent's
            # caps or attribution — no principal, no budget, no attempt.
            raise WorkerBudgetExceeded(
                f"worker attempt asserts unknown agent {agent_name!r} — "
                "failing closed (no attempt, no push, no PR)"
            )
        return agent

    def per_task_usd_for(self, agent_name: str | None) -> float | None:
        """The per-task cap the ledger would enforce for this agent."""
        return self._agent_for(agent_name).spec.budget.per_task_usd

    def _missing_cap_reason(self, agent: Agent) -> str | None:
        if (
            self.require_per_task_cap
            and agent.spec.budget.per_task_usd is None
        ):
            return (
                f"agent {agent.metadata.name} has no per-task spend cap, "
                "required by this worker backend"
            )
        return None

    async def check_monthly(
        self,
        agent_name: str | None,
        *,
        job_id: str,
        approval_id: str | None = None,
        approver: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Refuse the attempt if the invoking agent's monthly budget is spent.

        Runs *before* the attempt does any work (no credential resolve, no
        clone, no model spend). Reuses :func:`check_budget`, so ``on_exceeded:
        warn`` logs and proceeds — the ``block | warn`` semantics are the
        budget's, not the ledger's. A blocked attempt is recorded (zero cost,
        ``outcome="blocked"``) so the refusal is visible in ``/audit`` — and it
        carries the full attribution envelope, because a refusal is the *only*
        audit row that attempt ever writes (it then raises), so dropping the
        approval/approver/session here would lose them permanently.
        """
        agent = self._agent_for(agent_name)
        envelope = dict(
            job_id=job_id,
            approval_id=approval_id,
            approver=approver,
            session_id=session_id,
        )
        if reason := self._missing_cap_reason(agent):
            await self._record(agent, cost_usd=0.0, outcome="blocked", **envelope)
            raise WorkerBudgetExceeded(
                f"worker job {job_id} refused for agent {agent.metadata.name}: "
                f"{reason} — failing closed (no attempt, no push, no PR)"
            )

        decision = await check_budget(agent, self.usage)
        if decision.allowed:
            return
        await self._record(agent, cost_usd=0.0, outcome="blocked", **envelope)
        raise WorkerBudgetExceeded(
            f"worker job {job_id} refused for agent {agent.metadata.name}: "
            f"{decision.reason} — failing closed (no attempt, no push, no PR)"
        )

    async def settle(
        self,
        *,
        agent: str | None = None,
        job_id: str,
        idempotency_key: str | None = None,
        cost_usd: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        record_cost_usd: float | None = None,
        record_prompt_tokens: int | None = None,
        record_completion_tokens: int | None = None,
        cap_cost_usd: float | None = None,
        approval_id: str | None = None,
        approver: str | None = None,
        session_id: str | None = None,
        broker_job_id: str | None = None,
        broker_generation: int | None = None,
    ) -> None:
        """Record one segment delta and cap against cumulative task spend.

        The legacy ``cost_usd``/token inputs remain accepted as a one-segment
        task. Cold resume supplies the explicit ``record_*`` deltas plus
        ``cap_cost_usd`` cumulative total in the same call, preventing either
        view from drifting away from the other.
        """
        record_cost = record_cost_usd if record_cost_usd is not None else cost_usd
        record_prompt = (
            record_prompt_tokens
            if record_prompt_tokens is not None
            else (prompt_tokens or 0)
        )
        record_completion = (
            record_completion_tokens
            if record_completion_tokens is not None
            else (completion_tokens or 0)
        )
        if record_cost is None:
            raise ValueError("worker spend settlement requires record cost")
        cap_cost = cap_cost_usd if cap_cost_usd is not None else record_cost
        if min(record_cost, record_prompt, record_completion, cap_cost) < 0:
            raise ValueError("worker spend settlement cannot be negative")
        attributed = self._agent_for(agent)
        per_task_usd = attributed.spec.budget.per_task_usd
        if reason := self._missing_cap_reason(attributed):
            await self._record(
                attributed,
                cost_usd=record_cost,
                prompt_tokens=record_prompt,
                completion_tokens=record_completion,
                outcome="blocked",
                idempotency_key=idempotency_key,
                job_id=job_id,
                approval_id=approval_id,
                approver=approver,
                session_id=session_id,
                broker_job_id=broker_job_id,
                broker_generation=broker_generation,
            )
            raise WorkerBudgetExceeded(
                f"worker job {job_id} spent ${cap_cost:.4f}, but {reason} "
                "— failing closed (no push, no PR)"
            )

        over = per_task_usd is not None and cap_cost > per_task_usd
        await self._record(
            attributed,
            cost_usd=record_cost,
            prompt_tokens=record_prompt,
            completion_tokens=record_completion,
            outcome="over_task_budget" if over else "ok",
            idempotency_key=idempotency_key,
            job_id=job_id,
            approval_id=approval_id,
            approver=approver,
            session_id=session_id,
            broker_job_id=broker_job_id,
            broker_generation=broker_generation,
        )
        if over:
            raise WorkerBudgetExceeded(
                f"worker job {job_id} spent ${cap_cost:.4f}, over agent "
                f"{attributed.metadata.name}'s ${per_task_usd:.2f} per-task "
                "budget — failing closed (no push, no PR)"
            )
        logger.debug(
            "worker job %s segment spend recorded: $%.4f (cumulative $%.4f) "
            "against %s",
            job_id,
            record_cost,
            cap_cost,
            budget_scope_key(attributed),
        )

    async def _record(
        self,
        agent: Agent,
        *,
        cost_usd: float,
        outcome: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        idempotency_key: str | None = None,
        job_id: str | None = None,
        approval_id: str | None = None,
        approver: str | None = None,
        session_id: str | None = None,
        broker_job_id: str | None = None,
        broker_generation: int | None = None,
    ) -> None:
        await self.usage.record(
            UsageRecord(
                scope_key=budget_scope_key(agent),
                workspace=agent.metadata.workspace,
                agent=agent.metadata.name,
                model=self.model,
                task_kind=self.task_kind,
                idempotency_key=idempotency_key,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                outcome=outcome,
                # Attribution envelope (finding 4) — trace a worker charge to the
                # job, the approval that authorized it, and its origin session.
                job_id=job_id,
                broker_job_id=broker_job_id,
                broker_generation=broker_generation,
                approval_id=approval_id,
                approver=approver,
                session_id=session_id,
            )
        )
