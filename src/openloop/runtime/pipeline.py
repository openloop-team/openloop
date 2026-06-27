"""The async task pipeline.

An inbound mention becomes a :class:`Task`. The runtime enforces budget,
recalls channel memory, resolves a model, then runs a tool-calling loop: the
model may call tools the agent is allowed; the gateway enforces the allowlist
and routes write actions through human approval; results feed back until the
model produces a final answer. Usage is recorded and the exchange remembered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from openloop.agents.schema import Agent
from openloop.memory import (
    Embedder,
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
    scope_key_for,
)
from openloop.models.gateway import ModelGateway, ModelResponse
from openloop.tools import ToolGateway
from openloop.usage import (
    InMemoryUsageStore,
    UsageRecord,
    UsageStore,
    budget_scope_key,
    check_budget,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are {name}, a team AI agent operating in the {workspace} workspace. "
    "You are reachable across shared channels and act on behalf of the team. "
    "Be concise and helpful. When unsure, ask a clarifying question."
)

# How many memories to pull into context per task.
RECALL_LIMIT = 5
# Safety cap on model<->tool round-trips per task.
MAX_TOOL_ITERS = 4


def _tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


@dataclass(slots=True)
class Task:
    """A unit of work routed to an agent from some surface."""

    text: str
    surface: str
    channel: str | None = None
    user: str | None = None
    # Optional task class (e.g. "summarize", "code") used for model routing.
    kind: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)


class Runtime:
    """Routes a task to a model and produces a reply, with channel memory."""

    def __init__(
        self,
        agent: Agent,
        gateway: ModelGateway | None = None,
        memory: MemoryStore | None = None,
        embedder: Embedder | None = None,
        usage: UsageStore | None = None,
        tools: ToolGateway | None = None,
        *,
        remember: bool = True,
    ) -> None:
        self.agent = agent
        self.gateway = gateway or ModelGateway()
        self.memory = memory or InMemoryStore()
        self.embedder = embedder
        self.usage = usage or InMemoryUsageStore()
        self.tools = tools
        self.remember = remember

    def _build_messages(
        self, task: Task, recalled: list[MemoryRecord]
    ) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT.format(
            name=self.agent.metadata.name,
            workspace=self.agent.metadata.workspace,
        )
        messages = [{"role": "system", "content": system}]
        if recalled:
            bullets = "\n".join(f"- {r.text}" for r in recalled)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant team memory for this channel "
                        "(most relevant first):\n" + bullets
                    ),
                }
            )
        messages.extend(task.history)
        messages.append({"role": "user", "content": task.text})
        return messages

    async def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        vectors = await self.embedder.embed([text])
        return vectors[0] if vectors else None

    async def handle(self, task: Task) -> ModelResponse:
        model = self.agent.model_for(task.kind)
        scope = scope_key_for(self.agent, task.channel)
        logger.info("routing task on %s/%s -> %s", task.surface, task.channel, model)

        # Budget: gate on accumulated monthly spend before any model call.
        decision = await check_budget(self.agent, self.usage)
        if not decision.allowed:
            logger.warning("blocked task for %s: %s", scope, decision.reason)
            await self._record_usage(task, model, ModelResponse(text="", model=model),
                                     outcome="blocked")
            return ModelResponse(
                text=f"💸 Budget guard: {decision.reason}. Action blocked.",
                model="budget-guard",
            )

        # Recall: embed the request once and reuse the vector when remembering.
        query_embedding = await self._embed(task.text)
        recalled = await self.memory.recall(
            scope, query_embedding, limit=RECALL_LIMIT
        )
        if recalled:
            logger.info("recalled %d memory item(s) for %s", len(recalled), scope)

        messages = self._build_messages(task, recalled)
        final_text, accounted, approval_ids = await self._run_tool_loop(
            model, messages, task
        )

        if self.remember:
            await self.memory.remember(
                MemoryRecord(
                    scope_key=scope,
                    text=task.text,
                    kind="message",
                    metadata={"user": task.user or "", "surface": task.surface},
                    embedding=query_embedding,
                )
            )

        outcome = self._task_outcome(accounted)
        await self._record_usage(task, model, accounted, outcome=outcome)
        logger.info(
            "completed task on %s/%s with %s (%d+%d tok, $%.4f, %s)",
            task.surface,
            task.channel,
            accounted.model,
            accounted.prompt_tokens,
            accounted.completion_tokens,
            accounted.cost_usd,
            outcome,
        )
        return ModelResponse(
            text=final_text,
            model="approval-gate" if approval_ids else accounted.model,
            prompt_tokens=accounted.prompt_tokens,
            completion_tokens=accounted.completion_tokens,
            cost_usd=accounted.cost_usd,
            approval_ids=approval_ids,
        )

    async def _run_tool_loop(
        self, model: str, messages: list[dict], task: Task
    ) -> tuple[str, ModelResponse, list[str]]:
        """Drive model<->tool round-trips until a final answer or an approval.

        Returns the user-facing text, an accumulated ModelResponse for usage
        accounting (real model + summed tokens/cost), and the IDs of any write
        actions left awaiting human approval.
        """
        specs = self.tools.tool_specs(self.agent) if self.tools else None
        tool_defs = specs.definitions if specs and specs.definitions else None
        by_name = specs.by_name if specs else {}

        total_cost = 0.0
        total_pt = total_ct = 0
        final_model = model
        final_text = ""
        approval_messages: list[str] = []
        approval_ids: list[str] = []
        response = None

        for _ in range(MAX_TOOL_ITERS):
            response = await self.gateway.complete(model, messages, tools=tool_defs)
            total_cost += response.cost_usd
            total_pt += response.prompt_tokens
            total_ct += response.completion_tokens
            final_model = response.model or model

            if not response.tool_calls:
                final_text = response.text
                break

            messages.append(
                response.raw_message
                or {"role": "assistant", "content": response.text}
            )
            stop_for_approval = False
            for call in response.tool_calls:
                action = by_name.get(call.name)
                if action is None:
                    messages.append(
                        _tool_message(call.id, f"error: unknown tool {call.name}")
                    )
                    continue
                inv = await self.tools.invoke(
                    self.agent, action, call.arguments, requested_by=task.user
                )
                if inv.status == "executed":
                    summary = inv.result.summary if inv.result else "done"
                    messages.append(_tool_message(call.id, summary))
                elif inv.status == "pending_approval":
                    approval_messages.append(inv.message or "approval required")
                    if inv.approval is not None:
                        approval_ids.append(inv.approval.id)
                    messages.append(
                        _tool_message(call.id, f"held for human approval: {inv.message}")
                    )
                    stop_for_approval = True
                else:  # forbidden / denied
                    messages.append(
                        _tool_message(call.id, f"{inv.status}: {inv.message}")
                    )
            if stop_for_approval:
                break
        else:
            final_text = (
                (response.text if response else "")
                or "I couldn't finish that within the tool-call limit."
            )

        if approval_messages:
            final_text = "\n".join(approval_messages)

        accounted = ModelResponse(
            text=final_text,
            model=final_model,
            prompt_tokens=total_pt,
            completion_tokens=total_ct,
            cost_usd=total_cost,
        )
        return final_text, accounted, approval_ids

    def _task_outcome(self, response: ModelResponse) -> str:
        per_task = self.agent.spec.budget.per_task_usd
        if per_task is not None and response.cost_usd > per_task:
            logger.warning(
                "task cost $%.4f exceeded per-task budget $%.4f",
                response.cost_usd,
                per_task,
            )
            return "over_task_budget"
        return "ok"

    async def _record_usage(
        self, task: Task, model: str, response: ModelResponse, *, outcome: str
    ) -> None:
        await self.usage.record(
            UsageRecord(
                scope_key=budget_scope_key(self.agent),
                workspace=self.agent.metadata.workspace,
                agent=self.agent.metadata.name,
                model=response.model or model,
                channel=task.channel,
                surface=task.surface,
                user=task.user,
                task_kind=task.kind,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd,
                outcome=outcome,
            )
        )
