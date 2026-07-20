"""The async task pipeline.

An inbound mention becomes a :class:`Task`. The runtime enforces throughput
limits and budget, recalls channel memory, resolves a model, then runs a
tool-calling loop: the
model may call tools the agent is allowed; the gateway enforces the allowlist
and routes write actions through human approval; results feed back until the
model produces a final answer. Usage is recorded and the exchange remembered.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openloop.agents.schema import Agent
from openloop.memory import (
    Embedder,
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
    scope_key_for,
)
from openloop.models.gateway import ModelGateway, ModelResponse
from openloop.tools import ToolGateway, ToolResult
from openloop.usage import (
    InMemoryTaskLimiter,
    InMemoryUsageStore,
    TaskLimiter,
    UsageRecord,
    UsageStore,
    budget_scope_key,
    check_budget,
    limit_scope_key,
)

if TYPE_CHECKING:
    from openloop.workflows.engine import WorkflowContext, WorkflowEngine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are {name}, a team AI agent operating in the {workspace} workspace. "
    "You are reachable across shared channels and act on behalf of the team. "
    "Be concise and helpful. When unsure, ask a clarifying question."
)

# Grounding facts the model cannot infer from its context: tools are its only
# reach into external systems, and the loop expires after MAX_TOOL_ITERS model
# turns. Stating both makes honest refusal the default behavior instead of a
# model virtue. Appended only when the agent actually has tools — a capability
# claim about tools that don't exist would itself be an invitation to invent.
TOOL_FACTS = (
    "Your tools are your only access to external systems and live data. "
    "If no available tool fits a request, say plainly what you cannot do — "
    "never invent data, links, or tool results. You have at most {max_iters} "
    "model turns per task, and your final answer must fit within them, so "
    "batch related tool calls and answer as soon as you have what you need."
)

# Per-surface output hints appended to the system prompt. These shape *what*
# the model writes for the destination (length, structure); rendering the
# model's standard Markdown is the delivery layer's job. Slack's server-side
# conversion has no table support and flattens heading hierarchies, so the
# hint steers away from those shapes — it must NOT ask for Slack's mrkdwn
# dialect, which models drift out of and which would pollute history/memory.
SURFACE_HINTS = {
    "slack": (
        "You are replying in a Slack thread. Keep replies brief and "
        "conversational. Prefer short bullet lists and bold key phrases over "
        "headings. Never use Markdown tables — use a bulleted list instead. "
        "Use code blocks for code or command output."
    ),
}

# How many memories to pull into context per task.
RECALL_LIMIT = 5
# Safety cap on model<->tool round-trips per task.
MAX_TOOL_ITERS = 4
# Safety cap on a single tool result fed back to the model. Connectors are
# expected to return already-trimmed `data`; this guards the context (and the
# per-task budget) against any that don't.
TOOL_RESULT_MAX_CHARS = 6000


def _tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# --- turn-scoped replay cursor (M0a) ---
#
# In a threaded conversation `messages` is prefixed with prior delivered turns, so
# the replay cursor (what has this turn already done?) must read only the CURRENT
# turn's slice — everything from its user message onward. A whole-log cursor would
# see the previous turn's final answer and treat the new turn as already finished.
# The model still receives the full `messages`; only the cursor is turn-scoped.


def _turn_start(messages: list[dict]) -> int:
    """Index of the current turn's user message (the last user-role message)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return 0


def _turn_slice(messages: list[dict]) -> list[dict]:
    return messages[_turn_start(messages):]


def _trailing_assistant(turn: list[dict]) -> dict | None:
    for msg in reversed(turn):
        if msg.get("role") == "assistant":
            return msg
    return None


def _executed_ids(turn: list[dict]) -> set[str]:
    return {
        m["tool_call_id"]
        for m in turn
        if m.get("role") == "tool" and m.get("tool_call_id")
    }


def _rounds_used(turn: list[dict]) -> int:
    return sum(1 for m in turn if m.get("role") == "assistant")


def _unresolved_tool_calls(turn: list[dict]) -> list[dict]:
    """Tool-call dicts of the trailing assistant round that lack a tool result."""
    last = _trailing_assistant(turn)
    if last is None or not last.get("tool_calls"):
        return []
    executed = _executed_ids(turn)
    return [tc for tc in last["tool_calls"] if tc.get("id") not in executed]


def _parse_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _result_content(result: ToolResult | None) -> str:
    """Serialize an executed tool's result for the model.

    The model must see `data`, not just the summary — answering from a
    data-free "read issue #7" summary is how fabricated answers happen.
    """
    if result is None:
        return "done"
    payload: dict = {"ok": result.ok, "summary": result.summary}
    if result.data:
        payload["data"] = result.data
    content = json.dumps(payload, default=str)
    if len(content) > TOOL_RESULT_MAX_CHARS:
        content = content[:TOOL_RESULT_MAX_CHARS] + "… [truncated]"
    return content


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
    # The requesting thread's durable scope key (Phase B). Set by the session
    # runner for threaded turns; threaded down to a workflow-backed tool so it can
    # reuse the thread's warm execution context — and, since Phase 4, stamped by
    # the gateway as the trusted request scope upload provisioning is checked
    # against. None for non-threaded turns.
    thread_key: str | None = None
    # The originating surface session's id (SurfaceSession.id), stamped by the
    # session runner. Threaded down to a workflow-backed tool so worker spend
    # attributes to the session it was invoked from (UsageRecord.session_id).
    # Distinct from thread_key, which is a workspace-reuse key, not an identity.
    # None for paths with no session (direct runtime.handle, tests).
    session_id: str | None = None
    # Trusted surrounding context the surface/runner supplies (e.g. the
    # thread's shared-file inventory), rendered as one system message.
    context_notes: list[str] = field(default_factory=list)


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
        engine: "WorkflowEngine",
        remember: bool = True,
        limiter: TaskLimiter | None = None,
    ) -> None:
        self.agent = agent
        self.gateway = gateway or ModelGateway()
        self.memory = memory or InMemoryStore()
        self.embedder = embedder
        self.usage = usage or InMemoryUsageStore()
        self.tools = tools
        self.remember = remember
        # Phase 5 throughput limits: per-(tenant, agent) rate/concurrency,
        # enforced at handle() before any other work. Config: spec.limits.
        self.limiter = limiter or InMemoryTaskLimiter()
        # Every task runs as an `agent_task` workflow. The caller chooses the
        # engine's store explicitly: Postgres for durable composition, or an
        # in-memory store for tests / deliberately non-durable local development.
        # Namespaced per agent so multiple agents on one engine don't collide.
        self.engine = engine
        self.workflow_name = f"agent_task:{agent.metadata.name}"
        engine.register(self._build_workflow())

    def _build_messages(
        self, task: Task, recalled: list[MemoryRecord]
    ) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT.format(
            name=self.agent.metadata.name,
            workspace=self.agent.metadata.workspace,
        )
        if self.tools is not None and self.tools.tool_specs(self.agent).definitions:
            system = f"{system}\n\n{TOOL_FACTS.format(max_iters=MAX_TOOL_ITERS)}"
        hint = SURFACE_HINTS.get(task.surface)
        if hint:
            system = f"{system}\n\n{hint}"
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
        if task.context_notes:
            messages.append(
                {"role": "system", "content": "\n\n".join(task.context_notes)}
            )
        messages.extend(task.history)
        messages.append({"role": "user", "content": task.text})
        return messages

    async def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        vectors = await self.embedder.embed([text])
        return vectors[0] if vectors else None

    async def handle(
        self, task: Task, *, instance_id: str | None = None
    ) -> ModelResponse:
        # Throughput guard first — cheaper than the budget check and refused
        # tasks must not consume a model call, a memory hit, or a usage row
        # beyond the audit record of the refusal itself.
        scope = limit_scope_key(self.agent)
        decision = await self.limiter.acquire(scope, self.agent.spec.limits)
        if not decision.allowed:
            logger.warning("rate-limited task for %s: %s", scope, decision.reason)
            model = self.agent.model_for(task.kind)
            await self._record_usage(
                task, model, ModelResponse(text="", model=model),
                outcome="rate_limited",
            )
            return _limited_response(decision.reason)
        try:
            return await self._handle_workflow(task, instance_id)
        finally:
            await self.limiter.release(scope)

    def _accounted_from_state(self, state: dict) -> ModelResponse:
        """The accumulated ModelResponse for usage accounting, from turn state."""
        usage = state.get("usage_total") or {}
        return ModelResponse(
            text=state.get("final_text") or "",
            model=usage.get("model") or state.get("model", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost_usd=usage.get("cost_usd", 0.0),
        )

    def _final_from_state(self, state: dict) -> ModelResponse:
        """The user-facing ModelResponse (approval-gate model tag if parked)."""
        accounted = self._accounted_from_state(state)
        return _final_response(
            state.get("final_text") or "",
            accounted,
            state.get("approval_ids", []),
            accounted.model,
        )

    # --- shared phases (used by both the inline and workflow paths) ---

    async def _prepare(
        self, task: Task
    ) -> tuple[str, str, list[dict] | None, list[float] | None, str | None]:
        """Resolve model, enforce budget, recall memory, build messages.

        Returns ``(model, scope, messages, query_embedding, block_reason)``;
        a non-None ``block_reason`` means the budget guard tripped (no model call).
        """
        model = self.agent.model_for(task.kind)
        scope = scope_key_for(self.agent, task.channel)
        logger.info("routing task on %s/%s -> %s", task.surface, task.channel, model)

        decision = await check_budget(self.agent, self.usage)
        if not decision.allowed:
            logger.warning("blocked task for %s: %s", scope, decision.reason)
            return model, scope, None, None, decision.reason

        # Embed the request once and reuse the vector when remembering.
        query_embedding = await self._embed(task.text)
        recalled = await self.memory.recall(scope, query_embedding, limit=RECALL_LIMIT)
        if recalled:
            logger.info("recalled %d memory item(s) for %s", len(recalled), scope)
        messages = self._build_messages(task, recalled)
        return model, scope, messages, query_embedding, None

    async def _remember(
        self, task: Task, scope: str, query_embedding: list[float] | None
    ) -> None:
        await self.memory.remember(
            MemoryRecord(
                scope_key=scope,
                text=task.text,
                kind="message",
                metadata={"user": task.user or "", "surface": task.surface},
                embedding=query_embedding,
            )
        )

    def _log_completion(
        self, task: Task, accounted: ModelResponse, outcome: str
    ) -> None:
        logger.info(
            "completed task on %s/%s with %s (%d+%d tok, $%.4f, %s)",
            task.surface, task.channel, accounted.model,
            accounted.prompt_tokens, accounted.completion_tokens,
            accounted.cost_usd, outcome,
        )

    # --- durable workflow path (consumer #2) ---

    def _build_workflow(self):
        # Imported here to avoid a cycle (engine has no runtime dependency).
        from openloop.workflows.engine import Step, Workflow

        return Workflow(
            self.workflow_name,
            [
                Step("prepare", self._wf_prepare),
                # M0a: `run` is resumable because the tool loop is checkpoint-driven
                # and resume-safe — it derives position from the committed `messages`
                # log, so a resume never re-issues a recorded model round or
                # re-executes a recorded tool call (see `_run_tool_loop`).
                Step("run", self._wf_run, resumable=True),
                Step("persist", self._wf_persist),
            ],
        )

    async def _handle_workflow(
        self, task: Task, instance_id: str | None = None
    ) -> ModelResponse:
        # A caller (e.g. the Phase D session runner) can bind the workflow
        # instance to its own id so the two share one identity; otherwise mint one.
        instance = await self.engine.start(
            self.workflow_name,
            instance_id or uuid.uuid4().hex,
            {"task": _task_to_dict(task)},
        )
        return self._response_from(instance)

    async def continue_turn(
        self, task: Task, messages: list[dict], *, instance_id: str
    ) -> ModelResponse:
        """Re-run the model on a pre-seeded message log (M0b continuation).

        Used after a human approval: the caller folds the approved tool result into
        the held round and hands the log here. Prepare skips message-building (the
        log is authoritative) but still enforces budget; the resume-aware loop sees
        the round now resolved and produces a fresh answer. Idempotent on
        ``instance_id`` (deterministic per session+approval), so a re-spawn returns
        the existing continuation instead of re-driving the model.
        """
        model = self.agent.model_for(task.kind)
        scope = scope_key_for(self.agent, task.channel)
        instance = await self.engine.start(
            self.workflow_name,
            instance_id,
            {
                "task": _task_to_dict(task), "model": model, "scope": scope,
                "messages": messages, "query_embedding": None, "continuation": True,
            },
        )
        return self._response_from(instance)

    async def _wf_prepare(self, ctx: "WorkflowContext") -> None:
        s = ctx.state
        task = _task_from_dict(s["task"])
        if s.get("continuation"):
            # Continuation: model/scope/messages are pre-seeded and authoritative —
            # only re-enforce the budget guard (a continuation still spends).
            decision = await check_budget(self.agent, self.usage)
            if not decision.allowed:
                s.update({"blocked": True, "block_reason": decision.reason})
            return
        model, scope, messages, query_embedding, block_reason = await self._prepare(task)
        s.update({"model": model, "scope": scope})
        if block_reason is not None:
            s.update({"blocked": True, "block_reason": block_reason})
            return
        # Persisted turn state: messages (system+history+user), recall vector.
        s.update({"messages": messages, "query_embedding": query_embedding})

    async def _wf_run(self, ctx: "WorkflowContext") -> None:
        s = ctx.state
        if s.get("blocked"):
            return
        task = _task_from_dict(s["task"])
        # Checkpoint-driven and resume-safe: the loop persists `messages`,
        # `usage_total`, `final_text`, and `approval_ids` into `s` via ctx.checkpoint
        # as it goes, so a re-drive of this step continues from the committed log.
        await self._run_tool_loop(s, task, checkpoint=ctx.checkpoint)

    async def _wf_persist(self, ctx: "WorkflowContext") -> None:
        s = ctx.state
        task = _task_from_dict(s["task"])
        if s.get("blocked"):
            if not s.get("usage_recorded"):
                await self._record_usage(
                    task, s["model"], ModelResponse(text="", model=s["model"]),
                    outcome="blocked",
                )
                s["usage_recorded"] = True
            return

        accounted = self._accounted_from_state(s)
        # Idempotent writes: flags guard against a resumed persist double-writing. A
        # continuation must NOT re-remember (the original turn already did) but DOES
        # record its own new model spend.
        if self.remember and not s.get("remembered") and not s.get("continuation"):
            await self._remember(task, s["scope"], s.get("query_embedding"))
            s["remembered"] = True
            await ctx.checkpoint()
        if not s.get("usage_recorded"):
            outcome = self._task_outcome(accounted)
            await self._record_usage(task, s["model"], accounted, outcome=outcome)
            s["usage_recorded"] = True
            self._log_completion(task, accounted, outcome)

    async def recover_response(
        self, instance_id: str
    ) -> tuple[bool, "ModelResponse | None"]:
        """For the Phase D session reconciler: recover a crashed turn's response
        from its persisted workflow, **without** re-running it.

        Returns ``(found, response)``:

        - ``(False, None)`` — no engine, or the instance is gone: unrecoverable,
          so the reconciler should post an interrupted notice.
        - ``(True, None)`` — the instance exists but is **not terminal** (still
          running/waiting, e.g. the engine's own resume hasn't finished or failed):
          leave it for a later restart rather than delivering a half-finished turn.
        - ``(True, response)`` — terminal: the answer (``completed``) or an
          interrupted notice (``failed`` / ``cancelled`` / ``abandoned``).
        """
        from openloop.workflows.store import TERMINAL as WF_TERMINAL

        instance = await self.engine.store.get(instance_id)
        if instance is None:
            return False, None
        if instance.status not in WF_TERMINAL:
            return True, None
        if instance.status in ("failed", "cancelled", "abandoned"):
            return True, _interrupted_response()
        return True, self._response_from(instance)

    def _response_from(self, instance) -> ModelResponse:
        s = instance.state
        if s.get("blocked"):
            return _blocked_response(s["block_reason"])
        if instance.status in ("failed", "abandoned"):
            return _interrupted_response()
        return self._final_from_state(s)

    async def _run_tool_loop(self, state: dict, task: Task, *, checkpoint) -> None:
        """Drive model<->tool round-trips, checkpoint-driven and resume-safe (M0a).

        Reads and writes ``state`` in place — ``messages`` (the durable round log),
        ``usage_total`` (summed tokens/cost), ``final_text``, ``approval_ids`` — and
        calls ``checkpoint`` after each committed model round and tool result. Loop
        position is derived from the current turn's slice of ``messages`` (see the
        turn-cursor helpers), so a resume never re-issues a recorded model round or
        re-executes a recorded tool call. ``checkpoint`` is a no-op on the inline
        path and the workflow engine's checkpoint on the durable path.

        Ends the turn on a final answer, budget exhaustion (a synthesized answer is
        committed so the log stays complete), or a human-approval gate (M0a records
        the approval and stops; the continuation that re-runs the model is M0b).
        """
        specs = self.tools.tool_specs(self.agent) if self.tools else None
        tool_defs = specs.definitions if specs and specs.definitions else None
        by_name = specs.by_name if specs else {}

        model = state["model"]
        messages = state["messages"]
        usage = state.setdefault(
            "usage_total",
            {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        )

        while True:
            # (T) TERMINAL FIRST — a committed final answer (or approval end) ends the
            # turn, at any rounds_used. Covers the crash window where the answer was
            # logged/persisted but the run step wasn't yet marked complete.
            if state.get("final_text") is not None:
                break
            turn = _turn_slice(messages)
            last = _trailing_assistant(turn)
            if last is not None and not last.get("tool_calls"):
                state["final_text"] = last.get("content") or ""
                break

            pending = _unresolved_tool_calls(turn)
            if not pending:
                # Would start a NEW model round (turn start, or round fully resolved).
                if _rounds_used(turn) >= MAX_TOOL_ITERS:  # (G) budget stops a new round only
                    messages.append({
                        "role": "assistant",
                        "content": "I couldn't finish that within the tool-call limit.",
                    })
                    state["final_text"] = messages[-1]["content"]
                    await checkpoint()
                    break
                response = await self.gateway.complete(model, messages, tools=tool_defs)
                usage["prompt_tokens"] += response.prompt_tokens
                usage["completion_tokens"] += response.completion_tokens
                usage["cost_usd"] += response.cost_usd
                usage["model"] = response.model or model
                messages.append(
                    response.raw_message
                    or {"role": "assistant", "content": response.text}
                )
                await checkpoint()  # (B) COMMIT the round BEFORE any tool runs
                if not response.tool_calls:
                    state["final_text"] = response.text
                    break
                continue  # re-derive pending from the just-appended assistant round

            # (C) Execute this round's unresolved tool calls (budget can't abandon them).
            for tc in pending:
                call_id = tc.get("id")
                fn = tc.get("function", {})
                call_name = fn.get("name", "")
                action = by_name.get(call_name)
                if action is None:
                    messages.append(_tool_message(call_id, f"error: unknown tool {call_name}"))
                    await checkpoint()
                    continue
                inv = await self.tools.invoke(
                    self.agent, action, _parse_json(fn.get("arguments")),
                    requested_by=task.user,
                    warm_key=task.thread_key,
                    session_id=task.session_id,
                )
                if inv.status == "executed":
                    messages.append(_tool_message(call_id, _result_content(inv.result)))
                elif inv.status == "pending_approval":
                    if inv.approval is not None:
                        state.setdefault("approval_ids", []).append(inv.approval.id)
                        # Map approval -> the tool call it gates, so a continuation
                        # (M0b) can fold the approved result into this exact call's
                        # held message and re-run the model.
                        state.setdefault("approval_calls", {})[inv.approval.id] = call_id
                    note = inv.message or "approval required"
                    prev = state.get("final_text")
                    state["final_text"] = f"{prev}\n{note}" if prev else note
                    messages.append(
                        _tool_message(call_id, f"held for human approval: {inv.message}")
                    )
                else:  # forbidden / denied
                    messages.append(_tool_message(call_id, f"{inv.status}: {inv.message}"))
                await checkpoint()  # (D) commit each tool result / approval hold
            if state.get("approval_ids"):
                # M0a: approval ends the turn (M0b adds the continuation post-Phase-C).
                break
            # Round fully resolved with no approval → loop re-derives the next round.

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


def _blocked_response(reason: str) -> ModelResponse:
    return ModelResponse(
        text=f"💸 Budget guard: {reason}. Action blocked.", model="budget-guard"
    )


def _limited_response(reason: str) -> ModelResponse:
    return ModelResponse(
        text=f"🚦 Throughput guard: {reason}. Try again shortly.",
        model="throughput-guard",
    )


def _interrupted_response() -> ModelResponse:
    return ModelResponse(
        text="⚠️ This task was interrupted and could not be completed.",
        model="error",
    )


def _final_response(
    final_text: str,
    accounted: ModelResponse,
    approval_ids: list[str],
    model: str,
) -> ModelResponse:
    return ModelResponse(
        text=final_text,
        model="approval-gate" if approval_ids else (accounted.model or model),
        prompt_tokens=accounted.prompt_tokens,
        completion_tokens=accounted.completion_tokens,
        cost_usd=accounted.cost_usd,
        approval_ids=approval_ids,
    )


def _task_to_dict(task: Task) -> dict:
    return {
        "text": task.text,
        "surface": task.surface,
        "channel": task.channel,
        "user": task.user,
        "kind": task.kind,
        "history": task.history,
        # thread_key must survive the durable round-trip: the workflow path
        # re-hydrates the task before running the tool loop, and the gateway's
        # warm-context reuse AND the analysis upload scope stamp both ride it.
        "thread_key": task.thread_key,
        # session_id must survive the durable round-trip too: the workflow-backed
        # tool loop stamps it into the approval args so worker spend traces to it.
        "session_id": task.session_id,
        "context_notes": task.context_notes,
    }


def _task_from_dict(data: dict) -> Task:
    return Task(
        text=data["text"],
        surface=data["surface"],
        channel=data.get("channel"),
        user=data.get("user"),
        kind=data.get("kind"),
        history=data.get("history", []),
        thread_key=data.get("thread_key"),
        session_id=data.get("session_id"),
        context_notes=data.get("context_notes", []),
    )

