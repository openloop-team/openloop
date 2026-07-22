"""The tool gateway: enforce the allowlist, gate writes on approval, execute."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openloop.agents.schema import Agent
from openloop.approvals.store import (
    ApprovalRequest,
    ApprovalStore,
    InMemoryApprovalStore,
)
from openloop.tools.base import (
    Invocation,
    Tool,
    ToolResult,
    format_validation_error,
    split_action,
    validate_args,
)
from openloop.tools.policy import is_allowed
from openloop.workflows.store import TERMINAL as WORKFLOW_TERMINAL

if TYPE_CHECKING:
    from openloop.tools.workspace_pool import WarmWorkspacePool
    from openloop.workflows.engine import WorkflowEngine

logger = logging.getLogger(__name__)

# Function names the model sees must match ^[A-Za-z0-9_-]+$, but action names
# use "." and ":". Encode for the wire, keep an exact reverse map per request.
_FN_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _fn_name(action: str) -> str:
    return _FN_SAFE.sub("_", action)


@dataclass(slots=True)
class ToolSpecs:
    """OpenAI/LiteLLM function definitions plus the name→action reverse map."""

    definitions: list[dict] = field(default_factory=list)
    by_name: dict[str, str] = field(default_factory=dict)


class ToolGateway:
    """Routes tool actions through policy and approval before execution."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        approvals: ApprovalStore | None = None,
        engine: "WorkflowEngine | None" = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)
        self.approvals = approvals or InMemoryApprovalStore()
        # Optional: when set, a tool that declares a `workflow` runs as a durable
        # workflow — approval becomes a wait node and resolve() emits the event.
        self.engine = engine
        # Optional Phase B warm-workspace pool (set during app wiring). Held here
        # so the app can reach it to wire its durable sink + lifecycle without a
        # separate registry; the orchestrator uses it directly.
        self.warm_pool: "WarmWorkspacePool | None" = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def available_actions(self, agent: Agent) -> list[str]:
        """Allowed actions that also have a registered tool to run them."""
        out = []
        for tool in agent.spec.tools:
            impl = self._tools.get(tool.name)
            if impl is None:
                continue
            for perm in tool.permissions:
                if perm in impl.supported_permissions():
                    out.append(f"{tool.name}.{perm}")
        return sorted(out)

    def tool_specs(self, agent: Agent) -> ToolSpecs:
        """Function-calling definitions for the agent's available actions."""
        specs = ToolSpecs()
        for action in self.available_actions(agent):
            tool_name, permission = split_action(action)
            spec = self._tools[tool_name].describe(permission)
            fname = _fn_name(action)
            specs.by_name[fname] = action
            specs.definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": fname,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
            )
        return specs

    async def invoke(
        self,
        agent: Agent,
        action: str,
        args: dict,
        *,
        requested_by: str | None = None,
        warm_key: str | None = None,
        session_id: str | None = None,
    ) -> Invocation:
        tool_name, permission = split_action(action)

        if not is_allowed(agent, action):
            return Invocation(
                status="forbidden",
                message=f"{action} is not in {agent.metadata.name}'s tool allowlist",
            )

        tool = self._tools.get(tool_name)
        if tool is None or permission not in tool.supported_permissions():
            return Invocation(
                status="forbidden",
                message=f"no registered tool provides {action}",
            )

        # Typed actions (docs/typed-tool-args.md §3.3) are PARSED, not
        # validated: raw args go through the pydantic model the declared
        # schema was generated from, before identity stamping and before
        # anything durable exists — only the parse's model_dump() can become a
        # record. Parse failure maps onto the same "invalid" status the model
        # can correct and retry.
        spec = tool.describe(permission)
        if spec.model is not None:
            try:
                parsed = spec.model.model_validate(args)
            except Exception as exc:
                detail = format_validation_error(exc)
                logger.info("rejected %s: invalid arguments (%s)", action, detail)
                return Invocation(
                    status="invalid",
                    message=f"invalid arguments for {action}: {detail}",
                )
            # exclude_none keeps optional-and-omitted fields absent from the
            # record, so an execute-time `args.get(key, default)` still
            # defaults instead of seeing an explicit None.
            args = parsed.model_dump(mode="json", exclude_none=True)

        # Let a tool finalize its args before they cross the approval boundary
        # (e.g. the coding worker mints a job_id here so it's persisted in the
        # approval request and reused verbatim at execute time, and stamps the
        # invoking agent so spend attribution survives the approval hop).
        prepare = getattr(tool, "prepare_args", None)
        if prepare is not None:
            # warm_key (Phase B) rides the args across the approval boundary so a
            # workflow-backed tool can reuse the requesting thread's warm context;
            # session_id (step 5) rides the same way so worker spend attributes to
            # the originating surface session. Both are gateway-supplied — a
            # model-supplied value is ignored.
            args = prepare(
                permission, args, agent, warm_key=warm_key, session_id=session_id
            )

        # Untyped actions (MCP passthrough) keep the permissive-subset seam:
        # enforce the action's own declared schema on the prepared args, before
        # anything durable exists. This is the one validation seam every path
        # shares — a workflow-backed tool never runs execute(), so per-connector
        # checks there cannot protect the durable path, and a request that can
        # never run must not become an approval card a human is asked to decide.
        if spec.model is None:
            problems = validate_args(spec.parameters, args)
            if problems:
                detail = "; ".join(problems)
                logger.info("rejected %s: invalid arguments (%s)", action, detail)
                return Invocation(
                    status="invalid",
                    message=f"invalid arguments for {action}: {detail}",
                )

        # Optional async, LOCAL-ONLY resolution step (no external fetch —
        # approve-before-work still holds): a connector can verify references
        # against its own stores and stamp trusted display metadata so the
        # approval card can truthfully name what it gates. A violation refuses
        # here as "invalid" — a human is never asked to approve a request
        # policy already forbids.
        resolve_args = getattr(tool, "resolve_args", None)
        if resolve_args is not None:
            args, problem = await resolve_args(permission, args)
            if problem is not None:
                logger.info("rejected %s: %s", action, problem)
                return Invocation(
                    status="invalid",
                    message=f"invalid arguments for {action}: {problem}",
                )

        # Connectors may require approval intrinsically regardless of an
        # accidental omission in an agent's config.
        if (
            agent.spec.approvals.requires_approval(action)
            or getattr(tool, "requires_approval", False)
        ):
            # The execution mode this invoke() commits to, recorded durably
            # with the row so every decided path routes on the marker — never
            # on the resolver's current engine/tool shape (mode drift there
            # means a double-run or a phantom instance). Keep the condition
            # visibly identical to _maybe_start_workflow's.
            workflow_backed = self.engine is not None and bool(
                getattr(tool, "workflow", None)
            )
            request = ApprovalRequest(
                agent=agent.metadata.name,
                action=action,
                tool=tool_name,
                permission=permission,
                args=args,
                approvers=list(agent.spec.approvals.approvers),
                requested_by=requested_by,
                summary=_summarize(action, args),
                # The args-contract version these args were parsed under, so
                # consumers can refuse the record after a breaking change.
                args_schema=spec.version,
                workflow_backed=workflow_backed,
            )
            if workflow_backed:
                request.workflow_instance_id = _instance_id(request)
            await self.approvals.create(request)
            # For a workflow-backed tool, start the workflow now; it parks on its
            # approval wait node. The approval event (from resolve) wakes it.
            await self._maybe_start_workflow(tool, request)
            logger.info("approval required for %s (id=%s)", action, request.id)
            return Invocation(
                status="pending_approval",
                approval=request,
                message=(
                    f"⏳ Write action ({request.summary}) — approval required. "
                    f"{', '.join(request.approvers)}: approve {request.id}?"
                ),
            )

        result = await tool.execute(permission, args)
        return Invocation(status="executed", result=result)

    async def resolve(
        self, request_id: str, approver: str, *, approve: bool
    ) -> Invocation:
        # Decide-first: claim_decision on the approval row is the single
        # arbiter — one claim wins, and every effect (wake, cancel, direct
        # execute) follows the durable decision. Crash windows between claim
        # and effect are healed by the idempotent re-ensure on the next click
        # and by reconcile_decisions() in the recovery pass.
        request = await self.approvals.get(request_id)
        if request is None:
            return Invocation(status="forbidden", message="no such approval request")
        # Membership before any claim OR decided-row handling: the decided
        # path performs effects (re-ensure start/send_event/cancel), so a
        # non-approver must get forbidden — never a decision report or an
        # effect trigger.
        if approver not in request.approvers:
            return Invocation(
                status="forbidden",
                message=f"{approver} is not an approver for {request.action}",
            )
        if request.status != "pending":
            return await self._decided_outcome(request)
        # Gate a NEW approve on effect availability before its claim: a claim
        # cannot be undone, and an approve claimed with no performable effect
        # here would land irreversibly in recovery. Deny is never gated — the
        # fail-closed action must always be available.
        if approve:
            unavailable = self._approve_effect_unavailable(request)
            if unavailable is not None:
                return Invocation(
                    status="forbidden",
                    message=(
                        f"cannot approve {request.id} on this gateway: "
                        f"{unavailable}; the request stays pending"
                    ),
                )

        claimed = await self.approvals.claim_decision(
            request_id, approver, approve=approve
        )
        if claimed is None:
            # Lost the race (or decided between get and claim) — the stored
            # decision is the truth; report it, never this caller's intent.
            fresh = await self.approvals.get(request_id)
            if fresh is None:
                return Invocation(
                    status="forbidden", message="no such approval request"
                )
            return await self._decided_outcome(fresh)

        if not approve:
            await self._ensure_denied(claimed)
            logger.info("approval %s denied by %s", request_id, approver)
            return Invocation(
                status="denied",
                message="action denied",
                decided_by=claimed.decided_by,
            )

        kind, _ = self._classify(claimed)
        if kind == "workflow":
            instance = await self._ensure_approved_workflow(claimed)
            logger.info(
                "approval %s approved by %s; workflow woken", request_id, approver
            )
            inv = _workflow_invocation(instance)
            inv.decided_by = claimed.decided_by
            return inv

        tool = self._tools[claimed.tool]
        result = await tool.execute(
            claimed.permission, _args_for_execute(tool, claimed)
        )
        # Mark AFTER execute returns; containment in the helper — a marking
        # failure must never discard the only copy of the ToolResult.
        await self._mark_reconciled_safe(claimed.id)
        logger.info("approval %s approved by %s; executed", request_id, approver)
        return Invocation(
            status="executed", result=result, decided_by=claimed.decided_by
        )

    async def reconcile_decisions(self) -> int:
        """Heal decided approvals whose effect never landed; returns the count.

        Called from the recovery pass. Walks the entire unreconciled set in
        keyset-paginated batches, advancing the cursor unconditionally so a
        skipped (deferred/poison) row never shadows younger healable ones.
        Every operation is an idempotent no-op on an already-healed pair, so
        the sweep is safe to repeat and safe under the recovery lock's
        coordination-not-correctness contract.
        """
        healed = 0
        cursor: tuple | None = None
        while True:
            batch = await self.approvals.decided_unreconciled(
                limit=200, after=cursor
            )
            if not batch:
                return healed
            cursor = (batch[-1].created_at, batch[-1].id)
            for request in batch:
                try:
                    healed += 1 if await self._reconcile_decision(request) else 0
                except Exception:  # noqa: BLE001 — per-row isolation
                    logger.exception(
                        "failed to reconcile approval decision %s", request.id
                    )

    async def _reconcile_decision(self, request: ApprovalRequest) -> bool:
        """Heal one decided-but-unreconciled row; True when it was retired."""
        if request.status == "denied":
            return await self._ensure_denied(request)
        kind, _ = self._classify(request)
        if kind == "workflow":
            unavailable = self._approve_effect_unavailable(request)
            if unavailable is not None:
                # A capability-drifted row defers — never raises, never
                # retires: it heals on the first pass after the capability
                # (connector/engine/workflow registration) returns.
                logger.warning(
                    "approved approval %s cannot be healed here (%s); deferring",
                    request.id,
                    unavailable,
                )
                return False
            await self._ensure_approved_workflow(request)
            return True
        if kind == "direct":
            # No automatic effect is possible (direct execution is not
            # idempotent) — retire the row so it doesn't haunt the sweep;
            # matches today's semantics for a crash mid-execute.
            await self._mark_reconciled_safe(request.id)
            return True
        logger.warning(
            "approved approval %s names unregistered tool %r; deferring until "
            "the connector returns",
            request.id,
            request.tool,
        )
        return False

    async def _decided_outcome(self, request: ApprovalRequest) -> Invocation:
        """Report — and idempotently re-ensure — an already-decided request.

        The row is the truth: every identity on the way out is the row's
        ``decided_by``, never the caller who happened to observe the decision.
        """
        if request.status == "denied":
            # Re-ensure heals the crash-after-deny-claim window on this click.
            await self._ensure_denied(request)
            return Invocation(
                status="denied",
                message=f"approval {request.id} already denied",
                decided_by=request.decided_by,
            )
        kind, _ = self._classify(request)
        if kind == "workflow" and self._approve_effect_unavailable(request) is None:
            instance = await self._ensure_approved_workflow(request)
            inv = _workflow_invocation(instance)
            inv.decided_by = request.decided_by
            return inv
        # Direct rows (winner's execute may still be in flight — no result
        # exists to return), unknown-tool rows, and workflow rows whose effect
        # can't run here all report the durable decision, non-terminal: a
        # placeholder must never let the session run a continuation off it,
        # and a duplicate click must never be answered "tool unavailable".
        return Invocation(
            status="approved",
            message=(
                f"approval {request.id} already approved by "
                f"{request.decided_by or 'an approver'}"
            ),
            decided_by=request.decided_by,
        )

    def _classify(self, request: ApprovalRequest) -> tuple[str, str | None]:
        """``("workflow", instance_id)`` / ``("direct", None)`` / ``("unknown", None)``.

        Stamped rows route on the durable marker alone — reclassifying them
        from the resolver's current engine/tool shape is forbidden (mode
        drift = double-run or phantom instance). Legacy rows (``None``
        marker) classify by the registry, the best available truth; an
        unregistered tool is *unknown* (connector disabled by config or
        credential drift), never conflated with "direct".
        """
        if request.workflow_backed is True:
            return "workflow", request.workflow_instance_id
        if request.workflow_backed is False:
            return "direct", None
        tool = self._tools.get(request.tool)
        if tool is None:
            return "unknown", None
        if getattr(tool, "workflow", None):
            # Legacy workflow row: its job_id is gateway-generated for
            # workflow tools, so _instance_id is trustworthy here.
            return "workflow", _instance_id(request)
        return "direct", None

    def _approve_effect_unavailable(self, request: ApprovalRequest) -> str | None:
        """Operational reason an approve's effect cannot run on this gateway.

        Availability, never reclassification: a stamped workflow row whose
        capability has drifted away stays workflow-backed and defers.
        ``None`` means every capability the effect dereferences is present.
        """
        if request.workflow_backed is False:
            if self._tools.get(request.tool) is None:
                return f"tool {request.tool!r} is not registered"
            return None
        if request.workflow_backed is True:
            return self._workflow_effect_unavailable(request.tool)
        kind, _ = self._classify(request)
        if kind == "unknown":
            return f"tool {request.tool!r} is not registered"
        if kind == "workflow":
            return self._workflow_effect_unavailable(request.tool)
        return None

    def _workflow_effect_unavailable(self, tool_name: str) -> str | None:
        """Check the exact capabilities ``_ensure_approved_workflow`` dereferences."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"tool {tool_name!r} is not registered"
        workflow = getattr(tool, "workflow", None)
        if not workflow:
            return f"tool {tool_name!r} no longer declares a workflow"
        if self.engine is None:
            return "this gateway has no workflow engine"
        if workflow not in self.engine.workflows:
            return f"workflow {workflow!r} is not registered on this engine"
        return None

    async def _ensure_approved_workflow(self, request: ApprovalRequest):
        """Idempotently ensure an approved request's workflow exists and is woken.

        Shared by the resolve winner, the lost-claim re-ensure, and the
        decision reconciler; every call is a no-op when the work already
        happened (start recreates a missing instance, send_event's claim is
        consume-once). Callers gate on ``_approve_effect_unavailable`` first.
        The event payload's approver is always the row's ``decided_by``.
        """
        tool = self._tools[request.tool]
        instance_id = request.workflow_instance_id or _instance_id(request)
        await self.engine.start(
            tool.workflow, instance_id, _workflow_initial_state(request)
        )
        instance = await self.engine.send_event(
            instance_id,
            "await_approval",
            {"approver": request.decided_by, "approval_id": request.id},
            drive=False,
        )
        if instance is not None and instance.status not in WORKFLOW_TERMINAL:
            self.engine.drive_background(instance_id)
        await self._mark_reconciled_safe(request.id)
        return instance

    async def _ensure_denied(self, request: ApprovalRequest) -> bool:
        """Idempotently ensure a denied request's cancel effect; True = marked.

        Cancels only what the row durably names — never a target derived from
        model-supplied args (a direct request's job_id may collide with an
        unrelated live workflow). Engine absence defers instead of marking:
        ``workflow_backed=True`` is fleet-global truth, and marking here would
        permanently hide the missed cancel from every sweep.
        """
        kind, instance_id = self._classify(request)
        if kind == "workflow":
            if self.engine is None:
                logger.warning(
                    "denied approval %s targets workflow %s but this gateway "
                    "has no engine; deferring the cancel to a capable "
                    "resolver or sweep",
                    request.id,
                    instance_id,
                )
                return False
            # A None return means already terminal or never created — nothing
            # can revive a denied row, so the effect is ensured either way.
            await self.engine.cancel(instance_id, "approval denied")
            await self._mark_reconciled_safe(request.id)
            return True
        if kind == "direct":
            await self._mark_reconciled_safe(request.id)
            return True
        logger.warning(
            "denied approval %s names unregistered tool %r; leaving its "
            "cancel classification to a later sweep",
            request.id,
            request.tool,
        )
        return False

    async def _mark_reconciled_safe(self, request_id: str) -> None:
        """Contained effect-marking — bookkeeping must never fail a heal or
        discard a performed effect's result."""
        try:
            await self.approvals.mark_reconciled(request_id)
        except Exception:  # noqa: BLE001 — the sweep retries an unmarked row
            logger.exception("failed to mark approval %s reconciled", request_id)

    async def _maybe_start_workflow(self, tool: Tool, request: ApprovalRequest) -> None:
        workflow = getattr(tool, "workflow", None)
        if self.engine is None or not workflow:
            return
        await self.engine.start(
            workflow,
            instance_id=request.workflow_instance_id or _instance_id(request),
            initial_state=_workflow_initial_state(request),
        )


def _instance_id(request: ApprovalRequest) -> str:
    """Legacy-row fallback for a workflow instance id — the job_id thread, else
    the req id. Stamped rows carry ``workflow_instance_id`` and never call this;
    it survives only for rows created before that column and for the invoke-time
    stamp itself."""
    return request.args.get("job_id") or request.id


def _workflow_initial_state(request: ApprovalRequest) -> dict:
    state = dict(request.args)
    state.setdefault("approval_id", request.id)
    # The record's args-contract version rides into the durable workflow state
    # so the consuming step can refuse a stale record. A pre-version approval
    # (or a parked instance from before versioning) simply lacks the key —
    # the NULL sentinel that always refuses.
    if request.args_schema is not None:
        state.setdefault("args_schema", request.args_schema)
    return state


def _args_for_execute(tool: Tool, request: ApprovalRequest) -> dict:
    """Args for the direct (engine-less) execute of an approved request.

    A versioned action gets the record's ``args_schema`` folded in — the same
    key the workflow path carries in its initial state — so the consumer can
    refuse a record written under an older contract. Untyped actions (MCP)
    receive their args untouched; an injected key would leak into the foreign
    tool call.
    """
    if tool.describe(request.permission).version is None:
        return request.args
    return {
        **request.args,
        "args_schema": request.args_schema,
        # Internal surface identity, added only after approval resolution. Native
        # typed tools may persist it for later authorization; it was never part
        # of model-facing arguments.
        "approved_by": (request.decided_by or "").lstrip("@") or None,
        # The approval id rides the direct execute path too (the workflow path
        # already carries it), so worker spend traces to its authorization.
        "approval_id": request.id,
    }


def _workflow_invocation(instance) -> Invocation:
    """Map a workflow instance's terminal state onto an Invocation."""
    if instance is None:
        return Invocation(status="forbidden", message="no such workflow instance")
    if instance.status == "completed":
        result = instance.result or {}
        return Invocation(
            status="executed",
            result=ToolResult(
                ok=True, summary=result.get("summary", "workflow completed"),
                data=result,
            ),
        )
    if instance.status == "failed":
        return Invocation(
            status="executed",
            result=ToolResult(
                ok=False,
                summary=f"workflow {instance.id} failed: {instance.error}",
                data={"error": instance.error, "status": "failed"},
            ),
        )
    # Still running/waiting (e.g. another wait node) — surface progress.
    return Invocation(
        status="started",
        result=ToolResult(
            ok=True,
            summary=f"workflow {instance.id} started",
            data={
                "instance_id": instance.id,
                "status": instance.status,
                "waiting_on": instance.waiting_on,
            },
        ),
    )


def _summarize(action: str, args: dict) -> str:
    if action == "github.issues:write":
        return f"create issue in {args.get('repo', '?')}: {args.get('title', '')}".strip()
    if action == "github.pulls:write":
        return (
            f"open PR in {args.get('repo', '?')} "
            f"({args.get('head', '?')} → {args.get('base', 'main')}): "
            f"{args.get('title', '')}"
        ).strip()
    if action == "coding_worker.pr:write":
        # Be explicit: this gate lets the worker START and open a *draft* PR.
        # It is NOT a review of a generated diff (the draft PR is that gate).
        return (
            f"run coding worker + open draft PR in {args.get('repo', '?')}: "
            f"{args.get('instruction', '')}"
        ).strip()
    return f"{action} {args}"
