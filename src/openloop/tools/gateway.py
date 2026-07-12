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
            # workflow-backed tool can reuse the requesting thread's warm context.
            args = prepare(permission, args, agent, warm_key=warm_key)

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

        # Some actions are intrinsically high-risk regardless of an accidental
        # omission in an agent's config. Phase 1 sealed analysis is one: it can
        # process provisioned sensitive data and spend model budget.
        if (
            agent.spec.approvals.requires_approval(action)
            or getattr(tool, "requires_approval", False)
        ):
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
            )
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
        request = await self.approvals.get(request_id)
        if request is None:
            return Invocation(status="forbidden", message="no such approval request")
        if request.status != "pending":
            if request.status == "approved":
                tool = self._tools.get(request.tool)
                if self.engine is not None and getattr(tool, "workflow", None):
                    instance = await self.engine.store.get(_instance_id(request))
                    return _workflow_invocation(instance)
            return Invocation(
                status="denied" if request.status == "denied" else "executed",
                message=f"approval {request_id} already {request.status}",
            )
        if approver not in request.approvers:
            return Invocation(
                status="forbidden",
                message=f"{approver} is not an approver for {request.action}",
            )

        request.decided_by = approver
        if not approve:
            request.status = "denied"
            await self.approvals.update(request)
            tool = self._tools[request.tool]
            # Cancel the parked workflow so a denied request isn't left waiting.
            if self.engine is not None and getattr(tool, "workflow", None):
                await self.engine.cancel(_instance_id(request), "approval denied")
            logger.info("approval %s denied by %s", request_id, approver)
            return Invocation(status="denied", message="action denied")

        tool = self._tools[request.tool]

        # Thin adapter: for a workflow-backed tool, approval is just an event that
        # wakes the parked workflow — not a direct execute(). Record that durable
        # wake *before* flipping the approval to "approved", so a crash before the
        # state change leaves the request still "pending" and re-resolvable:
        #   - start() ensures the workflow exists (covers a crash in invoke() after
        #     the approval was created but before the workflow was started),
        #   - send_event() records the approval event (a no-op if already past the
        #     wait node),
        #   - the rest of the workflow runs in the background after the button
        #     handler can return "started".
        if self.engine is not None and getattr(tool, "workflow", None):
            instance_id = _instance_id(request)
            await self.engine.start(
                tool.workflow, instance_id, _workflow_initial_state(request)
            )
            instance = await self.engine.send_event(
                instance_id,
                "await_approval",
                {"approver": approver, "approval_id": request.id},
                drive=False,
            )
            request.status = "approved"
            await self.approvals.update(request)
            if instance is not None and instance.status not in WORKFLOW_TERMINAL:
                self.engine.drive_background(instance_id)
            logger.info("approval %s approved by %s; workflow woken", request_id, approver)
            return _workflow_invocation(instance)

        request.status = "approved"
        await self.approvals.update(request)
        result = await tool.execute(
            request.permission, _args_for_execute(tool, request)
        )
        logger.info("approval %s approved by %s; executed", request_id, approver)
        return Invocation(status="executed", result=result)

    async def _maybe_start_workflow(self, tool: Tool, request: ApprovalRequest) -> None:
        workflow = getattr(tool, "workflow", None)
        if self.engine is None or not workflow:
            return
        await self.engine.start(
            workflow,
            instance_id=_instance_id(request),
            initial_state=_workflow_initial_state(request),
        )


def _instance_id(request: ApprovalRequest) -> str:
    """Workflow instance id for an approval — the job_id thread, else the req id."""
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
    return {**request.args, "args_schema": request.args_schema}


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
    if action == "analysis.report:write":
        # The sealed worker has no external write. Approval covers spend and
        # the data scope that will be provisioned into its isolated sandbox,
        # so the copy names each source concretely. Upload names come from the
        # trusted display metadata the pre-approval resolution stamped
        # (`upload_meta`) — the model only supplies an opaque upload_ref, so
        # without the stamp this card could not truthfully name the file. Repo
        # names are model-supplied by design: naming the repo to the approver
        # IS the gate.
        return (
            f"run sealed analysis over {_describe_analysis_inputs(args)} "
            f"(subject to configured spend limits): "
            f"{args.get('instruction', '')}"
        ).strip()
    return f"{action} {args}"


def _describe_analysis_inputs(args: dict) -> str:
    upload_meta = args.get("upload_meta") or {}
    parts = []
    for entry in args.get("inputs") or []:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        if source == "staged":
            parts.append(f"staged input {entry.get('input_ref', '?')}")
        elif source == "upload":
            ref = entry.get("upload_ref", "?")
            meta = upload_meta.get(ref) or {}
            name = meta.get("name") or ref
            parts.append(f"the file `{name}` shared in this thread")
        elif source == "github":
            ref = entry.get("ref") or "default branch"
            parts.append(f"repo {entry.get('repo', '?')}@{ref}")
        else:
            parts.append(f"{source or '?'} input")
    return ", ".join(parts) or "?"
