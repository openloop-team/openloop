"""The tool gateway: enforce the allowlist, gate writes on approval, execute."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from openloop.agents.schema import Agent
from openloop.approvals.store import (
    ApprovalRequest,
    ApprovalStore,
    InMemoryApprovalStore,
)
from openloop.tools.base import Invocation, Tool, ToolResult, split_action
from openloop.tools.policy import is_allowed

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
    ) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)
        self.approvals = approvals or InMemoryApprovalStore()

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

        # Let a tool finalize its args before they cross the approval boundary
        # (e.g. the coding worker mints a job_id here so it's persisted in the
        # approval request and reused verbatim at execute time).
        prepare = getattr(tool, "prepare_args", None)
        if prepare is not None:
            args = prepare(permission, args)

        if agent.spec.approvals.requires_approval(action):
            request = ApprovalRequest(
                agent=agent.metadata.name,
                action=action,
                tool=tool_name,
                permission=permission,
                args=args,
                approvers=list(agent.spec.approvals.approvers),
                requested_by=requested_by,
                summary=_summarize(action, args),
            )
            await self.approvals.create(request)
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
            logger.info("approval %s denied by %s", request_id, approver)
            return Invocation(status="denied", message="action denied")

        request.status = "approved"
        await self.approvals.update(request)
        tool = self._tools[request.tool]
        result = await tool.execute(request.permission, request.args)
        logger.info("approval %s approved by %s; executed", request_id, approver)
        return Invocation(status="executed", result=result)


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
