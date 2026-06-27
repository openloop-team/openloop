"""Core tool types: the Tool protocol, results, and invocation outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from openloop.approvals.store import ApprovalRequest


@dataclass(slots=True)
class ToolResult:
    ok: bool
    summary: str
    data: dict = field(default_factory=dict)


@dataclass(slots=True)
class ActionSpec:
    """Describes an action to the model: a description + JSON-schema args."""

    description: str
    parameters: dict


@dataclass(slots=True)
class Invocation:
    """Outcome of asking the gateway to run an action."""

    status: str  # executed | pending_approval | forbidden | denied
    result: ToolResult | None = None
    approval: ApprovalRequest | None = None
    message: str | None = None


@runtime_checkable
class Tool(Protocol):
    """A native or MCP-backed connector exposing permissioned actions."""

    name: str

    def supported_permissions(self) -> set[str]: ...

    def describe(self, permission: str) -> ActionSpec: ...

    async def execute(self, permission: str, args: dict) -> ToolResult: ...


def split_action(action: str) -> tuple[str, str]:
    """Split ``github.issues:write`` into ``("github", "issues:write")``."""
    tool, _, permission = action.partition(".")
    if not tool or not permission:
        raise ValueError(f"malformed action {action!r}; expected '<tool>.<perm>'")
    return tool, permission
