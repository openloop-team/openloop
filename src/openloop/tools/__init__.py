"""Tool gateway — an explicit allowlist of tools the agent may use.

Actions are named ``<tool>.<permission>`` (e.g. ``github.issues:write``) so they
line up with the agent's `tools` allowlist and `approvals.require_for` policy.
The gateway enforces the allowlist, routes write actions through human approval,
then executes via a registered :class:`Tool`.
"""

from openloop.tools.base import (
    ActionSpec,
    Invocation,
    Tool,
    ToolResult,
    split_action,
)
from openloop.tools.gateway import ToolGateway, ToolSpecs
from openloop.tools.policy import allowed_actions, is_allowed

__all__ = [
    "ActionSpec",
    "Invocation",
    "Tool",
    "ToolResult",
    "ToolGateway",
    "ToolSpecs",
    "allowed_actions",
    "is_allowed",
    "split_action",
]
