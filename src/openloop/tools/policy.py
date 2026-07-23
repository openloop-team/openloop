"""Tool-policy enforcement — the agent's explicit allowlist (least privilege)."""

from __future__ import annotations

from openloop.agents.schema import Agent
from openloop.tools.aliases import _canonical_action


def allowed_actions(agent: Agent) -> set[str]:
    """Every ``<tool>.<permission>`` action the agent's policy permits.

    Canonicalized: an agent's YAML may still declare a legacy action name
    (e.g. ``coding_worker.pr:write``) that has since been renamed. The
    allowlist is built in canonical space so it matches a canonical query
    without requiring the YAML to be migrated first.
    """
    actions: set[str] = set()
    for tool in agent.spec.tools:
        for permission in tool.permissions:
            actions.add(_canonical_action(f"{tool.name}.{permission}"))
    return actions


def is_allowed(agent: Agent, action: str) -> bool:
    # The queried action is canonicalized too, so a caller that (still) passes
    # the legacy spelling directly to is_allowed matches a policy declared
    # either way.
    return _canonical_action(action) in allowed_actions(agent)
