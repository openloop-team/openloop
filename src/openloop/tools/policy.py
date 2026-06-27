"""Tool-policy enforcement — the agent's explicit allowlist (least privilege)."""

from __future__ import annotations

from openloop.agents.schema import Agent


def allowed_actions(agent: Agent) -> set[str]:
    """Every ``<tool>.<permission>`` action the agent's policy permits."""
    actions: set[str] = set()
    for tool in agent.spec.tools:
        for permission in tool.permissions:
            actions.add(f"{tool.name}.{permission}")
    return actions


def is_allowed(agent: Agent, action: str) -> bool:
    return action in allowed_actions(agent)
