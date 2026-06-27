"""Typed schema for the `apiVersion: openloop.ai/v1alpha1` Agent config.

This is the in-code representation of `agents/*.yaml`. The schema is PRELIMINARY
and tracks the README; expect it to evolve.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

API_VERSION = "openloop.ai/v1alpha1"


class Surface(BaseModel):
    type: Literal["slack", "discord", "zoom", "github", "linear"]
    channel: str | None = None


class Memory(BaseModel):
    scope: Literal["channel", "agent", "workspace"] = "channel"
    backend: Literal["postgres"] = "postgres"
    retention_days: int = 90


class ModelRoute(BaseModel):
    match: dict[str, str]
    model: str


class ModelPolicy(BaseModel):
    default: str
    routes: list[ModelRoute] = Field(default_factory=list)

    def resolve(self, task: str | None) -> str:
        """Pick a model for a task, falling back to the default.

        A route matches when every key/value in its `match` block equals the
        task context. For the first slice the only context key is `task`.
        """
        if task is not None:
            for route in self.routes:
                if route.match.get("task") == task:
                    return route.model
        return self.default


class Tool(BaseModel):
    name: str
    type: Literal["native", "mcp"]
    server: str | None = None
    permissions: list[str] = Field(default_factory=list)


class Approvals(BaseModel):
    require_for: list[str] = Field(default_factory=list)
    approvers: list[str] = Field(default_factory=list)

    def requires_approval(self, action: str) -> bool:
        return action in self.require_for


class Budget(BaseModel):
    monthly_usd: float | None = None
    per_task_usd: float | None = None
    on_exceeded: Literal["block", "warn"] = "block"


class AgentSpec(BaseModel):
    surfaces: list[Surface] = Field(default_factory=list)
    memory: Memory = Field(default_factory=Memory)
    model_policy: ModelPolicy
    tools: list[Tool] = Field(default_factory=list)
    approvals: Approvals = Field(default_factory=Approvals)
    budget: Budget = Field(default_factory=Budget)


class AgentMetadata(BaseModel):
    name: str
    workspace: str


class Agent(BaseModel):
    """A team agent — the unit of identity, memory, and policy."""

    apiVersion: str = API_VERSION
    kind: Literal["Agent"] = "Agent"
    metadata: AgentMetadata
    spec: AgentSpec

    def model_for(self, task: str | None = None) -> str:
        return self.spec.model_policy.resolve(task)

    def has_slack_surface(self) -> bool:
        return any(s.type == "slack" for s in self.spec.surfaces)
