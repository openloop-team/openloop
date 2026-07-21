"""Factory for programmatic test agents.

``AgentMetadata.id`` is required (the durable identity of record), so every
hand-built test agent needs one; the factory mints a throwaway id per call
unless the test pins one to assert on identity itself.
"""

from __future__ import annotations

from uuid import uuid4

from openloop.agents.schema import Agent, AgentMetadata, AgentSpec, ModelPolicy


def make_agent(
    name: str = "a", workspace: str = "w", *, id: str | None = None, **spec
) -> Agent:
    spec.setdefault("model_policy", ModelPolicy(default="m"))
    return Agent(
        metadata=AgentMetadata(
            name=name, workspace=workspace, id=id or uuid4().hex
        ),
        spec=AgentSpec(**spec),
    )
