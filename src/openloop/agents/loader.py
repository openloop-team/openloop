"""Load agent config-as-code from YAML files into typed `Agent` objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from openloop.agents.schema import API_VERSION, Agent


class AgentConfigError(ValueError):
    """Raised when an agent YAML file is malformed or unsupported."""


def load_agent(path: str | Path) -> Agent:
    """Parse and validate a single agent YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise AgentConfigError(f"{path}: expected a YAML mapping at the top level")

    api_version = raw.get("apiVersion")
    if api_version != API_VERSION:
        raise AgentConfigError(
            f"{path}: unsupported apiVersion {api_version!r} "
            f"(expected {API_VERSION!r})"
        )

    try:
        return Agent.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError and friends
        raise AgentConfigError(f"{path}: {exc}") from exc


def load_agents(directory: str | Path) -> dict[str, Agent]:
    """Load every `*.yaml` / `*.yml` agent in a directory, keyed by name."""
    directory = Path(directory)
    agents: dict[str, Agent] = {}
    # One id = one principal. Ids are mint-only by convention (`agents id
    # issue`); this check is the integrity guarantee when a copied template
    # carries an id along anyway.
    seen_ids: dict[str, Path] = {}
    for file in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        agent = load_agent(file)
        if agent.metadata.name in agents:
            raise AgentConfigError(
                f"duplicate agent name {agent.metadata.name!r} in {file}"
            )
        if agent.metadata.id in seen_ids:
            raise AgentConfigError(
                f"duplicate agent id {agent.metadata.id!r} in {file} "
                f"(also {seen_ids[agent.metadata.id]})"
            )
        agents[agent.metadata.name] = agent
        seen_ids[agent.metadata.id] = file
    return agents
