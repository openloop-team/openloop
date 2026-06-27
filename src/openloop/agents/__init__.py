"""Agent config-as-code: schema and loader."""

from openloop.agents.loader import load_agent, load_agents
from openloop.agents.schema import Agent

__all__ = ["Agent", "load_agent", "load_agents"]
