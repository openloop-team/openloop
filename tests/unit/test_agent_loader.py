"""Tests for loading and validating agent config-as-code."""

from pathlib import Path

import pytest

from openloop.agents import load_agent, load_agents
from openloop.agents.loader import AgentConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "agents" / "dev-platform.yaml"


def test_loads_example_agent():
    agent = load_agent(EXAMPLE)
    assert agent.metadata.name == "dev-platform"
    assert agent.metadata.workspace == "acme"
    assert agent.has_slack_surface()
    assert agent.spec.budget.on_exceeded == "block"


def test_load_agents_directory_keys_by_name():
    agents = load_agents(REPO_ROOT / "agents")
    assert "dev-platform" in agents


def test_rejects_unsupported_api_version(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "apiVersion: openloop.ai/v0\nkind: Agent\n"
        "metadata: {name: x, workspace: y}\n"
        "spec: {model_policy: {default: openai/gpt-4o-mini}}\n"
    )
    with pytest.raises(AgentConfigError):
        load_agent(bad)


def test_rejects_missing_required_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "apiVersion: openloop.ai/v1alpha1\nkind: Agent\n"
        "metadata: {name: x, workspace: y}\nspec: {}\n"  # no model_policy
    )
    with pytest.raises(AgentConfigError):
        load_agent(bad)
