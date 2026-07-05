"""Tests for loading and validating agent config-as-code."""

from pathlib import Path

import pytest

from openloop.agents import load_agent, load_agents
from openloop.agents.loader import AgentConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "agents" / "dev-platform.yaml"


def test_loads_example_agent():
    agent = load_agent(EXAMPLE)
    assert agent.metadata.name == "openloop"
    assert agent.metadata.workspace == "openloop-team"
    assert agent.has_slack_surface()
    assert agent.spec.budget.on_exceeded == "block"


def test_load_agents_directory_keys_by_name():
    agents = load_agents(REPO_ROOT / "agents")
    assert "openloop" in agents


def test_example_github_mcp_exposes_ci_actions():
    """The github-mcp connector must keep the `actions` toolset enabled so CI
    answers work, and the pinned X-MCP-Toolsets header must cover every toolset
    the allowlisted tools live in. X-MCP-Toolsets REPLACES the server default
    (verified against live discovery), so dropping a toolset here silently
    breaks the tools that depend on it — this guards that coupling.
    """
    agent = load_agent(EXAMPLE)
    mcp = next(t for t in agent.spec.tools if t.name == "github-mcp")

    # CI/Actions read tools are allowlisted.
    assert {"actions_list", "actions_get", "get_job_logs"} <= set(mcp.permissions)

    # The toolset each allowlisted tool needs is pinned in the header.
    toolsets = {t.strip() for t in mcp.headers["X-MCP-Toolsets"].split(",")}
    tool_toolset = {
        "list_issues": "issues",
        "issue_read": "issues",
        "search_issues": "issues",
        "list_pull_requests": "pull_requests",
        "pull_request_read": "pull_requests",
        "actions_list": "actions",
        "actions_get": "actions",
        "get_job_logs": "actions",
    }
    for tool in mcp.permissions:
        assert tool_toolset[tool] in toolsets, (
            f"{tool!r} needs the {tool_toolset[tool]!r} toolset, "
            f"missing from X-MCP-Toolsets={sorted(toolsets)}"
        )

    # Readonly stays on — defense in depth on top of the allowlist.
    assert mcp.headers.get("X-MCP-Readonly") == "true"


def test_rejects_unsupported_api_version(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "apiVersion: openloop.team/v0\nkind: Agent\n"
        "metadata: {name: x, workspace: y}\n"
        "spec: {model_policy: {default: openai/gpt-4o-mini}}\n"
    )
    with pytest.raises(AgentConfigError):
        load_agent(bad)


def test_rejects_missing_required_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "apiVersion: openloop.team/v1alpha1\nkind: Agent\n"
        "metadata: {name: x, workspace: y}\nspec: {}\n"  # no model_policy
    )
    with pytest.raises(AgentConfigError):
        load_agent(bad)
