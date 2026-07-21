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


VALID_ID = "9f2c1d4e8a7b4c3d9e0f1a2b3c4d5e6f"


def _agent_yaml(name="x", id=VALID_ID):
    return (
        "apiVersion: openloop.team/v1alpha1\nkind: Agent\n"
        f"metadata: {{name: {name}, workspace: y, id: {id}}}\n"
        "spec: {model_policy: {default: openai/gpt-4o-mini}}\n"
    )


def test_accepts_a_well_formed_id(tmp_path):
    file = tmp_path / "a.yaml"
    file.write_text(_agent_yaml())
    assert load_agent(file).metadata.id == VALID_ID


@pytest.mark.parametrize(
    "bad", ["nope", VALID_ID.upper(), VALID_ID[:-1], VALID_ID + "0"]
)
def test_rejects_a_malformed_id(tmp_path, bad):
    file = tmp_path / "a.yaml"
    file.write_text(_agent_yaml(id=bad))
    with pytest.raises(AgentConfigError):
        load_agent(file)


def test_rejects_an_id_less_agent(tmp_path):
    # An agent without issued identity is not loadable. The one path that
    # reads an id-less file is `openloop agents id issue`, which reads raw
    # YAML precisely so it can stamp the id this loader requires.
    file = tmp_path / "a.yaml"
    file.write_text(
        "apiVersion: openloop.team/v1alpha1\nkind: Agent\n"
        "metadata: {name: x, workspace: y}\n"
        "spec: {model_policy: {default: openai/gpt-4o-mini}}\n"
    )
    with pytest.raises(AgentConfigError):
        load_agent(file)


def test_load_agents_rejects_a_duplicate_id_across_files(tmp_path):
    # One id = one principal. Ids are mint-only by convention; this load-time
    # check is the integrity guarantee when a copied template smuggles one in.
    (tmp_path / "a.yaml").write_text(_agent_yaml(name="a"))
    (tmp_path / "b.yaml").write_text(_agent_yaml(name="b"))  # same id
    with pytest.raises(AgentConfigError, match="duplicate agent id"):
        load_agents(tmp_path)


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
