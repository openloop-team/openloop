"""Tests for tool-policy enforcement, the approval gate, and the GitHub tool."""

import pytest

from openloop.agents import load_agent
from openloop.tools import ToolGateway, allowed_actions, is_allowed, split_action
from openloop.tools.github import GitHubConnector
from openloop.testing import EXAMPLE_AGENT, FakeGitHub


def _agent():
    # tools: github [issues:read, issues:write, pulls:read]; ci-logs [read]
    # approvals.require_for: github.issues:write, github.pulls:write
    return load_agent(EXAMPLE_AGENT)


def _gateway():
    return ToolGateway(tools=[GitHubConnector(FakeGitHub())])


def test_split_action():
    assert split_action("github.issues:write") == ("github", "issues:write")
    with pytest.raises(ValueError):
        split_action("nodot")


def test_allowlist_reflects_policy():
    agent = _agent()
    actions = allowed_actions(agent)
    assert "github.issues:write" in actions
    assert "ci-logs.get_run_logs" in actions  # MCP perms are tool names
    assert not is_allowed(agent, "github.repos:delete")


async def test_read_action_executes_without_approval():
    agent = _agent()
    inv = await _gateway().invoke(
        agent, "github.issues:read", {"repo": "acme/x", "number": 7}
    )
    assert inv.status == "executed"
    assert inv.result.ok
    assert "read issue #7" in inv.result.summary


async def test_write_action_is_held_for_approval():
    agent = _agent()
    gw = _gateway()
    inv = await gw.invoke(
        agent, "github.issues:write",
        {"repo": "acme/x", "title": "Track decision"},
        requested_by="U1",
    )
    assert inv.status == "pending_approval"
    assert inv.result is None  # not executed yet
    assert inv.approval.summary.startswith("create issue in acme/x")
    assert await gw.approvals.pending(agent="dev-platform")


async def test_disallowed_action_is_forbidden():
    agent = _agent()
    inv = await _gateway().invoke(agent, "github.repos:delete", {})
    assert inv.status == "forbidden"


async def test_action_without_registered_tool_is_forbidden():
    agent = _agent()
    gw = ToolGateway(tools=[])  # nothing registered
    inv = await gw.invoke(agent, "github.issues:read", {"repo": "a/b", "number": 1})
    assert inv.status == "forbidden"


async def test_approval_then_execute():
    agent = _agent()
    client = FakeGitHub()
    gw = ToolGateway(tools=[GitHubConnector(client)])
    pending = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)
    assert resolved.status == "executed"
    assert resolved.result.ok
    assert client.created == [{"number": 1, "repo": "acme/x", "title": "T"}]


async def test_non_approver_cannot_approve():
    agent = _agent()
    gw = _gateway()
    pending = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    inv = await gw.resolve(pending.approval.id, "@random", approve=True)
    assert inv.status == "forbidden"


async def test_denied_approval_does_not_execute():
    agent = _agent()
    client = FakeGitHub()
    gw = ToolGateway(tools=[GitHubConnector(client)])
    pending = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    inv = await gw.resolve(pending.approval.id, "@maciag.artur", approve=False)
    assert inv.status == "denied"
    assert client.created == []
