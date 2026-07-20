"""Tests for the Slack approval blocks and button resolution (no Bolt app)."""

from pathlib import Path
from openloop.agents import load_agent
from openloop.runtime import Runtime, Task
from openloop.surfaces.approvals import (
    APPROVE_ACTION,
    DENY_ACTION,
    OPENHANDS_ACCEPT_ACTION,
    OPENHANDS_REJECT_ACTION,
    approval_blocks,
    openhands_decision_blocks,
    resolve_from_action,
)
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.testing import (
    FakeGitHub,
    ScriptedGateway,
    in_memory_workflow_engine,
    tool_call_response,
)

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _agent():
    return load_agent(AGENT_YAML)


async def _pending(gateway):
    agent = _agent()
    inv = await gateway.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "Track decision"}
    )
    return inv.approval


async def test_runtime_surfaces_approval_ids():
    agent = _agent()
    tools = ToolGateway(tools=[GitHubConnector(FakeGitHub())])
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_issues_write",
                                  {"repo": "acme/x", "title": "T"})]),
    ])
    runtime = Runtime(
        agent, gateway=gateway, tools=tools, engine=in_memory_workflow_engine()
    )
    result = await runtime.handle(
        Task(text="open an issue", surface="slack", channel="#dev-platform")
    )
    assert len(result.approval_ids) == 1
    assert await tools.approvals.get(result.approval_ids[0]) is not None


async def test_approval_blocks_have_approve_and_deny_buttons():
    gateway = ToolGateway(tools=[GitHubConnector(FakeGitHub())])
    req = await _pending(gateway)
    blocks = approval_blocks([req])

    actions = [b for b in blocks if b["type"] == "actions"]
    assert len(actions) == 1
    action_ids = {e["action_id"] for e in actions[0]["elements"]}
    assert action_ids == {APPROVE_ACTION, DENY_ACTION}
    # Buttons carry the approval id so the click can resolve it.
    assert all(e["value"] == req.id for e in actions[0]["elements"])


def test_openhands_decision_blocks_are_explicit_and_opaque():
    blocks = openhands_decision_blocks("job-1", "decision-1", "Run tests")
    actions = next(block for block in blocks if block["type"] == "actions")
    assert {button["action_id"] for button in actions["elements"]} == {
        OPENHANDS_ACCEPT_ACTION,
        OPENHANDS_REJECT_ACTION,
    }
    assert all(
        button["value"] == "job-1|decision-1" for button in actions["elements"]
    )


async def test_resolve_approve_executes():
    github = FakeGitHub()
    gateway = ToolGateway(tools=[GitHubConnector(github)])
    req = await _pending(gateway)
    msg = await resolve_from_action(gateway, req.id, "@maciag.artur", approve=True)
    assert msg.startswith("✅ Approved by @maciag.artur")
    assert github.created  # the issue was created on approval


async def test_resolve_deny_does_not_execute():
    github = FakeGitHub()
    gateway = ToolGateway(tools=[GitHubConnector(github)])
    req = await _pending(gateway)
    msg = await resolve_from_action(gateway, req.id, "@maciag.artur", approve=False)
    assert msg.startswith("🚫 Denied")
    assert github.created == []


async def test_resolve_non_approver_is_blocked():
    gateway = ToolGateway(tools=[GitHubConnector(FakeGitHub())])
    req = await _pending(gateway)
    msg = await resolve_from_action(gateway, req.id, "@random", approve=True)
    assert msg.startswith("⛔")
