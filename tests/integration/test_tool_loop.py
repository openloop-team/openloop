"""Integration: the runtime's model<->tool calling loop."""

from pathlib import Path
import json

from openloop.agents import load_agent
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.runtime.pipeline import TOOL_RESULT_MAX_CHARS
from openloop.tools import ActionSpec, ToolGateway, ToolResult
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore
from openloop.testing import (
    FakeGitHub,
    ScriptedGateway,
    in_memory_workflow_engine,
    tool_call_response,
)

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _agent():
    return load_agent(AGENT_YAML)


def _task(text="do it"):
    return Task(text=text, surface="slack", channel="#dev-platform", user="U1")


async def test_model_calls_read_tool_then_answers():
    agent = _agent()
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_issues_read",
                                  {"repo": "acme/x", "number": 7})]),
        ModelResponse(text="Issue #7 is open.", model="m"),
    ])
    runtime = Runtime(
        agent, gateway=gateway, tools=tools, engine=in_memory_workflow_engine()
    )

    result = await runtime.handle(_task("status of issue 7?"))
    assert result.text == "Issue #7 is open."

    # The model was offered the tool definitions, and a tool result was fed back.
    offered = {t["function"]["name"] for t in gateway.calls[0]["tools"]}
    assert "github_issues_read" in offered
    tool_msgs = [m for m in gateway.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    # The result carries the actual data, not just a summary — a data-free
    # "read issue #7" would leave the model to invent the issue's content.
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["ok"] is True
    assert payload["data"]["state"] == "open"
    assert payload["data"]["number"] == 7


async def test_write_tool_call_is_held_for_approval():
    agent = _agent()
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_issues_write",
                                  {"repo": "acme/x", "title": "Track decision"})]),
    ])
    runtime = Runtime(
        agent, gateway=gateway, tools=tools, engine=in_memory_workflow_engine()
    )

    result = await runtime.handle(_task("open an issue to track this"))
    assert result.model == "approval-gate"
    assert "approval required" in result.text.lower()
    assert github.created == []  # not executed
    assert await tools.approvals.pending(agent="dev-platform")


async def test_unoffered_tool_call_is_reported_to_model():
    # The model can only be *offered* allowlisted actions, so a call to anything
    # else (e.g. a hallucinated function) comes back as "unknown tool" — the
    # gateway's forbidden path guards the direct API route, not this loop.
    agent = _agent()
    tools = ToolGateway(tools=[GitHubConnector(FakeGitHub())])
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_repos_delete", {})]),
        ModelResponse(text="I can't do that.", model="m"),
    ])
    runtime = Runtime(
        agent, gateway=gateway, tools=tools, engine=in_memory_workflow_engine()
    )

    result = await runtime.handle(_task("delete the repo"))
    assert result.text == "I can't do that."
    tool_msgs = [m for m in gateway.calls[1]["messages"] if m["role"] == "tool"]
    assert any("unknown tool" in m["content"] for m in tool_msgs)


class _VerboseTool:
    """A tool whose result data is far larger than the feedback cap."""

    name = "github"

    def supported_permissions(self):
        return {"issues:read"}

    def describe(self, permission):
        return ActionSpec("read", {"type": "object", "properties": {}})

    async def execute(self, permission, args):
        return ToolResult(ok=True, summary="read", data={"blob": "x" * 50_000})


async def test_oversized_tool_result_is_capped():
    agent = _agent()
    tools = ToolGateway(tools=[_VerboseTool()])
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_issues_read", {})]),
        ModelResponse(text="done", model="m"),
    ])
    runtime = Runtime(
        agent, gateway=gateway, tools=tools, engine=in_memory_workflow_engine()
    )

    await runtime.handle(_task("read the big one"))
    tool_msgs = [m for m in gateway.calls[1]["messages"] if m["role"] == "tool"]
    content = tool_msgs[0]["content"]
    assert content.endswith("… [truncated]")
    assert len(content) < TOOL_RESULT_MAX_CHARS + 100


async def test_no_tools_gateway_behaves_as_plain_chat():
    agent = _agent()
    gateway = ScriptedGateway([ModelResponse(text="hello", model="m")])
    runtime = Runtime(
        agent, gateway=gateway, engine=in_memory_workflow_engine()
    )  # tools=None
    result = await runtime.handle(_task("hi"))
    assert result.text == "hello"
    assert gateway.calls[0]["tools"] is None


async def test_usage_accumulates_across_loop():
    agent = _agent()
    tools = ToolGateway(tools=[GitHubConnector(FakeGitHub())])

    first = tool_call_response(
        "m", [("c1", "github_issues_read", {"repo": "a/b", "number": 1})]
    )
    first.cost_usd, first.prompt_tokens, first.completion_tokens = 0.01, 10, 5
    second = ModelResponse(text="done", model="m", cost_usd=0.02,
                           prompt_tokens=20, completion_tokens=8)

    usage = InMemoryUsageStore()
    runtime = Runtime(
        agent,
        gateway=ScriptedGateway([first, second]),
        tools=tools,
        usage=usage,
        engine=in_memory_workflow_engine(),
    )
    await runtime.handle(_task())

    rec = usage.records[0]
    assert rec.cost_usd == 0.03
    assert rec.prompt_tokens == 30
    assert rec.completion_tokens == 13
