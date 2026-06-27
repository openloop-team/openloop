"""Integration: the runtime's model<->tool calling loop."""

from openloop.agents import load_agent
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore
from openloop.testing import EXAMPLE_AGENT, FakeGitHub, ScriptedGateway, tool_call_response


def _agent():
    return load_agent(EXAMPLE_AGENT)


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
    runtime = Runtime(agent, gateway=gateway, tools=tools)

    result = await runtime.handle(_task("status of issue 7?"))
    assert result.text == "Issue #7 is open."

    # The model was offered the tool definitions, and a tool result was fed back.
    offered = {t["function"]["name"] for t in gateway.calls[0]["tools"]}
    assert "github_issues_read" in offered
    roles = [m["role"] for m in gateway.calls[1]["messages"]]
    assert "tool" in roles


async def test_write_tool_call_is_held_for_approval():
    agent = _agent()
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_issues_write",
                                  {"repo": "acme/x", "title": "Track decision"})]),
    ])
    runtime = Runtime(agent, gateway=gateway, tools=tools)

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
    runtime = Runtime(agent, gateway=gateway, tools=tools)

    result = await runtime.handle(_task("delete the repo"))
    assert result.text == "I can't do that."
    tool_msgs = [m for m in gateway.calls[1]["messages"] if m["role"] == "tool"]
    assert any("unknown tool" in m["content"] for m in tool_msgs)


async def test_no_tools_gateway_behaves_as_plain_chat():
    agent = _agent()
    gateway = ScriptedGateway([ModelResponse(text="hello", model="m")])
    runtime = Runtime(agent, gateway=gateway)  # tools=None
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
    runtime = Runtime(agent, gateway=ScriptedGateway([first, second]),
                      tools=tools, usage=usage)
    await runtime.handle(_task())

    rec = usage.records[0]
    assert rec.cost_usd == 0.03
    assert rec.prompt_tokens == 30
    assert rec.completion_tokens == 13
