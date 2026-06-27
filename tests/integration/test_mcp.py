"""Tests for the MCP connector — discovery, policy fit, gateway, and loop."""

from openloop.agents.schema import Agent
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.mcp import MCPConnector, MCPToolInfo
from openloop.testing import ScriptedGateway, tool_call_response


class FakeMCPClient:
    """In-memory MCP server with two tools — no network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return [
            MCPToolInfo(
                name="get_run_logs",
                description="Fetch CI run logs.",
                input_schema={
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            ),
            MCPToolInfo(name="list_runs", description="List recent CI runs."),
        ]

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return f"logs for {args.get('run_id')}" if name == "get_run_logs" else "runs"


def _mcp_agent(permissions):
    """An agent allowed the given MCP tool names on a 'ci-logs' server."""
    return Agent.model_validate(
        {
            "apiVersion": "openloop.ai/v1alpha1",
            "kind": "Agent",
            "metadata": {"name": "ci", "workspace": "acme"},
            "spec": {
                "model_policy": {"default": "openai/gpt-4o-mini"},
                "tools": [
                    {
                        "name": "ci-logs",
                        "type": "mcp",
                        "server": "http://localhost:8931",
                        "permissions": permissions,
                    }
                ],
            },
        }
    )


async def _connector():
    conn = MCPConnector("ci-logs", FakeMCPClient())
    await conn.setup()
    return conn


async def test_setup_discovers_tools_as_permissions():
    conn = await _connector()
    assert conn.supported_permissions() == {"get_run_logs", "list_runs"}
    spec = conn.describe("get_run_logs")
    assert "run_id" in spec.parameters["properties"]


async def test_gateway_exposes_only_allowlisted_mcp_tools():
    conn = await _connector()
    gw = ToolGateway(tools=[conn])
    agent = _mcp_agent(["get_run_logs"])  # list_runs not allowed
    actions = gw.available_actions(agent)
    assert actions == ["ci-logs.get_run_logs"]
    specs = gw.tool_specs(agent)
    # '.' -> '_' but '-' is valid in function names and kept.
    assert "ci-logs_get_run_logs" in specs.by_name


async def test_invoke_executes_mcp_tool():
    conn = await _connector()
    gw = ToolGateway(tools=[conn])
    agent = _mcp_agent(["get_run_logs"])
    inv = await gw.invoke(agent, "ci-logs.get_run_logs", {"run_id": "42"})
    assert inv.status == "executed"
    assert inv.result.data["text"] == "logs for 42"


async def test_runtime_loop_calls_mcp_tool():
    conn = await _connector()
    gw = ToolGateway(tools=[conn])
    agent = _mcp_agent(["get_run_logs", "list_runs"])
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "ci-logs_get_run_logs", {"run_id": "7"})]),
        ModelResponse(text="The build failed on step 3.", model="m"),
    ])
    runtime = Runtime(agent, gateway=gateway, tools=gw)
    result = await runtime.handle(Task(text="why did CI fail?", surface="slack",
                                       channel="#ci"))
    assert result.text == "The build failed on step 3."
    assert conn.client.calls == [("get_run_logs", {"run_id": "7"})]


async def test_unconfigured_mcp_tool_not_offered():
    # Before setup(), nothing is discovered, so no actions are exposed.
    gw = ToolGateway(tools=[MCPConnector("ci-logs", FakeMCPClient())])
    agent = _mcp_agent(["get_run_logs"])
    assert gw.available_actions(agent) == []
