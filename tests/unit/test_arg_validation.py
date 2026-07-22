"""The gateway's declared-schema validation seam (invoke-time, pre-approval).

Per-connector arg checks live in ``execute()``, which a workflow-backed tool
never runs — so the gateway enforces each action's own :class:`ActionSpec` on
the prepared args, on every path, before an approval request or workflow
instance exists. A request that can never run must not become an approval card,
a parked workflow, or a paid model call.
"""

from pathlib import Path

import pytest

from openloop.agents import load_agent
from openloop.agents.schema import Tool
from openloop.testing import (
    FakeGitHub,
    FakeWorkerOrchestrator,
)
from openloop.tools import ToolGateway
from openloop.tools.base import ActionSpec, ToolResult, validate_args
from openloop.tools.coding_worker import CodingWorkerConnector
from openloop.tools.github import GitHubConnector

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"

pytestmark = pytest.mark.unit


# --- the validator itself --------------------------------------------------

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "items": {"type": "array"},
    },
    "required": ["name"],
}


def test_validate_args_enforces_required_minlength_and_types():
    assert validate_args(_SCHEMA, {"name": "x"}) == []
    assert validate_args(_SCHEMA, {}) == ["missing required argument 'name'"]
    assert validate_args(_SCHEMA, {"name": ""}) == [
        "argument 'name' must not be empty"
    ]
    assert validate_args(_SCHEMA, {"name": "x", "count": "5"}) == [
        "argument 'count' must be of type integer"
    ]
    # bool is an int subclass in Python; JSON schema says it is not an integer.
    assert validate_args(_SCHEMA, {"name": "x", "count": True}) == [
        "argument 'count' must be of type integer"
    ]
    assert validate_args(_SCHEMA, {"name": "x", "ratio": 2}) == []  # int is a number
    assert validate_args(_SCHEMA, {"name": "x", "flag": 1}) == [
        "argument 'flag' must be of type boolean"
    ]
    assert validate_args(_SCHEMA, {"name": "x", "items": {}}) == [
        "argument 'items' must be of type array"
    ]
    # Undeclared extra args are the connectors' business (prepare_args mints
    # job_id/agent identity keys), never a validation failure.
    assert validate_args(_SCHEMA, {"name": "x", "job_id": "j1"}) == []


def test_validate_args_is_permissive_outside_its_subset():
    # Enforcement must never be stricter than what the validator provably
    # understands: foreign constructs and malformed declarations never reject.
    assert validate_args({}, {"anything": 1}) == []
    assert validate_args({"type": "object"}, {}) == []
    anyof = {
        "type": "object",
        "properties": {"x": {"anyOf": [{"type": "string"}]}},
        "required": ["x"],
    }
    assert validate_args(anyof, {"x": 5}) == []  # anyOf: not enforced
    unknown_type = {"type": "object", "properties": {"x": {"type": "date"}}}
    assert validate_args(unknown_type, {"x": 5}) == []
    malformed = {"type": "object", "required": "name", "properties": None}
    assert validate_args(malformed, {}) == []


# --- the gateway seam ------------------------------------------------------


async def test_whitespace_coding_worker_instruction_is_rejected():
    tools = ToolGateway(
        tools=[CodingWorkerConnector(FakeWorkerOrchestrator(), FakeGitHub())]
    )

    inv = await tools.invoke(
        load_agent(AGENT_YAML),
        "coding_worker.pr:write",
        # prepare_args canonicalizes to "" so minLength rejects whitespace too.
        {"repo": "acme/x", "instruction": "   "},
    )

    assert inv.status == "invalid"
    assert "instruction" in inv.message
    assert await tools.approvals.pending() == []


async def test_missing_github_title_is_rejected_without_an_approval():
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])

    inv = await tools.invoke(
        load_agent(AGENT_YAML), "github.issues:write", {"repo": "acme/x"}
    )

    assert inv.status == "invalid"
    assert "title" in inv.message
    assert await tools.approvals.pending() == []
    assert github.created == []


async def test_wrong_argument_type_is_rejected_before_execute():
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])

    inv = await tools.invoke(
        load_agent(AGENT_YAML),
        "github.issues:read",
        {"repo": "acme/x", "number": "5"},
    )

    assert inv.status == "invalid"
    assert "number" in inv.message


async def test_foreign_schema_constructs_never_reject_at_the_gateway():
    # An MCP server's richer schema (anyOf etc.) degrades to permissive: the
    # gateway must not refuse what it cannot provably validate.
    class LooseTool:
        name = "loose"

        def supported_permissions(self):
            return {"do"}

        def describe(self, permission):
            return ActionSpec(
                "loose",
                {
                    "type": "object",
                    "properties": {"x": {"anyOf": [{"type": "string"}]}},
                    "required": ["x"],
                },
            )

        async def execute(self, permission, args):
            return ToolResult(ok=True, summary="ran")

    agent = load_agent(AGENT_YAML)
    agent.spec.tools.append(Tool(name="loose", type="native", permissions=["do"]))
    tools = ToolGateway(tools=[LooseTool()])

    inv = await tools.invoke(agent, "loose.do", {"x": 123})

    assert inv.status == "executed"
    assert inv.result.ok
