"""The gateway's declared-schema validation seam (invoke-time, pre-approval).

Per-connector arg checks live in ``execute()``, which a workflow-backed tool
never runs — so the gateway enforces each action's own :class:`ActionSpec` on
the prepared args, on every path, before an approval request or workflow
instance exists. A request that can never run must not become an approval card,
a parked workflow, or a paid model call.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from openloop.analysis import (
    AnalysisReportArgs,
    InMemoryUploadStore,
    UploadRecord,
)
from openloop.agents import load_agent
from openloop.agents.schema import Tool
from openloop.testing import (
    FakeAnalysisOrchestrator,
    FakeGitHub,
    FakeWorkerOrchestrator,
)
from openloop.tools import ToolGateway
from openloop.tools.analysis_worker import AnalysisWorkerConnector
from openloop.tools.base import ActionSpec, ToolResult, validate_args
from openloop.tools.coding_worker import CodingWorkerConnector
from openloop.tools.github import GitHubConnector
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.analysis_worker import build_analysis_worker_workflow

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


def _analysis_agent():
    agent = load_agent(AGENT_YAML)
    agent.spec.tools.append(
        Tool(name="analysis", type="native", permissions=["report:write"])
    )
    agent.spec.approvals.require_for.append("analysis.report:write")
    return agent


def _analysis_tools():
    orch = FakeAnalysisOrchestrator()
    engine = WorkflowEngine(InMemoryWorkflowStore())
    engine.register(build_analysis_worker_workflow(orch))
    tools = ToolGateway(tools=[AnalysisWorkerConnector(orch)], engine=engine)
    return tools, engine, orch


async def test_blank_analysis_instruction_is_rejected_before_approval_and_workflow():
    tools, engine, orch = _analysis_tools()

    inv = await tools.invoke(
        _analysis_agent(),
        "analysis.report:write",
        # A field validator strips to "" — the typed parse's min_length rejects.
        {
            "instruction": "   ",
            "inputs": [{"source": "staged", "input_ref": "staged:one"}],
        },
    )

    assert inv.status == "invalid"
    assert "instruction" in inv.message
    assert inv.approval is None
    assert await tools.approvals.pending() == []  # nothing for a human to decide
    assert await engine.store.recent() == []  # no parked workflow instance
    assert orch.runs == []  # and certainly no run


async def test_missing_analysis_inputs_is_rejected():
    tools, engine, orch = _analysis_tools()

    inv = await tools.invoke(
        _analysis_agent(), "analysis.report:write", {"instruction": "summarize"}
    )

    assert inv.status == "invalid"
    assert "inputs" in inv.message
    assert await tools.approvals.pending() == []


async def test_unknown_input_source_is_rejected_by_the_discriminated_union():
    tools, engine, orch = _analysis_tools()

    inv = await tools.invoke(
        _analysis_agent(),
        "analysis.report:write",
        {"instruction": "summarize", "inputs": [{"source": "warehouse", "q": "1"}]},
    )

    assert inv.status == "invalid"
    assert await tools.approvals.pending() == []
    assert orch.runs == []


async def test_analysis_inputs_are_bounded_and_identity_is_not_model_suppliable():
    tools, engine, orch = _analysis_tools()
    entries = [
        {"source": "staged", "input_ref": f"staged:{i}"}
        for i in range(9)
    ]

    inv = await tools.invoke(
        _analysis_agent(),
        "analysis.report:write",
        {"instruction": "summarize", "inputs": entries},
    )

    assert inv.status == "invalid"
    assert "inputs" in inv.message
    schema = AnalysisReportArgs.model_json_schema()
    assert schema["properties"]["inputs"]["maxItems"] == 8
    for trusted in ("job_id", "attempt_id", "agent", "scope_key"):
        assert trusted not in schema["properties"]
    assert await engine.store.recent() == []
    assert orch.runs == []


async def test_upload_resolution_checks_scope_and_stamps_trusted_summary_metadata():
    scope = "slack\x1facme\x1fdev-platform\x1fC1\x1fT1"
    uploads = InMemoryUploadStore()
    await uploads.record(
        UploadRecord(
            upload_ref="F1",
            scope_key=scope,
            name="trusted.csv",
            size=42,
            user="U1",
            shared_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        )
    )
    orch = FakeAnalysisOrchestrator()
    engine = WorkflowEngine(InMemoryWorkflowStore())
    engine.register(build_analysis_worker_workflow(orch))
    tools = ToolGateway(
        tools=[
            AnalysisWorkerConnector(
                orch,
                uploads=uploads,
                available_sources={"staged", "upload"},
            )
        ],
        engine=engine,
    )

    inv = await tools.invoke(
        _analysis_agent(),
        "analysis.report:write",
        {
            "instruction": "summarize",
            "inputs": [{"source": "upload", "upload_ref": "F1"}],
        },
        warm_key=scope,
    )

    assert inv.status == "pending_approval"
    assert inv.approval.args["upload_meta"] == {
        "F1": {"name": "trusted.csv", "size": 42}
    }
    assert "`trusted.csv` shared in this thread" in inv.approval.summary


async def test_upload_resolution_refuses_scopeless_or_cross_thread_requests():
    scope = "slack\x1facme\x1fdev-platform\x1fC1\x1fT1"
    uploads = InMemoryUploadStore()
    await uploads.record(UploadRecord("F1", scope, "private.csv", 42))
    connector = AnalysisWorkerConnector(
        FakeAnalysisOrchestrator(),
        uploads=uploads,
        available_sources={"staged", "upload"},
    )
    tools = ToolGateway(tools=[connector])
    args = {
        "instruction": "summarize",
        "inputs": [{"source": "upload", "upload_ref": "F1"}],
    }

    scopeless = await tools.invoke(
        _analysis_agent(), "analysis.report:write", args
    )
    other_channel = await tools.invoke(
        _analysis_agent(),
        "analysis.report:write",
        args,
        warm_key="slack\x1facme\x1fdev-platform\x1fC2\x1fT1",
    )

    assert scopeless.status == other_channel.status == "invalid"
    assert await tools.approvals.pending() == []


async def test_valid_analysis_args_still_park_for_approval():
    tools, engine, orch = _analysis_tools()

    inv = await tools.invoke(
        _analysis_agent(),
        "analysis.report:write",
        {
            "instruction": "summarize the sales data",
            "inputs": [{"source": "staged", "input_ref": "staged:one"}],
        },
    )

    assert inv.status == "pending_approval"
    parked = await engine.store.get(inv.approval.args["job_id"])
    assert parked is not None and parked.status == "waiting"
    # The record is stamped with the args-contract version so a consumer can
    # refuse it after a breaking change.
    assert inv.approval.args_schema == 1
    assert parked.state["args_schema"] == 1


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
