"""Phase 2 — the sealed analysis worker as a durable workflow.

The workflow itself never touches a surface, an input store, or the artifact
store: ``run_analysis`` is one opaque unit through the orchestrator, and
``store_result`` only persists the ``{artifact_ref, prose_summary, spend}``
record the session runner later delivers from. The report body must never
appear in the durable workflow record.
"""

import json
from pathlib import Path

import pytest

from openloop.agents import load_agent
from openloop.agents.schema import Tool
from openloop.testing import FakeAnalysisOrchestrator
from openloop.tools import ToolGateway
from openloop.tools.analysis_worker import AnalysisWorkerConnector
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.analysis_worker import (
    WORKFLOW_NAME,
    _analysis_phase,
    build_analysis_worker_workflow,
)

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"

pytestmark = pytest.mark.unit


def _agent():
    agent = load_agent(AGENT_YAML)
    agent.spec.tools.append(
        Tool(name="analysis", type="native", permissions=["report:write"])
    )
    agent.spec.approvals.require_for.append("analysis.report:write")
    return agent


def _engine(orchestrator):
    engine = WorkflowEngine(InMemoryWorkflowStore())
    engine.register(build_analysis_worker_workflow(orchestrator))
    return engine


_ARGS = {
    "job_id": "job-1",
    "attempt_id": "attempt-1",
    "instruction": "summarize the sales data",
    "input_ref": "upload:one",
    "agent": "dev-platform",
}


async def test_workflow_parks_on_approval_then_stores_ref_result():
    orch = FakeAnalysisOrchestrator()
    engine = _engine(orch)

    parked = await engine.start(WORKFLOW_NAME, "job-1", dict(_ARGS))
    assert parked.status == "waiting"
    assert parked.waiting_on == "await_approval"
    assert orch.runs == []  # nothing runs before the approval event

    done = await engine.send_event("job-1", "await_approval", {"approver": "@a"})

    assert done.status == "completed"
    result = done.result
    assert result["deliverable"] == "artifact"
    assert result["artifact_ref"] == "analysis://job-1/report.md"
    assert result["prose_summary"] == orch.prose
    assert result["artifact_filename"] == "report.md"
    assert result["snippet_type"] == "markdown"
    assert result["cost_usd"] == orch.cost_usd
    assert result["prompt_tokens"] == orch.prompt_tokens
    assert "report ready" in result["summary"]
    # Identity threads through: the orchestrator saw the persisted job/attempt
    # and the stamped invoking agent — spend attribution survives the park.
    state = orch.runs[0]
    assert (state.job_id, state.attempt_id) == ("job-1", "attempt-1")
    assert state.agent == "dev-platform"
    # The report BODY never lands in the durable record — only the ref and the
    # replay-safe prose summary do (the body's tail extends past the prose).
    stored = await engine.store.get("job-1")
    persisted = json.dumps(stored.state) + json.dumps(stored.result)
    assert "FULL-REPORT-TAIL" not in persisted
    # The artifact was written exactly once, by the orchestrator, keyed by job.
    artifact = await orch.artifacts.get(result["artifact_ref"])
    assert artifact.body == orch.body


async def test_workflow_failure_is_terminal_failed():
    orch = FakeAnalysisOrchestrator(
        error="analysis attempt attempt-1 is already charged"
    )
    engine = _engine(orch)
    await engine.start(WORKFLOW_NAME, "job-2", dict(_ARGS, job_id="job-2"))

    done = await engine.send_event("job-2", "await_approval", {})

    assert done.status == "failed"
    assert "already charged" in done.error


async def test_progress_phrase_lands_in_state_and_covers_every_milestone():
    orch = FakeAnalysisOrchestrator()
    engine = _engine(orch)
    await engine.start(WORKFLOW_NAME, "job-3", dict(_ARGS, job_id="job-3"))
    done = await engine.send_event("job-3", "await_approval", {})

    # The last checkpointed phrase is the final milestone's.
    assert done.state["progress"] == "is finalizing the report…"

    # And the mapping tracks each orchestrator milestone in order.
    reached: list[str] = []
    phases = [_analysis_phase(reached)]
    for step in FakeAnalysisOrchestrator.STEPS:
        reached.append(step)
        phases.append(_analysis_phase(reached))
    assert phases == [
        "is provisioning the inputs…",
        "is writing the analysis program…",
        "is running the sealed analysis…",
        "is reading the results…",
        "is storing the report…",
        "is finalizing the report…",
    ]
    # The surface's progress relay expects this phrasing shape.
    assert all(p.startswith("is ") and p.endswith("…") for p in phases)


async def test_stale_record_with_empty_instruction_fails_closed_at_run_time():
    # The gateway rejects these at invoke() time now; this pins the backstop
    # for args that re-enter from persisted records (an approval or parked
    # workflow written before the validation seam existed): the orchestrator
    # refuses before the ledger, the input store, or any model call.
    from openloop.analysis import InMemoryArtifactStore, InMemoryInputStore
    from openloop.tools.analysis_worker import SealedAnalysisOrchestrator

    class _NeverRunsWorker:
        async def run(self, workspace, state, on_step=None, on_charge=None):
            raise AssertionError("a run without an instruction must never execute")

    orch = SealedAnalysisOrchestrator(
        _NeverRunsWorker(), InMemoryInputStore(), InMemoryArtifactStore()
    )
    engine = _engine(orch)
    await engine.start(
        WORKFLOW_NAME, "job-stale", dict(_ARGS, job_id="job-stale", instruction="")
    )

    done = await engine.send_event("job-stale", "await_approval", {})

    assert done.status == "failed"
    assert "instruction is required" in done.error


async def test_gateway_approval_event_drives_analysis_workflow():
    orch = FakeAnalysisOrchestrator()
    engine = _engine(orch)
    tools = ToolGateway(tools=[AnalysisWorkerConnector(orch)], engine=engine)
    agent = _agent()

    inv = await tools.invoke(
        agent,
        "analysis.report:write",
        {"instruction": "summarize the sales data", "input_ref": "upload:one"},
    )

    assert inv.status == "pending_approval"
    job_id = inv.approval.args["job_id"]
    parked = await engine.store.get(job_id)
    assert parked is not None and parked.status == "waiting"
    assert orch.runs == []

    resolved = await tools.resolve(inv.approval.id, "@maciag.artur", approve=True)
    assert resolved.status == "started"
    done = await engine.wait_background(job_id)

    assert done.status == "completed"
    assert done.result["artifact_ref"] == f"analysis://{job_id}/report.md"
    # prepare_args identity crossed the approval boundary intact.
    assert orch.runs[0].agent == "dev-platform"
    assert orch.runs[0].attempt_id == inv.approval.args["attempt_id"]


async def test_denied_approval_cancels_parked_analysis_workflow():
    orch = FakeAnalysisOrchestrator()
    engine = _engine(orch)
    tools = ToolGateway(tools=[AnalysisWorkerConnector(orch)], engine=engine)

    inv = await tools.invoke(
        _agent(),
        "analysis.report:write",
        {"instruction": "summarize the sales data", "input_ref": "upload:one"},
    )
    job_id = inv.approval.args["job_id"]

    resolved = await tools.resolve(inv.approval.id, "@maciag.artur", approve=False)

    assert resolved.status == "denied"
    assert (await engine.store.get(job_id)).status == "cancelled"
    assert orch.runs == []  # a denied request never provisions or spends
