"""Integration: the coding worker as a durable workflow through the gateway.

Phase C — approval is a wait node, and ToolGateway.resolve is a thin adapter that
emits the approval event to wake the parked workflow.
"""

from pathlib import Path
from openloop.agents import load_agent
from openloop.tools import ToolGateway
from openloop.tools.coding_worker import CodingWorkerConnector
from openloop.tools.coding_worker import WorkerOutcome
from openloop.tools.openhands_artifacts import (
    WorkspaceArtifact,
    WorkspaceArtifactIdentity,
)
from openloop.tools.openhands_docker import DEFAULT_OPENHANDS_SERVER_IMAGE
from openloop.tools.openhands_resume import (
    OpenHandsResumeState,
    WorkerPaused,
    WorkspaceArtifactRef,
)
from openloop.tools.github import GitHubConnector
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.coding_worker import _worker_phase, build_coding_worker_workflow
from openloop.testing import FakeGitHub, FakeWorkerOrchestrator

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _agent():
    return load_agent(AGENT_YAML)


def test_worker_phase_maps_latest_milestone():
    # Reads the most-advanced milestone reached, not merely the last appended.
    assert _worker_phase([]) == "is starting…"
    assert _worker_phase(["clone"]) == "is setting up the workspace…"
    assert _worker_phase(["clone", "branch"]) == "is working on the changes…"
    assert _worker_phase(["clone", "branch", "edit"]) == "is finalizing the changes…"
    assert _worker_phase(["clone", "branch", "edit", "commit"]) == (
        "is committing the changes…"
    )
    assert _worker_phase(["clone", "branch", "edit", "commit", "push"]) == (
        "is pushing the branch…"
    )


def _setup(runner=None, github=None):
    # Both durable paths share ONE orchestrator (the Phase 2 convergence
    # invariant): the workflow and the connector fallback run attempts
    # through the same AttemptRunner.
    runner = runner or FakeWorkerOrchestrator(title="Add retries")
    github = github or FakeGitHub()
    store = InMemoryWorkflowStore()
    engine = WorkflowEngine(store)
    engine.register(build_coding_worker_workflow(runner, github))
    gw = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(runner, github)],
        engine=engine,
    )
    return gw, engine, store, runner, github


async def test_invoke_parks_workflow_at_approval():
    gw, engine, store, runner, github = _setup()
    inv = await gw.invoke(
        _agent(), "coding_worker.pr:write",
        {"repo": "acme/x", "instruction": "add retries"},
    )
    assert inv.status == "pending_approval"
    job_id = inv.approval.args["job_id"]

    inst = await store.get(job_id)
    assert inst is not None
    assert inst.status == "waiting"
    assert inst.waiting_on == "await_approval"
    # Nothing ran before approval.
    assert runner.runs == []
    assert github.pulls == []


async def test_approval_event_drives_worker_and_opens_pr():
    gw, engine, store, runner, github = _setup()
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write",
        {"repo": "acme/x", "instruction": "add retries"},
    )
    job_id = pending.approval.args["job_id"]

    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "started"
    assert resolved.result.ok
    assert resolved.result.data["instance_id"] == job_id
    assert resolved.result.data["status"] == "running"
    done = await engine.wait_background(job_id)
    # The workflow completed, the worker ran once, one draft PR opened.
    assert done.status == "completed"
    assert done.result["job_id"] == job_id
    assert done.result["pr_number"] == 1
    assert runner.runs[0].job_id == job_id
    assert github.pulls[0]["draft"] is True
    assert github.pulls[0]["head"] == f"openloop/job-{job_id}"


async def test_resolve_recovers_when_workflow_was_never_started():
    # P1 create-side gap: invoke() crashed after creating the approval but before
    # starting the workflow (instance missing). resolve() must idempotently
    # ensure-start the workflow, then drive it — not get stuck.
    gw, engine, store, runner, github = _setup()
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    job_id = pending.approval.args["job_id"]
    store._by_id.pop(job_id)  # simulate the lost workflow start
    assert await store.get(job_id) is None

    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "started" and resolved.result.ok
    done = await engine.wait_background(job_id)
    assert done.status == "completed"
    assert len(github.pulls) == 1


async def test_result_includes_full_spend_telemetry():
    # P2: the workflow result must carry prompt/completion tokens, not just cost,
    # matching the connector's execute() telemetry.
    costing = FakeWorkerOrchestrator(
        title="t", body="b", cost_usd=0.2, prompt_tokens=120, completion_tokens=40
    )
    gw, engine, store, runner, github = _setup(runner=costing)
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "started"
    await engine.wait_background(pending.approval.args["job_id"])
    data = (await store.get(pending.approval.args["job_id"])).result
    assert data["cost_usd"] == 0.2
    assert data["prompt_tokens"] == 120
    assert data["completion_tokens"] == 40


async def test_denied_approval_cancels_workflow():
    gw, engine, store, runner, github = _setup()
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    job_id = pending.approval.args["job_id"]

    inv = await gw.resolve(pending.approval.id, "@maciag.artur", approve=False)

    assert inv.status == "denied"
    assert (await store.get(job_id)).status == "cancelled"
    assert runner.runs == []
    assert github.pulls == []


async def test_open_pr_failure_marks_workflow_failed():
    class FlakyGitHub(FakeGitHub):
        async def create_pull(self, *a, **k):
            raise RuntimeError("422 blip")

    gw, engine, store, runner, github = _setup(github=FlakyGitHub())
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    job_id = pending.approval.args["job_id"]

    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "started"
    assert resolved.result.ok
    await engine.wait_background(job_id)
    inst = await store.get(job_id)
    assert inst.status == "failed"
    assert "422 blip" in inst.error
    # The worker step still completed (branch pushed); only PR open failed.
    assert "run_worker" in inst.completed_steps


async def test_resume_after_crash_between_approval_and_pr():
    # Simulate a crash during the post-approval drive: run_worker completes and is
    # checkpointed, then the process dies before open_pr. The startup reconciler
    # re-drives the instance (left "running") and finishes the PR.
    gw, engine, store, runner, github = _setup()
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    job_id = pending.approval.args["job_id"]

    # Hand-roll the crash: wake the workflow, but make open_pr raise *and* leave
    # the instance marked running (as a hard crash would, before failure persists).
    inst = await store.get(job_id)
    inst.status = "running"
    inst.waiting_on = None
    inst.completed_steps = ["await_approval", "run_worker"]
    inst.state.update({
        "branch": f"openloop/job-{job_id}", "title": "t", "body": "b",
        "repo": "acme/x", "instruction": "x", "base": "main", "job_id": job_id,
    })
    await store.upsert(inst)

    resumed = await engine.resume_incomplete()
    assert job_id in resumed
    assert (await store.get(job_id)).status == "completed"
    assert len(github.pulls) == 1  # PR opened on resume
    assert runner.runs == []  # run_worker was already done; not re-run


async def test_workflow_can_park_repeatedly_on_typed_openhands_decision():
    class PausingRunner:
        def __init__(self):
            self.decisions = []

        async def run_attempt(self, state, on_step=None):
            resume = OpenHandsResumeState(
                status="running",
                conversation_id="conversation-1",
                segment_id="segment-1",
                base_ref="main",
                resolved_base_commit="a" * 40,
                image_digest=DEFAULT_OPENHANDS_SERVER_IMAGE,
                master_key_id="key-v1",
                slack_requester_id="maciag.artur",
            )
            identity = WorkspaceArtifactIdentity(
                state.job_id, "conversation-1", "segment-1", "paused"
            )
            artifact = WorkspaceArtifactRef(
                WorkspaceArtifact(
                    identity=identity,
                    key=(
                        f"jobs/{state.job_id}/artifacts/conversation-1/"
                        "segment-1.paused.artifact"
                    ),
                    ciphertext_sha256="b" * 64,
                    ciphertext_bytes=10,
                    envelope_version=1,
                    master_key_id="key-v1",
                ),
                "git-delta",
                "a" * 40,
            )
            paused = WorkerPaused(
                "conversation-1",
                "segment-1",
                "decision-1",
                "Run terminal",
                "c" * 64,
                artifact,
            )
            resume.transition_to(
                "parking",
                decision_id=paused.decision_id,
                pending_action_summary=paused.pending_action_summary,
                pending_action_fingerprint=paused.pending_action_fingerprint,
                workspace_artifact=artifact,
            )
            resume.transition_to("parked")
            state.openhands_resume = resume
            if on_step:
                await on_step(state)
            return paused

        async def resume_attempt(self, state, decision, on_step=None):
            self.decisions.append(decision)
            state.openhands_resume = None
            return WorkerOutcome(state.branch, "Done", "Body")

    runner = PausingRunner()
    gw, engine, store, _, github = _setup(runner=runner)
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"}
    )
    job_id = pending.approval.args["job_id"]
    await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)
    parked = await engine.wait_background(job_id)

    assert parked.status == "waiting"
    assert parked.waiting_on == "openhands_decision:decision-1"
    assert github.pulls == []

    done = await engine.send_event(
        job_id,
        "openhands_decision:decision-1",
        {
            "kind": "accept",
            "decision_id": "decision-1",
            "event_id": "Ev123",
            "actor_id": "maciag.artur",
        },
    )
    assert done.status == "completed"
    assert runner.decisions[0].kind == "accept"
    assert len(github.pulls) == 1
