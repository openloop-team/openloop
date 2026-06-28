"""Integration: the coding worker as a durable workflow through the gateway.

Phase C — approval is a wait node, and ToolGateway.resolve is a thin adapter that
emits the approval event to wake the parked workflow.
"""

from openloop.agents import load_agent
from openloop.tools import ToolGateway
from openloop.tools.coding_worker import CodingWorkerConnector
from openloop.tools.github import GitHubConnector
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.coding_worker import build_coding_worker_workflow
from openloop.testing import EXAMPLE_AGENT, FakeCodingWorker, FakeGitHub


def _agent():
    return load_agent(EXAMPLE_AGENT)


def _setup(worker=None, github=None):
    worker = worker or FakeCodingWorker(title="Add retries")
    github = github or FakeGitHub()
    store = InMemoryWorkflowStore()
    engine = WorkflowEngine(store)
    engine.register(build_coding_worker_workflow(worker, github))
    gw = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(worker, github)],
        engine=engine,
    )
    return gw, engine, store, worker, github


async def test_invoke_parks_workflow_at_approval():
    gw, engine, store, worker, github = _setup()
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
    assert worker.runs == []
    assert github.pulls == []


async def test_approval_event_drives_worker_and_opens_pr():
    gw, engine, store, worker, github = _setup()
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write",
        {"repo": "acme/x", "instruction": "add retries"},
    )
    job_id = pending.approval.args["job_id"]

    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "executed"
    assert resolved.result.ok
    assert resolved.result.data["job_id"] == job_id
    assert resolved.result.data["pr_number"] == 1
    # The workflow completed, the worker ran once, one draft PR opened.
    assert (await store.get(job_id)).status == "completed"
    assert worker.runs[0].job_id == job_id
    assert github.pulls[0]["draft"] is True
    assert github.pulls[0]["head"] == f"openloop/job-{job_id}"


async def test_resolve_recovers_when_workflow_was_never_started():
    # P1 create-side gap: invoke() crashed after creating the approval but before
    # starting the workflow (instance missing). resolve() must idempotently
    # ensure-start the workflow, then drive it — not get stuck.
    gw, engine, store, worker, github = _setup()
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    job_id = pending.approval.args["job_id"]
    store._by_id.pop(job_id)  # simulate the lost workflow start
    assert await store.get(job_id) is None

    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "executed" and resolved.result.ok
    assert (await store.get(job_id)).status == "completed"
    assert len(github.pulls) == 1


async def test_result_includes_full_spend_telemetry():
    # P2: the workflow result must carry prompt/completion tokens, not just cost,
    # matching the connector's execute() telemetry.
    class CostingWorker:
        def __init__(self):
            self.runs = []

        async def run(self, state, on_step=None):
            from openloop.tools.coding_worker import STEPS, WorkerOutcome

            state.completed_steps.extend(STEPS)
            self.runs.append(state)
            return WorkerOutcome(
                branch=state.branch, title="t", body="b",
                cost_usd=0.2, prompt_tokens=120, completion_tokens=40,
            )

    gw, engine, store, worker, github = _setup(worker=CostingWorker())
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    data = resolved.result.data
    assert data["cost_usd"] == 0.2
    assert data["prompt_tokens"] == 120
    assert data["completion_tokens"] == 40


async def test_denied_approval_cancels_workflow():
    gw, engine, store, worker, github = _setup()
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    job_id = pending.approval.args["job_id"]

    inv = await gw.resolve(pending.approval.id, "@maciag.artur", approve=False)

    assert inv.status == "denied"
    assert (await store.get(job_id)).status == "cancelled"
    assert worker.runs == []
    assert github.pulls == []


async def test_open_pr_failure_marks_workflow_failed():
    class FlakyGitHub(FakeGitHub):
        async def create_pull(self, *a, **k):
            raise RuntimeError("422 blip")

    gw, engine, store, worker, github = _setup(github=FlakyGitHub())
    pending = await gw.invoke(
        _agent(), "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"},
    )
    job_id = pending.approval.args["job_id"]

    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "executed"
    assert not resolved.result.ok
    assert "failed" in resolved.result.summary
    inst = await store.get(job_id)
    assert inst.status == "failed"
    # The worker step still completed (branch pushed); only PR open failed.
    assert "run_worker" in inst.completed_steps


async def test_resume_after_crash_between_approval_and_pr():
    # Simulate a crash during the post-approval drive: run_worker completes and is
    # checkpointed, then the process dies before open_pr. The startup reconciler
    # re-drives the instance (left "running") and finishes the PR.
    gw, engine, store, worker, github = _setup()
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
    assert worker.runs == []  # run_worker was already done; not re-run
