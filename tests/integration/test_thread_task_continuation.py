"""Integration: a Slack thread is bound to one continuable workspace task.

The four properties this exercises end to end, over the same durable stores the
runtime wires in production (in-memory backends, same Protocols):

1. a task begun in a thread continues from a later reply without the model
   delegating a second task;
2. task identity outlives the workflow instance and the surface session that
   started it;
3. continuation survives a restart — a cold runtime, gateway, and engine rebuilt
   over the same stores continue the task with no warm state;
4. approvals, budget attribution, and branch/PR identity are preserved across
   continuations.
"""

from pathlib import Path

import pytest

from openloop.agents import load_agent
from openloop.memory import InMemoryStore
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.sessions import (
    InMemorySurfaceSessionStore,
    InMemoryThreadRecordStore,
    SessionRunner,
    SurfaceTarget,
    thread_scope_key,
)
from openloop.tasks import BUSY, InMemoryThreadTaskStore
from openloop.testing import (
    FakeGitHub,
    FakeSurfaceDelivery,
    FakeWorkerOrchestrator,
    ScriptedGateway,
    tool_call_response,
)
from openloop.tools import ToolGateway
from openloop.tools.coding_worker import CodingWorkerConnector
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.coding_worker import build_workspace_task_workflow

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"
APPROVER = "@maciag.artur"

pytestmark = pytest.mark.integration


class Stores:
    """The durable half of the runtime: what survives a restart."""

    def __init__(self) -> None:
        self.sessions = InMemorySurfaceSessionStore()
        self.threads = InMemoryThreadRecordStore()
        self.tasks = InMemoryThreadTaskStore()
        self.workflows = InMemoryWorkflowStore()
        self.approvals = None  # set by the first process (shared thereafter)


def _target(event_id: str) -> SurfaceTarget:
    return SurfaceTarget(
        surface="slack",
        workspace="acme",
        agent="dev-platform",
        channel="C1",
        thread="100.1",
        event_id=event_id,
    )


def _process(stores: Stores, responses, *, orchestrator=None, github=None):
    """Compose one 'process' over the durable stores — a restart rebuilds this.

    Everything here is process-local (engine, tool gateway, runtime, runner);
    only ``stores`` crosses the boundary, so anything a continuation needs must
    come from those rows.
    """
    orchestrator = orchestrator or FakeWorkerOrchestrator(title="Add retries")
    github = github or FakeGitHub()
    engine = WorkflowEngine(stores.workflows)
    engine.register(build_workspace_task_workflow(orchestrator, github))
    model = ScriptedGateway(list(responses))
    tools = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(orchestrator, github)],
        engine=engine,
        tasks=stores.tasks,
    )
    if stores.approvals is None:
        stores.approvals = tools.approvals
    else:
        tools.approvals = stores.approvals
    runtime = Runtime(
        load_agent(AGENT_YAML),
        gateway=model,
        tools=tools,
        usage=InMemoryUsageStore(),
        memory=InMemoryStore(),
        engine=engine,
    )
    delivery = FakeSurfaceDelivery()
    runner = SessionRunner(
        runtime, stores.sessions, delivery, threads=stores.threads, tasks=stores.tasks
    )
    return runner, engine, tools, model, delivery, orchestrator, github


def _delegate_call():
    """The model round that delegates the coding task."""
    return tool_call_response(
        "m",
        [
            (
                "c1",
                "workspace_task_code_write",
                {"repo": "acme/x", "instruction": "add retries"},
            )
        ],
    )


def _delegation(text="I'll start on that."):
    """A delegating first turn, plus the answer its approval continuation gives."""
    return [_delegate_call(), ModelResponse(text=text, model="m")]


async def _start_task(stores: Stores, *, user="U1", orchestrator=None, github=None):
    """Run the gated first turn all the way to an opened draft PR."""
    runner, engine, tools, model, delivery, orchestrator, github = _process(
        stores, _delegation(), orchestrator=orchestrator, github=github
    )
    await runner.run_threaded(
        Task(text="fix the retries", surface="slack", channel="C1", user=user),
        _target("ev1"),
    )
    session = await stores.sessions.get_by_event("ev1")
    assert session.status == "waiting"
    approval_id = session.approval_ids[0]
    await runner.resolve_approval(approval_id, APPROVER, approve=True)
    request = await stores.approvals.get(approval_id)
    await engine.wait_background(request.workflow_instance_id)
    return runner, engine, tools, model, delivery, orchestrator, github


async def test_a_reply_continues_the_same_task_without_re_delegation():
    stores = Stores()
    runner, engine, tools, model, delivery, orchestrator, github = await _start_task(
        stores
    )
    first = await stores.sessions.get_by_event("ev1")
    record = await stores.tasks.bound(thread_scope_key(_target("ev1")))
    assert record is not None
    assert record.task_id == orchestrator.runs[0].job_id
    model_rounds = len(model.calls)

    await runner.run_threaded(
        Task(text="also handle nulls", surface="slack", channel="C1", user="U1"),
        _target("ev2"),
    )

    # (1) The reply ran as the task's next turn: no model round was spent on it,
    # so nothing could have delegated a second task.
    assert len(model.calls) == model_rounds
    assert len(orchestrator.runs) == 2
    second = await stores.sessions.get_by_event("ev2")
    assert second.status == "completed"
    assert second.task_id == record.task_id

    # (4) identity, authorization, and attribution are preserved …
    started, continued = orchestrator.runs
    assert continued.job_id == started.job_id
    assert continued.branch == started.branch
    assert continued.approval_id == started.approval_id
    assert continued.agent == started.agent
    assert continued.agent_id == started.agent_id
    assert continued.requester_id == started.requester_id
    # …while the per-turn attribution follows the session that asked for it.
    assert continued.session_id == second.id
    # … and exactly one approval was ever raised for the task.
    assert len(stores.approvals._by_id) == 1
    # … over one pull request, updated rather than re-opened.
    assert len(github.pulls) == 1
    assert second.result_summary.startswith("updated draft PR #1")
    # The continuation builds on the branch it already pushed.
    assert continued.base == started.branch
    assert "also handle nulls" in continued.instruction

    # (2) identity outlived the first turn's instance and session.
    assert second.workflow_instance_id != first.workflow_instance_id
    refreshed = await stores.tasks.get(record.task_id)
    assert refreshed.turns == 2
    assert refreshed.state["profile_state"]["code"]["pr_number"] == 1


async def test_continuation_survives_a_restart_and_reconstructs_cold():
    stores = Stores()
    _, _, _, _, _, orchestrator, github = await _start_task(stores)

    # A new process: new engine, new gateway, new runtime, new runner, and a
    # worker orchestrator with no memory of the first turn. Only the durable
    # rows cross over.
    cold_worker = FakeWorkerOrchestrator(title="Add retries")
    runner, engine, tools, model, delivery, *_ = _process(
        stores, [], orchestrator=cold_worker, github=github
    )
    await runner.reconcile()

    await runner.run_threaded(
        Task(text="also handle nulls", surface="slack", channel="C1", user="U1"),
        _target("ev2"),
    )

    assert len(cold_worker.runs) == 1  # the cold process ran the continuation
    continued = cold_worker.runs[0]
    assert continued.job_id == orchestrator.runs[0].job_id
    assert continued.branch == orchestrator.runs[0].branch
    assert continued.approval_id == orchestrator.runs[0].approval_id
    assert len(github.pulls) == 1
    session = await stores.sessions.get_by_event("ev2")
    assert session.status == "completed"
    assert delivery.finals[-1]["text"].startswith("updated draft PR #1")


async def test_only_the_initiating_human_continues_the_task():
    stores = Stores()
    await _start_task(stores, user="U1")

    # Somebody else's reply is an ordinary turn: the model answers it, and the
    # task is untouched.
    runner, engine, tools, model, delivery, orchestrator, github = _process(
        stores, [ModelResponse(text="that's being worked on", model="m")]
    )
    await runner.run_threaded(
        Task(text="what's the status?", surface="slack", channel="C1", user="U2"),
        _target("ev2"),
    )

    assert orchestrator.runs == []  # no continuation ran
    session = await stores.sessions.get_by_event("ev2")
    assert session.status == "completed"
    assert session.task_id is None
    assert session.result_summary == "that's being worked on"
    record = await stores.tasks.bound(thread_scope_key(_target("ev1")))
    assert record.turns == 1


async def test_a_reply_waits_for_the_turn_already_driving_the_task():
    stores = Stores()
    runner, engine, tools, model, delivery, orchestrator, github = await _start_task(
        stores
    )
    record = await stores.tasks.bound(thread_scope_key(_target("ev1")))
    # Simulate a turn already in flight on this task (another replica, or a
    # long-running attempt this process started).
    claimed = await stores.tasks.claim(
        record.task_id, instance_id="busy-instance", session_id="busy"
    )
    assert claimed.status == BUSY

    await runner.run_threaded(
        Task(text="also handle nulls", surface="slack", channel="C1", user="U1"),
        _target("ev2"),
    )

    # Nothing ran and nothing was delivered: the reply is durably queued.
    assert len(orchestrator.runs) == 1
    assert await stores.sessions.get_by_event("ev2") is None
    pending = await stores.threads.next_inbox(_target("ev2"))
    assert pending is not None and pending.event_id == "ev2"
    await stores.threads.append_inbox(_target("ev2"), "ev2", pending.payload)

    # Once the task is free again, the deferred reply is drained and continues it.
    await stores.tasks.release(record.task_id)
    await runner._drain_thread(_target("ev2"))

    assert len(orchestrator.runs) == 2
    assert orchestrator.runs[1].job_id == record.task_id
    assert (await stores.sessions.get_by_event("ev2")).status == "completed"


async def test_a_continuation_reopens_a_lost_pr_against_the_original_base():
    stores = Stores()
    _, _, _, _, _, orchestrator, github = await _start_task(stores)
    # Somebody closed the draft PR (or the first turn's open_pr failed after the
    # push): the continuation has to raise a new one — against the base the task
    # targets, never against its own head.
    github.pulls.clear()

    runner, *_ = _process(stores, [], orchestrator=orchestrator, github=github)
    await runner.run_threaded(
        Task(text="also handle nulls", surface="slack", channel="C1", user="U1"),
        _target("ev2"),
    )

    assert len(github.pulls) == 1
    reopened = github.pulls[0]
    assert reopened["head"] == orchestrator.runs[0].branch
    assert reopened["base"] == "main"
    assert (await stores.sessions.get_by_event("ev2")).status == "completed"


async def test_a_denied_task_is_never_continued():
    stores = Stores()
    # A denied first turn never reaches a model continuation, so the second
    # scripted response belongs to the reply that follows the denial.
    runner, engine, tools, model, delivery, orchestrator, github = _process(
        stores, [_delegate_call(), ModelResponse(text="anything else?", model="m")]
    )
    await runner.run_threaded(
        Task(text="fix the retries", surface="slack", channel="C1", user="U1"),
        _target("ev1"),
    )
    session = await stores.sessions.get_by_event("ev1")
    await runner.resolve_approval(session.approval_ids[0], APPROVER, approve=False)

    scope = thread_scope_key(_target("ev1"))
    assert await stores.tasks.bound(scope) is None  # the binding is retired

    await runner.run_threaded(
        Task(text="try it anyway", surface="slack", channel="C1", user="U1"),
        _target("ev2"),
    )

    # The reply ran as an ordinary model turn; the denied task never ran.
    assert orchestrator.runs == []
    assert (await stores.sessions.get_by_event("ev2")).result_summary == (
        "anything else?"
    )


async def test_recovery_drains_a_reply_a_crash_left_queued():
    stores = Stores()
    runner, _, _, _, _, orchestrator, github = await _start_task(stores)
    record = await stores.tasks.bound(thread_scope_key(_target("ev1")))
    # A turn claimed the task and died with it: the claim points at an instance
    # that no longer exists, and the reply that arrived meanwhile is queued with
    # nobody left to drain it.
    await stores.tasks.claim(record.task_id, instance_id="lost", session_id="gone")
    await runner.run_threaded(
        Task(text="also handle nulls", surface="slack", channel="C1", user="U1"),
        _target("ev2"),
    )
    assert await stores.sessions.get_by_event("ev2") is None

    # A fresh process recovers: the orphaned claim is freed and the queued reply
    # is finally run as the task's next turn.
    cold_worker = FakeWorkerOrchestrator(title="Add retries")
    cold, *_ = _process(stores, [], orchestrator=cold_worker, github=github)
    repaired = await cold.reconcile()

    assert record.task_id in repaired
    assert len(cold_worker.runs) == 1
    assert cold_worker.runs[0].job_id == record.task_id
    session = await stores.sessions.get_by_event("ev2")
    assert session.status == "completed"
    assert await stores.threads.pending_scopes() == []


async def test_a_crashed_turn_releases_its_claim_on_recovery():
    stores = Stores()
    runner, *_ = await _start_task(stores)
    record = await stores.tasks.bound(thread_scope_key(_target("ev1")))
    # A turn that died mid-flight: claimed, pointing at an instance that no
    # longer exists anywhere durable.
    await stores.tasks.claim(record.task_id, instance_id="lost", session_id="gone")

    repaired = await runner.reconcile()

    assert record.task_id in repaired
    assert (await stores.tasks.get(record.task_id)).status != BUSY
