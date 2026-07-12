"""Unit tests for Phase D — surface sessions, delivery, and the session runner.

Covers the session-store state transitions + event dedup, and the runner's
mention → progress → final / waiting / interrupted flows with idempotent delivery
(a duplicate event never starts a second turn or posts a second answer).
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openloop.agents import load_agent
from openloop.agents.schema import Tool
from openloop.analysis import InMemoryArtifactStore
from openloop.deliverable import Artifact, Prose
from openloop.memory import InMemoryStore
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.sessions import (
    InMemorySurfaceSessionStore,
    InMemoryThreadRecordStore,
    SessionRunner,
    SurfaceSession,
    SurfaceTarget,
    TranscriptFragment,
    thread_scope_key,
)
import time

from openloop.sessions.runner import PROGRESS_REFRESH_SECONDS, PROGRESS_STATUS_TEXT
from openloop.tools import ToolGateway
from openloop.tools.analysis_worker import AnalysisWorkerConnector
from openloop.tools.coding_worker import CodingWorkerConnector
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine, WorkflowInstance
from openloop.workflows.analysis_worker import build_analysis_worker_workflow
from openloop.workflows.coding_worker import build_coding_worker_workflow
from openloop.testing import (
    FakeAnalysisOrchestrator,
    FakeGitHub,
    FakeSurfaceDelivery,
    FakeWorkerOrchestrator,
    ScriptedGateway,
    tool_call_response,
)

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"

pytestmark = pytest.mark.unit


def _target(event_id="ev1"):
    return SurfaceTarget(
        surface="slack",
        workspace="acme",
        agent="dev-platform",
        channel="C1",
        thread="100.1",
        event_id=event_id,
    )


def _task(text="hi"):
    return Task(text=text, surface="slack", channel="C1", user="U1")


def _runner(model_gateway, *, tools=None, delivery=None, threads=None):
    sessions = InMemorySurfaceSessionStore()
    delivery = delivery or FakeSurfaceDelivery()
    engine = WorkflowEngine(InMemoryWorkflowStore())
    runtime = Runtime(
        load_agent(AGENT_YAML),
        gateway=model_gateway,
        tools=tools,
        usage=InMemoryUsageStore(),
        memory=InMemoryStore(),
        engine=engine,
    )
    return SessionRunner(runtime, sessions, delivery, threads=threads), sessions, delivery


# --- session store -------------------------------------------------------

async def test_store_upsert_get_and_event_lookup():
    store = InMemorySurfaceSessionStore()
    session = SurfaceSession(id="s1", target=_target("ev-abc"))
    await store.upsert(session)

    assert (await store.get("s1")).id == "s1"
    assert (await store.get_by_event("ev-abc")).id == "s1"
    assert await store.get_by_event("nope") is None
    assert await store.get_by_event("") is None


async def test_store_upsert_preserves_created_at_bumps_updated_at():
    store = InMemorySurfaceSessionStore()
    session = SurfaceSession(id="s1", target=_target())
    await store.upsert(session)
    created = session.created_at

    session.status = "completed"
    await store.upsert(session)

    stored = await store.get("s1")
    assert stored.status == "completed"
    assert stored.created_at == created
    assert stored.updated_at >= created


# --- runner: happy path --------------------------------------------------

async def test_mention_to_progress_then_final():
    runner, sessions, delivery = _runner(
        ScriptedGateway([ModelResponse(text="here you go", model="m")])
    )

    session = await runner.run(_task(), _target())

    assert session.status == "completed"
    assert session.result_summary == "here you go"
    # A transient thinking indicator is set, then a single final answer is posted.
    assert delivery.statuses[0]["text"] == "is thinking..."
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["text"] == "here you go"
    # The final message id is persisted on the session, and the workflow shares
    # its id. The transient status indicator has no durable message id.
    assert session.progress_message_id is None
    assert session.final_message_id == delivery.finals[0]["id"]
    assert session.workflow_instance_id == session.id


async def test_threaded_turn_stamps_warm_context_key():
    # Phase B: a threaded turn carries its thread's warm-context key so a
    # workflow-backed tool can reuse the thread's warm checkout.
    runner, _, _ = _runner(
        ScriptedGateway([ModelResponse(text="ok", model="m")])
    )
    task = _task()
    await runner.run(task, _target())
    assert task.thread_key == thread_scope_key(_target())


async def test_top_level_turn_has_no_warm_context_key():
    runner, _, _ = _runner(
        ScriptedGateway([ModelResponse(text="ok", model="m")])
    )
    task = _task()
    top_level = SurfaceTarget(
        surface="slack", workspace="acme", agent="dev-platform",
        channel="C1", thread=None, event_id="top1",
    )
    await runner.run(task, top_level)
    assert task.thread_key is None


async def test_duplicate_event_is_deduped():
    gateway = ScriptedGateway([ModelResponse(text="once", model="m")])
    runner, sessions, delivery = _runner(gateway)

    first = await runner.run(_task(), _target("dupe"))
    second = await runner.run(_task(), _target("dupe"))

    assert first.id == second.id
    # No second turn, no second final answer.
    assert gateway._responses == []  # only one response consumed
    assert len(delivery.finals) == 1
    assert len(sessions._by_id) == 1


# --- runner: waiting for approval ---------------------------------------

async def test_pending_approval_parks_session_waiting():
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    runner, sessions, delivery = _runner(
        ScriptedGateway([
            tool_call_response(
                "m", [("c1", "github_issues_write", {"repo": "acme/x", "title": "T"})]
            ),
        ]),
        tools=tools,
    )

    session = await runner.run(_task("open an issue"), _target())

    assert session.status == "waiting"
    # The approval ids are persisted so Slice 4 can map a button back here.
    assert len(session.approval_ids) == 1
    assert (await sessions.get(session.id)).approval_ids == session.approval_ids
    # No final answer yet — the approval continuation (Slice 4) delivers it.
    assert delivery.finals == []
    # The thinking indicator was set, then an approval card was posted carrying
    # the request.
    assert delivery.statuses[0]["text"] == "is thinking..."
    assert len(delivery.approvals) == 1
    assert [r.id for r in delivery.approvals[0]["requests"]] == session.approval_ids
    assert session.progress_message_id == delivery.approvals[0]["id"]
    assert github.created == []  # write not executed


def _waiting_runner(*, delivery=None):
    """A runner whose session is parked on a github write approval."""
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    runner, sessions, delivery = _runner(
        ScriptedGateway([
            tool_call_response(
                "m", [("c1", "github_issues_write", {"repo": "acme/x", "title": "T"})]
            ),
            # M0b: after the write is approved, the model is re-run with the result
            # folded in and produces the fresh, user-facing answer.
            ModelResponse(text="Opened issue #1 ✅", model="m"),
        ]),
        tools=tools,
        delivery=delivery,
    )
    return runner, sessions, delivery, github


# --- runner: approval continuation (Slice 4) ----------------------------

async def test_approve_continues_session_and_posts_outcome_in_thread():
    runner, sessions, delivery, github = _waiting_runner()
    session = await runner.run(_task("open an issue"), _target())
    approval_id = session.approval_ids[0]

    message = await runner.resolve_approval(approval_id, "@maciag.artur", approve=True)

    assert message.startswith("✅ Approved by @maciag.artur")
    assert github.created  # the write executed on approval
    # M0b: the delivered answer is the model's FRESH reply (with the result folded
    # in), not the raw tool summary, posted in the original thread.
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["text"] == "Opened issue #1 ✅"
    assert delivery.finals[0]["target"].thread == "100.1"
    # The session is now completed and the approval card was collapsed (no buttons).
    done = await sessions.get(session.id)
    assert done.status == "completed"
    assert done.final_message_id is not None
    assert delivery.approvals[-1]["requests"] == []


async def test_approval_reruns_model_with_tool_result_folded_in():
    # The essence of M0b: on approval the model is re-run, and that continuation
    # call sees the ACTUAL tool result folded into the held round — not the
    # "held for human approval" placeholder — so the reply is a fresh model answer.
    runner, sessions, delivery, github = _waiting_runner()
    session = await runner.run(_task("open an issue"), _target())

    await runner.resolve_approval(session.approval_ids[0], "@maciag.artur", approve=True)

    calls = runner.runtime.gateway.calls
    assert len(calls) == 2  # initial turn + continuation
    tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs  # the held round is present in the continuation's context
    assert all("held for human approval" not in m["content"] for m in tool_msgs)
    # The continuation instance is a new id under the SAME session.
    done = await sessions.get(session.id)
    assert done.workflow_instance_id == f"{session.id}:cont:{session.approval_ids[0]}"
    assert done.status == "completed"
    assert delivery.finals[0]["text"] == "Opened issue #1 ✅"


async def test_deny_continues_session_without_executing():
    runner, sessions, delivery, github = _waiting_runner()
    session = await runner.run(_task("open an issue"), _target())

    message = await runner.resolve_approval(
        session.approval_ids[0], "@maciag.artur", approve=False
    )

    assert message.startswith("🚫 Denied")
    assert github.created == []
    done = await sessions.get(session.id)
    assert done.status == "completed"
    assert "Denied" in delivery.finals[-1]["text"]


async def test_non_approver_leaves_session_waiting():
    runner, sessions, delivery, github = _waiting_runner()
    session = await runner.run(_task("open an issue"), _target())

    message = await runner.resolve_approval(
        session.approval_ids[0], "@random", approve=True
    )

    assert message.startswith("⛔")
    assert github.created == []
    # The session stays parked; no final answer posted.
    assert (await sessions.get(session.id)).status == "waiting"
    assert delivery.finals == []


async def test_workflow_approval_waits_for_background_terminal_result():
    class PausedOrchestrator(FakeWorkerOrchestrator):
        def __init__(self):
            super().__init__(title="Add retries")
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def run_attempt(self, state, on_step=None):
            self.started.set()
            await self.release.wait()
            return await super().run_attempt(state, on_step=on_step)

    agent = load_agent(AGENT_YAML)
    engine = WorkflowEngine(InMemoryWorkflowStore())
    runner_impl = PausedOrchestrator()
    github = FakeGitHub()
    engine.register(build_coding_worker_workflow(runner_impl, github))
    tools = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(runner_impl, github)],
        engine=engine,
    )
    sessions = InMemorySurfaceSessionStore()
    delivery = FakeSurfaceDelivery()
    runtime = Runtime(
        agent,
        gateway=ScriptedGateway([
            tool_call_response(
                "m",
                [("c1", "coding_worker_pr_write",
                  {"repo": "acme/x", "instruction": "add retries"})],
            ),
            # M0b continuation: after the PR workflow finishes, the model is re-run.
            ModelResponse(text="Opened draft PR #1 🚀", model="m"),
        ]),
        tools=tools,
        usage=InMemoryUsageStore(),
        memory=InMemoryStore(),
        engine=engine,
    )
    runner = SessionRunner(runtime, sessions, delivery)
    session = await runner.run(_task("open a PR"), _target())
    approval_id = session.approval_ids[0]
    request = await tools.approvals.get(approval_id)
    job_id = request.args["job_id"]

    message = await runner.resolve_approval(
        approval_id, "@maciag.artur", approve=True
    )

    assert message.startswith("✅ Approved by @maciag.artur")
    await asyncio.wait_for(runner_impl.started.wait(), timeout=1)
    still_waiting = await sessions.get(session.id)
    assert still_waiting.status == "waiting"
    assert delivery.finals == []

    runner_impl.release.set()
    done = await engine.wait_background(job_id)

    assert done.status == "completed"
    completed = await sessions.get(session.id)
    assert completed.status == "completed"
    assert completed.final_message_id is not None
    # M0b: the final answer is the model's fresh reply, not the raw workflow summary.
    assert delivery.finals[-1]["text"] == "Opened draft PR #1 🚀"


async def test_workflow_progress_is_surfaced_as_transient_status():
    agent = load_agent(AGENT_YAML)
    engine = WorkflowEngine(InMemoryWorkflowStore())
    worker = FakeWorkerOrchestrator(title="Add retries")
    github = FakeGitHub()
    engine.register(build_coding_worker_workflow(worker, github))
    tools = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(worker, github)],
        engine=engine,
    )
    sessions = InMemorySurfaceSessionStore()
    delivery = FakeSurfaceDelivery()
    runtime = Runtime(
        agent,
        gateway=ScriptedGateway([
            tool_call_response(
                "m",
                [("c1", "coding_worker_pr_write",
                  {"repo": "acme/x", "instruction": "add retries"})],
            ),
            ModelResponse(text="Opened draft PR #1 🚀", model="m"),  # M0b continuation
        ]),
        tools=tools,
        usage=InMemoryUsageStore(),
        memory=InMemoryStore(),
        engine=engine,
    )
    runner = SessionRunner(runtime, sessions, delivery)
    session = await runner.run(_task("open a PR"), _target())
    approval_id = session.approval_ids[0]
    job_id = (await tools.approvals.get(approval_id)).args["job_id"]

    await runner.resolve_approval(approval_id, "@maciag.artur", approve=True)
    await engine.wait_background(job_id)
    # Statuses recorded before the terminal drain are final by the time the
    # background drive (awaited above) returns; let any just-scheduled ones run.
    await asyncio.sleep(0)

    phrases = [s["text"] for s in delivery.statuses]
    # A "still working" worker phrase is surfaced beyond the initial status, and
    # each phrase delivered is a real worker milestone (detached ticks coalesce
    # to the latest state, which is the right transient-status semantic).
    worker_phrases = [p for p in phrases if p != PROGRESS_STATUS_TEXT]
    assert worker_phrases  # progress was surfaced
    assert all(p.startswith("is ") and p.endswith("…") for p in worker_phrases)
    # Deduped: an unchanged phrase never re-hits the API back-to-back.
    assert all(a != b for a, b in zip(phrases, phrases[1:]))
    # It still delivers the final answer (the M0b model reply) after the ticks.
    assert delivery.finals[-1]["text"] == "Opened draft PR #1 🚀"


# --- runner: artifact-ref delivery (sealed analysis, Phase 2) -------------

def _analysis_runner(*, orch=None, threads=None, responses=None):
    """A runner whose agent can start the sealed analysis workflow."""
    agent = load_agent(AGENT_YAML)
    agent.spec.tools.append(
        Tool(name="analysis", type="native", permissions=["report:write"])
    )
    agent.spec.approvals.require_for.append("analysis.report:write")
    orch = orch or FakeAnalysisOrchestrator()
    engine = WorkflowEngine(InMemoryWorkflowStore())
    engine.register(build_analysis_worker_workflow(orch))
    tools = ToolGateway(tools=[AnalysisWorkerConnector(orch)], engine=engine)
    sessions = InMemorySurfaceSessionStore()
    delivery = FakeSurfaceDelivery()
    runtime = Runtime(
        agent,
        gateway=ScriptedGateway(
            responses
            if responses is not None
            else [
                # ONE scripted response only: the report must be delivered from
                # the ref, so no M0b continuation call may ever be made.
                tool_call_response(
                    "m",
                    [("c1", "analysis_report_write",
                      {"instruction": "summarize the sales data",
                       "inputs": [
                           {"source": "staged", "input_ref": "staged:one"}
                       ]})],
                ),
            ]
        ),
        tools=tools,
        usage=InMemoryUsageStore(),
        memory=InMemoryStore(),
        engine=engine,
    )
    runner = SessionRunner(
        runtime, sessions, delivery, threads=threads, artifacts=orch.artifacts
    )
    return runner, sessions, delivery, engine, tools, orch


async def test_analysis_report_is_delivered_as_artifact_from_ref_without_model_rerun():
    runner, sessions, delivery, engine, tools, orch = _analysis_runner()
    session = await runner.run(_task("analyze the sales data"), _target())
    assert session.status == "waiting"
    approval_id = session.approval_ids[0]
    job_id = (await tools.approvals.get(approval_id)).args["job_id"]

    message = await runner.resolve_approval(approval_id, "@maciag.artur", approve=True)

    assert message.startswith("✅ Approved by @maciag.artur")
    done_wf = await engine.wait_background(job_id)
    assert done_wf.status == "completed"

    done = await sessions.get(session.id)
    assert done.status == "completed"
    assert done.final_message_id is not None
    # The session persists the ref + replay-safe prose — never the body.
    assert done.result_artifact_ref == f"analysis://{job_id}/report.md"
    assert done.result_summary == orch.prose
    assert "FULL-REPORT-TAIL" not in done.result_summary

    # The final answer is the report itself, dereferenced into an Artifact.
    final = delivery.finals[-1]["text"]
    assert isinstance(final, Artifact)
    assert final.content == orch.body.decode()
    assert final.summary == orch.prose
    assert final.filename == "report.md"
    assert final.snippet_type == "markdown"
    # Bypass-M0b: exactly one model call (the original turn) was ever made.
    assert len(runner.runtime.gateway.calls) == 1
    # The approval card was collapsed (no buttons left).
    assert delivery.approvals[-1]["requests"] == []


async def test_analysis_transcript_replays_prose_never_the_report_body():
    threads = InMemoryThreadRecordStore()
    runner, sessions, delivery, engine, tools, orch = _analysis_runner(threads=threads)
    session = await runner.run(_task("analyze the sales data"), _target())
    approval_id = session.approval_ids[0]
    job_id = (await tools.approvals.get(approval_id)).args["job_id"]

    await runner.resolve_approval(approval_id, "@maciag.artur", approve=True)
    await engine.wait_background(job_id)

    frags = await threads.replayable_transcript(
        session.target, exclude_turn_id="other", limit=10
    )
    assert len(frags) == 1
    assert frags[0].answer == orch.prose
    assert "FULL-REPORT-TAIL" not in frags[0].answer


async def test_analysis_failure_continues_via_model_like_any_tool_failure():
    # A failed sealed run has no artifact ref, so it takes the normal M0b
    # continuation path: the model sees the failure and phrases the reply.
    orch = FakeAnalysisOrchestrator(error="input upload:one was never staged")
    runner, sessions, delivery, engine, tools, _ = _analysis_runner(
        orch=orch,
        responses=[
            tool_call_response(
                "m",
                [("c1", "analysis_report_write",
                  {"instruction": "summarize the sales data",
                   "inputs": [
                       {"source": "staged", "input_ref": "staged:one"}
                   ]})],
            ),
            ModelResponse(text="The analysis could not run: no staged input.", model="m"),
        ],
    )
    session = await runner.run(_task("analyze the sales data"), _target())
    approval_id = session.approval_ids[0]
    job_id = (await tools.approvals.get(approval_id)).args["job_id"]

    await runner.resolve_approval(approval_id, "@maciag.artur", approve=True)
    failed_wf = await engine.wait_background(job_id)

    assert failed_wf.status == "failed"
    done = await sessions.get(session.id)
    assert done.status == "completed"
    assert done.result_artifact_ref is None
    assert delivery.finals[-1]["text"] == "The analysis could not run: no staged input."


async def test_retry_rematerializes_artifact_final_from_persisted_ref():
    # A session that flipped completed with its ref recorded but crashed before
    # the post: _ensure_delivered rebuilds the Artifact from the store.
    orch = FakeAnalysisOrchestrator()
    ref = await orch.artifacts.put("job-9", orch.body)
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    runner.artifacts = orch.artifacts
    session = SurfaceSession(
        id="s-art", target=_target("ev-art"), status="completed",
        request_text="analyze", result_summary=orch.prose,
        result_artifact_ref=ref,
    )
    await sessions.upsert(session)

    await runner._ensure_delivered(session)

    final = delivery.finals[-1]["text"]
    assert isinstance(final, Artifact)
    assert final.content == orch.body.decode()
    assert final.summary == orch.prose
    assert (await sessions.get("s-art")).final_message_id is not None


async def test_artifact_final_degrades_to_prose_when_body_is_unavailable():
    # The ref outlived its body (e.g. an in-memory artifact store restarted):
    # the answer still lands as the prose summary with an explicit notice.
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    runner.artifacts = InMemoryArtifactStore()  # empty — the ref resolves to nothing
    session = SurfaceSession(
        id="s-lost", target=_target("ev-lost"), status="completed",
        request_text="analyze", result_summary="# Findings",
        result_artifact_ref="analysis://gone/report.md",
    )
    await sessions.upsert(session)

    await runner._ensure_delivered(session)

    final = delivery.finals[-1]["text"]
    assert isinstance(final, Prose)
    assert "# Findings" in final.text
    assert "could not be retrieved" in final.text


async def test_artifact_final_degrades_to_prose_without_a_store():
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    assert runner.artifacts is None
    session = SurfaceSession(
        id="s-nostore", target=_target("ev-nostore"), status="completed",
        request_text="analyze", result_summary="# Findings",
        result_artifact_ref="analysis://job/report.md",
    )
    await sessions.upsert(session)

    await runner._ensure_delivered(session)

    final = delivery.finals[-1]["text"]
    assert isinstance(final, Prose)
    assert "could not be retrieved" in final.text


async def test_progress_status_is_reasserted_after_refresh_interval():
    # Slack's status is transient; an unchanged phrase must be re-sent periodically
    # so a long single-phase run doesn't go blank. Bursts within the window still
    # collapse to one call.
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    await sessions.upsert(SurfaceSession(
        id="s1", target=_target("ev1"), status="waiting", approval_ids=["a1"],
    ))
    inst = WorkflowInstance(
        id="i1", workflow="w", status="running",
        state={"progress": "is working on the changes…", "approval_id": "a1"},
    )

    await runner._on_workflow_progress(inst)
    await runner._on_workflow_progress(inst)  # immediate repeat → collapsed
    assert [s["text"] for s in delivery.statuses] == ["is working on the changes…"]

    # Backdate the last-sent stamp past the refresh window → re-asserted.
    phrase, _ = runner._progress_seen["s1"]
    runner._progress_seen["s1"] = (
        phrase, time.monotonic() - PROGRESS_REFRESH_SECONDS - 1
    )
    await runner._on_workflow_progress(inst)
    assert [s["text"] for s in delivery.statuses] == [
        "is working on the changes…",
        "is working on the changes…",
    ]


async def test_progress_callback_bails_on_terminal_instance():
    # Defense-in-depth for the drain: a progress task that runs just after the
    # in-place-mutated instance goes terminal must not emit a stale status.
    runner, _, delivery = _runner(ScriptedGateway([]))
    terminal = WorkflowInstance(
        id="i1", workflow="w", status="completed",
        state={"progress": "is pushing the branch…", "approval_id": "a1"},
    )

    await runner._on_workflow_progress(terminal)

    assert delivery.statuses == []  # bailed before touching the surface


async def test_failed_outcome_delivery_is_repaired_on_second_click():
    # The write succeeds but the Slack post_final fails the first time. The session
    # is left terminal-without-final; a second click must re-deliver the answer
    # (and never re-execute the write), rather than the user getting nothing.
    class FlakyDelivery(FakeSurfaceDelivery):
        def __init__(self):
            super().__init__()
            self.fail_finals = 1

        async def post_final(self, target, result, *, key=None, recover=False):
            if self.fail_finals > 0:
                self.fail_finals -= 1
                raise RuntimeError("slack down")
            return await super().post_final(target, result, key=key, recover=recover)

    runner, sessions, delivery, github = _waiting_runner(delivery=FlakyDelivery())
    session = await runner.run(_task("open an issue"), _target())
    approval_id = session.approval_ids[0]

    # First click: write executes, but delivering the answer fails (swallowed).
    msg1 = await runner.resolve_approval(approval_id, "@maciag.artur", approve=True)
    assert msg1.startswith("✅ Approved by @maciag.artur")
    assert len(github.created) == 1
    stuck = await sessions.get(session.id)
    assert stuck.status == "completed" and stuck.final_message_id is None
    assert delivery.finals == []  # nothing delivered yet

    # Second click: no re-execution, and the persisted outcome is re-delivered.
    await runner.resolve_approval(approval_id, "@maciag.artur", approve=True)
    assert len(github.created) == 1  # write was not repeated
    repaired = await sessions.get(session.id)
    assert repaired.final_message_id is not None
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["text"] == repaired.result_summary


# --- runner: startup reconciler (Slice 6) --------------------------------

async def test_reconcile_redelivers_terminal_without_final():
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    await sessions.upsert(SurfaceSession(
        id="s1", target=_target("ev1"), status="completed",
        workflow_instance_id="s1", progress_message_id="p0",
        result_summary="the answer",
    ))

    repaired = await runner.reconcile()

    assert repaired == ["s1"]
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["text"] == "the answer"
    assert (await sessions.get("s1")).final_message_id is not None


async def test_reconcile_recovers_crashed_turn_from_completed_workflow():
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    # The workflow finished but the session crashed before delivering it.
    await runner.runtime.engine.store.upsert(WorkflowInstance(
        id="s2", workflow=runner.runtime.workflow_name, status="completed",
        state={
            "final_text": "recovered answer",
            "accounted": {"model": "m", "prompt_tokens": 0,
                          "completion_tokens": 0, "cost_usd": 0.0},
            "approval_ids": [],
        },
    ))
    await sessions.upsert(SurfaceSession(
        id="s2", target=_target("ev2"), status="running",
        workflow_instance_id="s2", progress_message_id="p0",
    ))

    await runner.reconcile()

    assert delivery.finals[-1]["text"] == "recovered answer"
    assert (await sessions.get("s2")).status == "completed"


async def test_reconcile_delivers_approved_workflow_waiting_session():
    agent = load_agent(AGENT_YAML)
    engine = WorkflowEngine(InMemoryWorkflowStore())
    worker = FakeWorkerOrchestrator()
    github = FakeGitHub()
    engine.register(build_coding_worker_workflow(worker, github))
    tools = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(worker, github)],
        engine=engine,
    )
    pending = await tools.invoke(
        agent,
        "coding_worker.pr:write",
        {"repo": "acme/x", "instruction": "x"},
    )
    request = pending.approval
    request.status = "approved"
    request.decided_by = "@maciag.artur"
    await tools.approvals.update(request)
    job_id = request.args["job_id"]
    inst = await engine.store.get(job_id)
    inst.status = "completed"
    inst.waiting_on = None
    inst.leased_until = None
    inst.result = {"summary": "opened draft PR #7 in acme/x"}
    await engine.store.upsert(inst)

    sessions = InMemorySurfaceSessionStore()
    delivery = FakeSurfaceDelivery()
    runtime = Runtime(
        agent,
        gateway=ScriptedGateway([]),
        tools=tools,
        usage=InMemoryUsageStore(),
        memory=InMemoryStore(),
        engine=engine,
    )
    runner = SessionRunner(runtime, sessions, delivery)
    await sessions.upsert(SurfaceSession(
        id="s-approved",
        target=_target("ev-approved"),
        status="waiting",
        approval_ids=[request.id],
        progress_message_id="p0",
    ))

    repaired = await runner.reconcile()

    assert repaired == ["s-approved"]
    assert delivery.finals[-1]["text"] == "opened draft PR #7 in acme/x"
    assert (await sessions.get("s-approved")).status == "completed"


async def test_reconcile_posts_interrupted_notice_for_abandoned_turn():
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    await runner.runtime.engine.store.upsert(WorkflowInstance(
        id="s3", workflow=runner.runtime.workflow_name, status="abandoned",
        state={"task": {}},
    ))
    await sessions.upsert(SurfaceSession(
        id="s3", target=_target("ev3"), status="running",
        workflow_instance_id="s3", progress_message_id="p0",
    ))

    await runner.reconcile()

    assert len(delivery.errors) == 1
    assert (await sessions.get("s3")).status == "abandoned"


async def test_reconcile_leaves_non_terminal_workflow_for_later():
    # The engine's own resume didn't (or couldn't) drive this to terminal — the
    # reconciler must not deliver a half-finished turn or abandon it; leave it.
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    await runner.runtime.engine.store.upsert(WorkflowInstance(
        id="s5", workflow=runner.runtime.workflow_name, status="running",
        completed_steps=["prepare"], state={"task": {}},
    ))
    await sessions.upsert(SurfaceSession(
        id="s5", target=_target("ev5"), status="running",
        workflow_instance_id="s5", progress_message_id="p0",
    ))

    repaired = await runner.reconcile()

    assert repaired == []  # left untouched
    assert delivery.finals == [] and delivery.errors == []
    assert (await sessions.get("s5")).status == "running"


async def test_reconcile_with_no_recoverable_workflow_posts_interrupted():
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    # A session whose workflow instance was lost (e.g. in-memory engine restart).
    await sessions.upsert(SurfaceSession(
        id="s4", target=_target("ev4"), status="running",
        workflow_instance_id="missing",
    ))

    await runner.reconcile()

    assert len(delivery.errors) == 1
    assert (await sessions.get("s4")).status == "abandoned"


async def test_reconcile_leaves_waiting_and_delivered_sessions_alone():
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    await sessions.upsert(SurfaceSession(
        id="w", target=_target("evw"), status="waiting",
        workflow_instance_id="w", approval_ids=["a1"],
    ))
    await sessions.upsert(SurfaceSession(
        id="d", target=_target("evd"), status="completed",
        workflow_instance_id="d", final_message_id="final-0",
    ))

    repaired = await runner.reconcile()

    assert repaired == []
    assert delivery.finals == [] and delivery.errors == []


async def test_reconcile_repairs_waiting_session_without_approval_card_id():
    runner, sessions, delivery, _github = _waiting_runner()
    session = await runner.run(_task("open an issue"), _target("ev-waiting"))
    original_card = session.progress_message_id

    # Simulate a crash after Slack accepted the approval card but before its ts
    # was persisted. The delivery key lets reconcile recover the same card id.
    session.progress_message_id = None
    await sessions.upsert(session)

    repaired = await runner.reconcile()

    assert repaired == [session.id]
    assert len(delivery.approvals) == 1
    assert (await sessions.get(session.id)).progress_message_id == original_card


# --- runner: crash-before-delivery repaired on retry ---------------------

async def test_retry_redelivers_terminal_session_without_final():
    # A session that reached `completed` but crashed before posting its final
    # answer (final_message_id is None). A retry of the same event re-delivers it
    # exactly once instead of returning a stuck, answerless session.
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    await sessions.upsert(SurfaceSession(
        id="s-crash", target=_target("ev-crash"), status="completed",
        workflow_instance_id="s-crash", progress_message_id="progress-0",
        result_summary="the answer",
    ))

    session = await runner.run(_task(), _target("ev-crash"))

    assert session.id == "s-crash"
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["text"] == "the answer"
    assert session.final_message_id == delivery.finals[0]["id"]

    # A further retry must not post a second final answer.
    again = await runner.run(_task(), _target("ev-crash"))
    assert again.final_message_id == session.final_message_id
    assert len(delivery.finals) == 1


async def test_final_delivery_is_idempotent_across_lost_id_window():
    # The narrow window the persisted id can't cover: the provider ACCEPTED the
    # final post but the process crashed before recording its id, so
    # final_message_id is still None. The deterministic delivery key lets the
    # retry recover the original message instead of posting a duplicate answer.
    runner, sessions, delivery = _runner(
        ScriptedGateway([ModelResponse(text="answer", model="m")])
    )
    session = await runner.run(_task(), _target())
    assert len(delivery.finals) == 1
    delivered_id = session.final_message_id
    assert delivered_id is not None

    # Simulate the lost-id crash window: the post landed, the id was never saved.
    session.final_message_id = None
    await sessions.upsert(session)

    # Recovery path (recover=True) dedups by key: no second message, id restored.
    await runner._ensure_delivered(session)
    assert len(delivery.finals) == 1
    assert session.final_message_id == delivered_id


# --- runner: interrupted / error ----------------------------------------

async def test_interrupted_turn_marks_abandoned_and_posts_error():
    # A model exception is caught by the workflow engine (step -> failed), so
    # handle() returns the interrupted `model="error"` response rather than
    # raising; the runner reflects that as an abandoned session + error notice.
    class BoomGateway:
        async def complete(self, model, messages, **kwargs):
            raise RuntimeError("model exploded")

    runner, sessions, delivery = _runner(BoomGateway())

    session = await runner.run(_task(), _target())

    assert session.status == "abandoned"
    assert session.error
    assert len(delivery.errors) == 1
    assert delivery.finals == []


# --- runner: conversation-history threading ------------------------------

def _completed(id_, *, request, answer, at, thread="100.1", delivered=True):
    """A completed session in `thread`, with an explicit created_at for ordering.

    `delivered` controls whether the answer actually reached the user (a recorded
    final_message_id) — an undelivered turn must not be replayed as history.
    """
    return SurfaceSession(
        id=id_,
        target=SurfaceTarget(
            surface="slack",
            workspace="acme",
            agent="dev-platform",
            channel="C1",
            thread=thread,
            event_id=f"ev-{id_}",
        ),
        status="completed",
        request_text=request,
        result_summary=answer,
        final_message_id=f"final-{id_}" if delivered else None,
        created_at=at,
    )


async def test_followup_turn_threads_prior_exchange_into_history():
    gateway = ScriptedGateway(
        [
            ModelResponse(text="first answer", model="m"),
            ModelResponse(text="second answer", model="m"),
        ]
    )
    runner, sessions, _ = _runner(gateway)

    first = await runner.run(_task("first question"), _target("ev1"))
    await runner.run(_task("second question"), _target("ev2"))

    # The inbound text is persisted so it can be replayed later.
    assert first.request_text == "first question"

    # The first turn had no prior history; the second saw the first exchange.
    first_roles = [m["role"] for m in gateway.calls[0]["messages"]]
    assert "assistant" not in first_roles

    second = gateway.calls[1]["messages"]
    pairs = [(m["role"], m.get("content")) for m in second]
    assert ("user", "first question") in pairs
    assert ("assistant", "first answer") in pairs
    # History is replayed before the current question, in order.
    assert pairs.index(("user", "first question")) < pairs.index(("assistant", "first answer"))
    assert pairs.index(("assistant", "first answer")) < pairs.index(("user", "second question"))


async def test_thread_store_records_fragment_and_feeds_followup():
    # Phase A slice 2: with a ThreadRecordStore wired, a delivered turn commits a
    # transcript fragment, and the next turn in the thread replays it as history.
    threads = InMemoryThreadRecordStore()
    gateway = ScriptedGateway([
        ModelResponse(text="first answer", model="m"),
        ModelResponse(text="second answer", model="m"),
    ])
    runner, _, _ = _runner(gateway, threads=threads)

    await runner.run(_task("first question"), _target("ev1"))
    await runner.run(_task("second question"), _target("ev2"))

    # Both delivered turns were committed to the thread's transcript, oldest-first.
    frags = await threads.replayable_transcript(_target("ev3"))
    assert [(f.request, f.answer) for f in frags] == [
        ("first question", "first answer"),
        ("second question", "second answer"),
    ]

    # The second turn saw that exchange (replayed before its own question, in order).
    pairs = [(m["role"], m.get("content")) for m in gateway.calls[1]["messages"]]
    assert ("user", "first question") in pairs
    assert ("assistant", "first answer") in pairs
    assert pairs.index(("assistant", "first answer")) < pairs.index(("user", "second question"))


async def test_run_threaded_serializes_concurrent_replies():
    # Phase C slice 2: two replies to the same thread arrive together; they must
    # run one at a time, in order, and the second must see the first's answer.
    threads = InMemoryThreadRecordStore()
    gateway = ScriptedGateway([
        ModelResponse(text="a1", model="m"),
        ModelResponse(text="a2", model="m"),
    ])
    runner, _, delivery = _runner(gateway, threads=threads)

    await asyncio.gather(
        runner.run_threaded(_task("q1"), _target("ev1")),
        runner.run_threaded(_task("q2"), _target("ev2")),
    )

    # Both answered, in order, exactly once — serialized, none dropped.
    assert [f["text"] for f in delivery.finals] == ["a1", "a2"]
    frags = await threads.replayable_transcript(_target("ev3"))
    assert [(f.request, f.answer) for f in frags] == [("q1", "a1"), ("q2", "a2")]
    # The second turn was driven AFTER the first delivered — it saw a1 as context.
    pairs = [(m["role"], m.get("content")) for m in gateway.calls[1]["messages"]]
    assert ("user", "q1") in pairs and ("assistant", "a1") in pairs


async def test_run_threaded_dedupes_duplicate_event():
    threads = InMemoryThreadRecordStore()
    gateway = ScriptedGateway([ModelResponse(text="only", model="m")])
    runner, _, delivery = _runner(gateway, threads=threads)

    await runner.run_threaded(_task("q1"), _target("dup"))
    await runner.run_threaded(_task("q1"), _target("dup"))  # same event_id

    assert len(gateway.calls) == 1  # the model ran once (run() dedups on event_id)
    assert [f["text"] for f in delivery.finals] == ["only"]


async def test_run_threaded_falls_back_without_thread_store():
    gateway = ScriptedGateway([ModelResponse(text="hi", model="m")])
    runner, _, delivery = _runner(gateway)  # threads=None
    await runner.run_threaded(_task("q"), _target("ev1"))
    assert [f["text"] for f in delivery.finals] == ["hi"]


async def test_apply_thread_history_reads_thread_store_over_sessions():
    # When a thread store is present, history comes from IT — not the per-session
    # scan. Prove it by seeding ONLY the thread store (sessions left empty).
    threads = InMemoryThreadRecordStore()
    runner, sessions, _ = _runner(ScriptedGateway([]), threads=threads)
    target = _target("ev-seed")
    await threads.append_delivered_fragment(
        target, TranscriptFragment(turn_id="t0", request="seeded q", answer="seeded a")
    )

    task = _task("now")
    await runner._apply_thread_history(task, SurfaceSession(id="cur", target=_target("ev-cur")))

    assert {"role": "user", "content": "seeded q"} in task.history
    assert {"role": "assistant", "content": "seeded a"} in task.history


async def test_apply_thread_history_falls_back_to_sessions_without_thread_store():
    # No thread store → the old per-session thread_history path still works.
    runner, sessions, _ = _runner(ScriptedGateway([]))  # threads=None
    await sessions.upsert(_completed(
        "a", request="q1", answer="a1", at=datetime(2026, 6, 28, tzinfo=timezone.utc)
    ))

    task = _task("now")
    await runner._apply_thread_history(task, SurfaceSession(id="cur", target=_target("ev-cur")))

    assert {"role": "user", "content": "q1"} in task.history
    assert {"role": "assistant", "content": "a1"} in task.history


async def test_history_skips_non_completed_and_orders_oldest_first():
    runner, sessions, _ = _runner(ScriptedGateway([]))
    base = datetime(2026, 6, 28, tzinfo=timezone.utc)
    await sessions.upsert(_completed("a", request="q1", answer="a1", at=base))
    # A failed turn has no trustworthy answer — skip it entirely.
    await sessions.upsert(
        SurfaceSession(
            id="b",
            target=_target("ev-b"),
            status="failed",
            request_text="q2",
            error="boom",
            created_at=base + timedelta(minutes=1),
        )
    )
    await sessions.upsert(
        _completed("c", request="q3", answer="a3", at=base + timedelta(minutes=2))
    )

    task = _task("now")
    await runner._apply_thread_history(task, SurfaceSession(id="cur", target=_target("ev-cur")))

    assert task.history == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


async def test_history_is_scoped_to_the_thread():
    runner, sessions, _ = _runner(ScriptedGateway([]))
    base = datetime(2026, 6, 28, tzinfo=timezone.utc)
    # A completed turn in a *different* thread must not leak in.
    await sessions.upsert(
        _completed("other", request="elsewhere", answer="nope", at=base, thread="999.9")
    )
    await sessions.upsert(_completed("mine", request="q", answer="a", at=base))

    task = _task("now")
    await runner._apply_thread_history(task, SurfaceSession(id="cur", target=_target("ev-cur")))

    assert task.history == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


async def test_history_left_untouched_without_thread_or_when_preset():
    runner, sessions, _ = _runner(ScriptedGateway([]))
    base = datetime(2026, 6, 28, tzinfo=timezone.utc)
    await sessions.upsert(_completed("a", request="q", answer="a", at=base))

    # No thread on the target → nothing to replay.
    no_thread = SurfaceTarget(
        surface="slack", workspace="acme", agent="dev-platform", channel="C1",
        thread=None, event_id="z",
    )
    task = _task("x")
    await runner._apply_thread_history(task, SurfaceSession(id="c1", target=no_thread))
    assert task.history == []

    # A caller that already supplied history is never clobbered.
    preset = _task("x")
    preset.history = [{"role": "user", "content": "preset"}]
    await runner._apply_thread_history(preset, SurfaceSession(id="c2", target=_target("ev-c2")))
    assert preset.history == [{"role": "user", "content": "preset"}]


async def test_history_excludes_completed_but_undelivered_turn():
    # A turn whose answer was persisted (status=completed, result_summary set) but
    # never reached the user (final_message_id is None — the transient-failure /
    # crash-before-delivery window) must NOT be replayed: the user never saw it.
    store = InMemorySurfaceSessionStore()
    base = datetime(2026, 6, 28, tzinfo=timezone.utc)
    await store.upsert(_completed("d", request="q1", answer="a1", at=base))
    await store.upsert(
        _completed(
            "nd", request="q2", answer="a2",
            at=base + timedelta(minutes=1), delivered=False,
        )
    )

    prior = await store.thread_history(_target("ev-cur"))
    assert [s.id for s in prior] == ["d"]


async def test_history_limit_counts_only_delivered_turns():
    # The limit is applied AFTER filtering to replayable turns, so a burst of
    # recent unusable turns can't crowd a valid older exchange out of the window.
    store = InMemorySurfaceSessionStore()
    base = datetime(2026, 6, 28, tzinfo=timezone.utc)
    # Oldest: one genuinely delivered exchange.
    await store.upsert(_completed("old", request="q", answer="a", at=base))
    # Newer noise that would fill a naive most-recent-N window: a failed turn, an
    # undelivered completed turn, and one still waiting on approval.
    await store.upsert(
        SurfaceSession(
            id="f", target=_target("ev-f"), status="failed",
            request_text="qf", error="boom", created_at=base + timedelta(minutes=1),
        )
    )
    await store.upsert(
        _completed(
            "u", request="qu", answer="au",
            at=base + timedelta(minutes=2), delivered=False,
        )
    )
    await store.upsert(
        SurfaceSession(
            id="w", target=_target("ev-w"), status="waiting",
            request_text="qw", created_at=base + timedelta(minutes=3),
        )
    )

    # Even with a tight limit (smaller than the noise count), the delivered
    # exchange survives — a pre-filter limit would have returned nothing.
    prior = await store.thread_history(_target("ev-cur"), limit=2)
    assert [s.id for s in prior] == ["old"]
