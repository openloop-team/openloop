"""Unit tests for the durable-workflow engine."""

import asyncio
from datetime import timedelta

import pytest

from openloop.workflows import (
    InMemoryWorkflowStore,
    Step,
    Workflow,
    WorkflowEngine,
)
from openloop.workflows.store import _now


def _logging_workflow():
    async def a(ctx):
        ctx.state.setdefault("log", []).append("a")

    async def b(ctx):
        ctx.state.setdefault("log", []).append("b")
        ctx.instance.result = {"done": True}

    return Workflow("t", [Step("a", a), Step("gate", wait=True), Step("b", b)])


def _engine(workflow=None, store=None):
    store = store or InMemoryWorkflowStore()
    wf = workflow or _logging_workflow()
    return WorkflowEngine(store, {wf.name: wf}), store


def _engine_with_lease(workflow, lease_seconds):
    store = InMemoryWorkflowStore()
    return WorkflowEngine(store, {workflow.name: workflow}, lease_seconds=lease_seconds), store


async def test_runs_until_wait_node_then_parks():
    engine, store = _engine()
    inst = await engine.start("t", "i1", {})
    assert inst.status == "waiting"
    assert inst.waiting_on == "gate"
    assert inst.completed_steps == ["a"]
    assert inst.state["log"] == ["a"]  # b has not run


async def test_event_wakes_and_drives_to_completion():
    engine, store = _engine()
    await engine.start("t", "i1", {})
    inst = await engine.send_event("i1", "gate", {"by": "maciag.artur"})
    assert inst.status == "completed"
    assert inst.completed_steps == ["a", "gate", "b"]
    assert inst.state["log"] == ["a", "b"]
    assert inst.result == {"done": True}
    assert inst.state["events"]["gate"] == {"by": "maciag.artur"}


async def test_event_can_wake_without_inline_drive():
    engine, store = _engine()
    await engine.start("t", "i1", {})

    inst = await engine.send_event("i1", "gate", {"by": "maciag.artur"}, drive=False)

    assert inst.status == "running"
    assert inst.leased_until is not None
    assert inst.state["log"] == ["a"]  # b has not run inline

    engine.drive_background("i1")
    done = await engine.wait_background("i1")
    assert done.status == "completed"
    assert done.state["log"] == ["a", "b"]


async def test_send_event_is_idempotent_after_completion():
    engine, store = _engine()
    await engine.start("t", "i1", {})
    await engine.send_event("i1", "gate")
    # A duplicate event must not re-run step b.
    inst = await engine.send_event("i1", "gate")
    assert inst.status == "completed"
    assert inst.state["log"] == ["a", "b"]


async def test_send_event_for_wrong_node_is_noop():
    engine, store = _engine()
    await engine.start("t", "i1", {})
    inst = await engine.send_event("i1", "not-the-gate")
    assert inst.status == "waiting"  # unchanged


async def test_step_exception_marks_failed_terminal():
    async def boom(ctx):
        raise RuntimeError("kaboom")

    wf = Workflow("t", [Step("boom", boom)])
    engine, store = _engine(wf)
    inst = await engine.start("t", "i1", {})
    assert inst.status == "failed"
    assert inst.error == "kaboom"
    # Terminal: a re-drive does not resurrect it.
    assert await engine.resume_incomplete() == []


async def test_start_is_idempotent_resume_not_restart():
    engine, store = _engine()
    await engine.start("t", "i1", {})  # parks at gate, log == ["a"]
    inst = await engine.start("t", "i1", {})  # same id: resume, not restart
    assert inst.state["log"] == ["a"]  # a not run twice


async def test_resume_incomplete_redrives_running_only():
    engine, store = _engine()
    # Seed a crashed-mid-run instance: status running, nothing completed.
    from openloop.workflows import WorkflowInstance

    await store.upsert(WorkflowInstance(id="crashed", workflow="t", status="running"))
    await store.upsert(WorkflowInstance(id="parked", workflow="t", status="waiting",
                                        waiting_on="gate", completed_steps=["a"]))

    resumed = await engine.resume_incomplete()
    assert resumed == ["crashed"]
    # The crashed one was driven forward to its wait node.
    assert (await store.get("crashed")).status == "waiting"
    # The parked one was left alone.
    assert (await store.get("parked")).status == "waiting"


async def test_resume_incomplete_skips_fresh_lease():
    engine, store = _engine()
    from openloop.workflows import WorkflowInstance

    await store.upsert(WorkflowInstance(
        id="active",
        workflow="t",
        status="running",
        leased_until=_now() + timedelta(seconds=30),
    ))

    resumed = await engine.resume_incomplete()

    assert resumed == []
    active = await store.get("active")
    assert active.status == "running"
    assert active.completed_steps == []


async def test_resume_incomplete_redrives_expired_lease():
    engine, store = _engine()
    from openloop.workflows import WorkflowInstance

    await store.upsert(WorkflowInstance(
        id="stale",
        workflow="t",
        status="running",
        leased_until=_now() - timedelta(seconds=1),
    ))

    resumed = await engine.resume_incomplete()

    assert resumed == ["stale"]
    assert (await store.get("stale")).status == "waiting"


async def test_quiet_in_flight_step_renews_lease_and_is_not_redriven():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def quiet(ctx):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        ctx.instance.result = {"done": True}

    wf = Workflow("quiet", [Step("gate", wait=True), Step("quiet", quiet)])
    engine, store = _engine_with_lease(wf, lease_seconds=0.12)

    await engine.start("quiet", "i1", {})
    await engine.send_event("i1", "gate", drive=False)
    engine.drive_background("i1")
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.sleep(0.25)
    active = await store.get("i1")
    assert active.status == "running"
    assert active.leased_until > _now()

    resumed = await engine.resume_incomplete()
    assert resumed == []
    assert calls == 1

    release.set()
    done = await engine.wait_background("i1")
    assert done.status == "completed"
    assert calls == 1


async def test_resume_incomplete_skips_in_process_background_drive_even_if_lease_expired():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def quiet(ctx):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()

    wf = Workflow("active", [Step("gate", wait=True), Step("quiet", quiet)])
    engine, store = _engine_with_lease(wf, lease_seconds=60)

    await engine.start("active", "i1", {})
    await engine.send_event("i1", "gate", drive=False)
    engine.drive_background("i1")
    await asyncio.wait_for(started.wait(), timeout=1)

    inst = await store.get("i1")
    inst.leased_until = _now() - timedelta(seconds=1)
    await store.upsert(inst)

    resumed = await engine.resume_incomplete()
    assert resumed == []
    assert calls == 1

    release.set()
    await engine.wait_background("i1")


def _two_step_workflow(calls):
    async def gen(ctx):
        calls.append("gen")

    async def save(ctx):
        calls.append("save")

    # gen is non-resumable (e.g. a model call); save is idempotent.
    return Workflow("t2", [Step("gen", gen, resumable=False), Step("save", save)])


async def test_resume_abandons_when_non_resumable_step_pending():
    from openloop.workflows import WorkflowInstance

    calls: list[str] = []
    wf = _two_step_workflow(calls)
    engine, store = _engine(wf)
    await store.upsert(WorkflowInstance(id="i", workflow="t2", status="running"))

    resumed = await engine.resume_incomplete()
    assert resumed == []
    assert (await store.get("i")).status == "abandoned"
    assert calls == []  # the non-resumable step was never replayed


async def test_resume_runs_when_only_resumable_steps_remain():
    from openloop.workflows import WorkflowInstance

    calls: list[str] = []
    wf = _two_step_workflow(calls)
    engine, store = _engine(wf)
    await store.upsert(WorkflowInstance(
        id="i", workflow="t2", status="running", completed_steps=["gen"]
    ))

    resumed = await engine.resume_incomplete()
    assert resumed == ["i"]
    assert (await store.get("i")).status == "completed"
    assert calls == ["save"]  # only the idempotent tail re-ran


async def test_start_does_not_redrive_existing_instance():
    from openloop.workflows import WorkflowInstance

    calls: list[str] = []
    wf = _two_step_workflow(calls)
    engine, store = _engine(wf)
    await store.upsert(WorkflowInstance(
        id="i", workflow="t2", status="running", completed_steps=["gen"]
    ))

    inst = await engine.start("t2", "i", {})
    assert inst.status == "running"  # returned as-is
    assert calls == []  # never driven into the non-resumable step


async def test_cancel_marks_terminal():
    engine, store = _engine()
    await engine.start("t", "i1", {})
    inst = await engine.cancel("i1", "approval denied")
    assert inst.status == "cancelled"
    assert inst.error == "approval denied"
    # A late event no longer wakes it.
    woken = await engine.send_event("i1", "gate")
    assert woken.status == "cancelled"
