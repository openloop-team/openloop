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
    stored = await store.get("i1")
    assert inst.status == "waiting"
    assert inst.waiting_on == "gate"
    assert inst.completed_steps == ["a"]
    assert inst.state["log"] == ["a"]  # b has not run
    assert inst.drive_gen == stored.drive_gen
    assert inst.leased_until is stored.leased_until is None


async def test_event_wakes_and_drives_to_completion():
    engine, store = _engine()
    await engine.start("t", "i1", {})
    inst = await engine.send_event("i1", "gate", {"by": "maciag.artur"})
    stored = await store.get("i1")
    assert inst.status == "completed"
    assert inst.completed_steps == ["a", "gate", "b"]
    assert inst.state["log"] == ["a", "b"]
    assert inst.result == {"done": True}
    assert inst.state["events"]["gate"] == {"by": "maciag.artur"}
    assert inst.drive_gen == stored.drive_gen
    assert inst.leased_until is stored.leased_until is None


async def test_event_can_wake_without_inline_drive():
    engine, store = _engine()
    await engine.start("t", "i1", {})

    inst = await engine.send_event("i1", "gate", {"by": "maciag.artur"}, drive=False)

    assert inst.status == "running"
    # Event consumption no longer takes the drive lease — whoever reaches
    # claim_drive first (background task or recovery sweep) wins it.
    assert inst.leased_until is None
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


async def test_two_replicas_claim_a_wait_event_once():
    store = InMemoryWorkflowStore()
    calls = 0

    async def side_effect(ctx):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)

    workflow = Workflow("t", [Step("gate", wait=True), Step("work", side_effect)])
    first = WorkflowEngine(store, {"t": workflow})
    second = WorkflowEngine(store, {"t": workflow})
    await first.start("t", "replicated", {})

    await asyncio.gather(
        first.send_event("replicated", "gate", {"event": "one"}),
        second.send_event("replicated", "gate", {"event": "two"}),
    )

    assert calls == 1
    assert (await store.get("replicated")).status == "completed"


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

    await store.create(WorkflowInstance(id="crashed", workflow="t", status="running"))
    await store.create(WorkflowInstance(id="parked", workflow="t", status="waiting",
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

    await store.create(WorkflowInstance(
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

    await store.create(WorkflowInstance(
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


async def test_resume_takes_over_expired_lease_and_evicts_stale_drive():
    # The stale-writer scenario: a drive's lease lapses (as if its process
    # stalled), a second claimant takes over, and the stale drive's writes
    # lose the fence — its progress can't clobber the new owner's.
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

    # Simulate the lease lapsing without the driver noticing (leased_until is
    # store-owned, so tests reach into the store to move the clock).
    store._by_id["i1"].leased_until = _now() - timedelta(seconds=1)

    resume_task = asyncio.create_task(engine.resume_incomplete())
    for _ in range(200):  # wait for the takeover to claim and re-run the step
        if calls == 2:
            break
        await asyncio.sleep(0.005)
    assert calls == 2

    release.set()  # lets both the stale step and the takeover step finish
    resumed = await resume_task
    await engine.wait_background("i1")  # the stale drive ends without writing

    assert resumed == ["i1"]
    final = await store.get("i1")
    assert final.status == "completed"
    assert final.completed_steps.count("quiet") == 1  # exactly one write won


async def test_terminal_drains_in_flight_progress_before_callbacks():
    # A progress task stalled mid-write (as if awaiting a slow surface call) is
    # cancelled on the terminal transition, *before* terminal callbacks fire —
    # so a stale "still working…" status can't land after the final answer.
    progress_started = asyncio.Event()
    progress_completed = False
    terminal_fired_at: list[bool] = []
    terminal_ownership: list[tuple[int, object]] = []

    async def stalling_progress(instance):
        nonlocal progress_completed
        progress_started.set()
        await asyncio.sleep(10)  # never completes; drain must cancel it
        progress_completed = True

    async def on_terminal(instance):
        # Records that terminal delivery ran, and that no progress task survives.
        terminal_fired_at.append("i1" not in instance_engine._progress_tasks)
        terminal_ownership.append((instance.drive_gen, instance.leased_until))

    async def work(ctx):
        await ctx.checkpoint()  # schedules the progress task
        await asyncio.wait_for(progress_started.wait(), timeout=1)  # it stalls

    wf = Workflow("w", [Step("gate", wait=True), Step("work", work)])
    instance_engine, store = _engine(wf)
    instance_engine.add_progress_callback(stalling_progress)
    instance_engine.add_terminal_callback(on_terminal)

    await instance_engine.start("w", "i1", {})
    await instance_engine.send_event("i1", "gate")  # drives inline to terminal

    stored = await store.get("i1")
    assert stored.status == "completed"
    assert progress_completed is False  # the stalled progress write was cancelled
    assert terminal_fired_at == [True]  # drained before the terminal callback ran
    assert terminal_ownership == [(stored.drive_gen, None)]
    assert "i1" not in instance_engine._progress_tasks


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
    await store.create(WorkflowInstance(id="i", workflow="t2", status="running"))

    resumed = await engine.resume_incomplete()
    assert resumed == []
    assert (await store.get("i")).status == "abandoned"
    assert calls == []  # the non-resumable step was never replayed


async def test_resume_runs_when_only_resumable_steps_remain():
    from openloop.workflows import WorkflowInstance

    calls: list[str] = []
    wf = _two_step_workflow(calls)
    engine, store = _engine(wf)
    await store.create(WorkflowInstance(
        id="i", workflow="t2", status="running", completed_steps=["gen"]
    ))

    resumed = await engine.resume_incomplete()
    assert resumed == ["i"]
    assert (await store.get("i")).status == "completed"
    assert calls == ["save"]  # only the idempotent tail re-ran


class _CheckpointAfterRecentStore(InMemoryWorkflowStore):
    """Land a non-resumable checkpoint after returning the scan snapshot."""

    async def recent(self, limit=100):
        snapshots = await super().recent(limit)
        current = await self.get("i")
        current.completed_steps.append("gen")
        assert await self.fenced_write(current, current.drive_gen)
        return snapshots


async def test_resume_rechecks_non_resumable_steps_after_claim():
    from openloop.workflows import WorkflowInstance

    calls: list[str] = []
    wf = _two_step_workflow(calls)
    store = _CheckpointAfterRecentStore()
    engine = WorkflowEngine(store, {wf.name: wf})
    await store.create(
        WorkflowInstance(
            id="i",
            workflow="t2",
            status="running",
            drive_gen=1,
            leased_until=_now() - timedelta(seconds=1),
        )
    )

    resumed = await engine.resume_incomplete()

    final = await store.get("i")
    assert resumed == ["i"]
    assert final.status == "completed"
    assert final.completed_steps == ["gen", "save"]
    assert calls == ["save"]


async def test_start_does_not_redrive_existing_instance():
    from openloop.workflows import WorkflowInstance

    calls: list[str] = []
    wf = _two_step_workflow(calls)
    engine, store = _engine(wf)
    await store.create(WorkflowInstance(
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


# ---------------------------------------------------------------------------
# Store arbitration primitives: claim, fence, release, evict, isolation.
# ---------------------------------------------------------------------------


def _instance(**kwargs):
    from openloop.workflows import WorkflowInstance

    defaults = dict(id="w1", workflow="t", status="running")
    defaults.update(kwargs)
    return WorkflowInstance(**defaults)


async def test_claim_drive_wins_unleased_running_and_bumps_gen():
    store = InMemoryWorkflowStore()
    await store.create(_instance())
    claimed = await store.claim_drive("w1", lease_seconds=30)
    assert claimed is not None
    assert claimed.drive_gen == 1
    assert claimed.leased_until > _now()


async def test_claim_drive_loses_to_live_lease_and_wins_expired():
    store = InMemoryWorkflowStore()
    await store.create(_instance())
    first = await store.claim_drive("w1", lease_seconds=30)
    assert first.drive_gen == 1
    assert await store.claim_drive("w1", lease_seconds=30) is None
    store._by_id["w1"].leased_until = _now() - timedelta(seconds=1)
    second = await store.claim_drive("w1", lease_seconds=30)
    assert second is not None
    assert second.drive_gen == 2


async def test_claim_drive_refuses_waiting_terminal_and_missing():
    store = InMemoryWorkflowStore()
    await store.create(_instance(id="parked", status="waiting", waiting_on="gate"))
    await store.create(_instance(id="done", status="completed"))
    assert await store.claim_drive("parked", lease_seconds=30) is None
    assert await store.claim_drive("done", lease_seconds=30) is None
    assert await store.claim_drive("missing", lease_seconds=30) is None


async def test_fenced_write_with_stale_gen_mutates_nothing():
    store = InMemoryWorkflowStore()
    await store.create(_instance(state={"v": 1}))
    claimed = await store.claim_drive("w1", lease_seconds=30)
    before = await store.get("w1")
    stale = await store.get("w1")
    stale.state["v"] = 99
    assert await store.fenced_write(stale, claimed.drive_gen - 1) is False
    after = await store.get("w1")
    assert after.state == before.state
    assert after.drive_gen == before.drive_gen
    assert after.updated_at == before.updated_at


async def test_release_write_invalidates_writers_own_gen_and_clears_lease():
    store = InMemoryWorkflowStore()
    await store.create(_instance())
    claimed = await store.claim_drive("w1", lease_seconds=30)
    claimed.status = "waiting"
    claimed.waiting_on = "gate"
    assert await store.fenced_write(claimed, claimed.drive_gen, release=True)
    stored = await store.get("w1")
    assert stored.drive_gen == claimed.drive_gen + 1
    assert stored.leased_until is None
    # The releasing writer's own gen is now stale: a straggler write fails.
    claimed.state["straggler"] = True
    assert await store.fenced_write(claimed, claimed.drive_gen) is False
    assert "straggler" not in (await store.get("w1")).state


async def test_renew_lease_honors_fence_and_never_decreases_deadline():
    store = InMemoryWorkflowStore()
    await store.create(_instance(state={"v": 1}))
    claimed = await store.claim_drive("w1", lease_seconds=1)
    assert await store.renew_lease("w1", claimed.drive_gen + 1, lease_seconds=99) is False
    assert await store.renew_lease("w1", claimed.drive_gen, lease_seconds=60)
    renewed_until = store._by_id["w1"].leased_until
    # A full-field checkpoint must not roll the renewed lease back: the lease
    # is store-owned and the payload's (stale) value is ignored.
    claimed.state["v"] = 2
    assert await store.fenced_write(claimed, claimed.drive_gen)
    stored = await store.get("w1")
    assert stored.leased_until == renewed_until
    assert stored.state == {"v": 2}


async def test_create_returns_false_on_conflict_and_never_overwrites():
    store = InMemoryWorkflowStore()
    assert await store.create(_instance(state={"progress": "real"}))
    assert await store.create(_instance(state={"progress": "clobber"})) is False
    assert (await store.get("w1")).state == {"progress": "real"}


async def test_cancel_instance_wins_once_and_evicts():
    store = InMemoryWorkflowStore()
    await store.create(_instance())
    claimed = await store.claim_drive("w1", lease_seconds=30)
    cancelled = await store.cancel_instance("w1", "denied")
    assert cancelled.status == "cancelled"
    assert cancelled.error == "denied"
    assert cancelled.drive_gen == claimed.drive_gen + 1
    # The live driver's next write loses the fence.
    assert await store.fenced_write(claimed, claimed.drive_gen) is False
    # Terminal now: a second cancel loses (at-most-once callbacks upstream).
    assert await store.cancel_instance("w1", "again") is None


async def test_claim_event_leaves_lease_unset():
    store = InMemoryWorkflowStore()
    await store.create(_instance(status="waiting", waiting_on="gate"))
    woken = await store.claim_event("w1", "gate", {"by": "someone"})
    assert woken.status == "running"
    assert woken.leased_until is None
    assert store._by_id["w1"].leased_until is None


async def test_store_boundaries_are_snapshot_isolated():
    store = InMemoryWorkflowStore()
    seed = _instance(state={"log": []})
    await store.create(seed)
    seed.state["log"].append("aliased-through-create")
    got = await store.get("w1")
    got.state["log"].append("aliased-through-get")
    assert (await store.get("w1")).state == {"log": []}
    claimed = await store.claim_drive("w1", lease_seconds=30)
    claimed.state["log"].append("aliased-through-claim")
    assert (await store.get("w1")).state == {"log": []}


async def test_park_wake_handoff_bumps_gen_each_transition():
    store = InMemoryWorkflowStore()
    await store.create(_instance())
    claimed = await store.claim_drive("w1", lease_seconds=30)  # gen 1
    claimed.status = "waiting"
    claimed.waiting_on = "gate"
    assert await store.fenced_write(claimed, claimed.drive_gen, release=True)  # gen 2
    woken = await store.claim_event("w1", "gate", {})
    assert woken.drive_gen == 2
    reclaimed = await store.claim_drive("w1", lease_seconds=30)  # gen 3
    assert reclaimed.drive_gen == 3
    # Pre-park gen is two transitions stale.
    claimed.state["straggler"] = True
    assert await store.fenced_write(claimed, claimed.drive_gen) is False


# ---------------------------------------------------------------------------
# Engine ownership: contention, eviction, and the ownership-loss boundary.
# ---------------------------------------------------------------------------


async def test_drive_contention_exactly_one_engine_runs_steps():
    store = InMemoryWorkflowStore()
    release = asyncio.Event()
    calls = 0

    async def work(ctx):
        nonlocal calls
        calls += 1
        await release.wait()

    wf = Workflow("t", [Step("gate", wait=True), Step("work", work)])
    first = WorkflowEngine(store, {"t": wf})
    second = WorkflowEngine(store, {"t": wf})
    await first.start("t", "i1", {})
    await first.send_event("i1", "gate", drive=False)

    first.drive_background("i1")
    await asyncio.sleep(0)  # let the first drive claim
    loser = await second.wait_background("i1")  # no local task: reads the store
    second.drive_background("i1")
    release.set()
    await first.wait_background("i1")
    await second.wait_background("i1")

    assert calls == 1
    assert loser.status == "running"
    assert (await store.get("i1")).status == "completed"


async def test_ticker_eviction_cancels_running_step():
    store = InMemoryWorkflowStore()
    step_cancelled = asyncio.Event()
    started = asyncio.Event()

    async def hang(ctx):
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            step_cancelled.set()
            raise

    wf = Workflow("t", [Step("gate", wait=True), Step("hang", hang)])
    engine = WorkflowEngine(store, {"t": wf}, lease_seconds=0.06)
    await engine.start("t", "i1", {})
    await engine.send_event("i1", "gate", drive=False)
    engine.drive_background("i1")
    await asyncio.wait_for(started.wait(), timeout=1)

    # A rival claim (as another replica would after our lease lapsed).
    store._by_id["i1"].leased_until = _now() - timedelta(seconds=1)
    rival = await store.claim_drive("i1", lease_seconds=60)
    assert rival is not None

    # The next renewal loses the fence, cancels the step, and the drive yields.
    await asyncio.wait_for(step_cancelled.wait(), timeout=1)
    result = await engine.wait_background("i1")
    assert result.status == "running"
    assert result.drive_gen == rival.drive_gen
    stored = await store.get("i1")
    assert stored.status == "running"  # the rival's claim, unclobbered
    assert "hang" not in stored.completed_steps


async def test_cancel_during_drive_stops_driver_and_notifies_once():
    store = InMemoryWorkflowStore()
    release = asyncio.Event()
    started = asyncio.Event()
    terminal: list[str] = []

    async def work(ctx):
        started.set()
        await release.wait()

    wf = Workflow("t", [Step("gate", wait=True), Step("work", work)])
    engine = WorkflowEngine(store, {"t": wf})
    engine.add_terminal_callback(lambda inst: _record(terminal, inst))
    await engine.start("t", "i1", {})
    await engine.send_event("i1", "gate", drive=False)
    engine.drive_background("i1")
    await asyncio.wait_for(started.wait(), timeout=1)

    cancelled = await engine.cancel("i1", "denied")
    assert cancelled.status == "cancelled"
    release.set()
    await engine.wait_background("i1")  # driver's post-step write loses

    stored = await store.get("i1")
    assert stored.status == "cancelled"
    assert "work" not in stored.completed_steps
    assert terminal == ["i1"]
    # Double-cancel: already terminal, no second callback.
    await engine.cancel("i1", "again")
    assert terminal == ["i1"]


async def _record(sink, instance):
    sink.append(instance.id)


async def test_racing_starts_run_the_first_step_once():
    store = InMemoryWorkflowStore()
    calls = 0

    async def gen(ctx):
        nonlocal calls
        calls += 1

    wf = Workflow("t", [Step("gen", gen, resumable=False), Step("gate", wait=True)])
    first = WorkflowEngine(store, {"t": wf})
    second = WorkflowEngine(store, {"t": wf})

    await asyncio.gather(
        first.start("t", "i1", {"seed": "one"}),
        second.start("t", "i1", {"seed": "two"}),
    )

    assert calls == 1
    assert (await store.get("i1")).status == "waiting"


async def test_drive_on_waiting_instance_is_a_noop():
    engine, store = _engine()
    await engine.start("t", "i1", {})  # parks at gate
    before = await store.get("i1")
    engine.drive_background("i1")
    after = await engine.wait_background("i1")
    assert after.status == "waiting"
    assert after.drive_gen == before.drive_gen  # no claim happened
    assert after.state["log"] == ["a"]


class _EvictingStore(InMemoryWorkflowStore):
    """Forces fence loss at a targeted write surface (state flag 'evict')."""

    async def fenced_write(self, instance, gen, *, release=False):
        if instance.state.get("evict"):
            return False
        return await super().fenced_write(instance, gen, release=release)


async def _drive_with_eviction(step_fn, *, extra_steps=()):
    """Run one step under an _EvictingStore; return (result, store, flags)."""
    store = _EvictingStore()
    flags = {"terminal": 0, "parked": 0}
    wf = Workflow("t", [Step("s", step_fn), *extra_steps])
    engine = WorkflowEngine(store, {"t": wf})

    async def on_terminal(inst):
        flags["terminal"] += 1

    async def on_park(inst):
        flags["parked"] += 1

    engine.add_terminal_callback(on_terminal)
    engine.add_park_callback(on_park)
    result = await engine.start("t", "i1", {})
    return result, store, flags


async def test_ownership_loss_at_midstep_checkpoint_yields_quietly():
    async def step(ctx):
        ctx.state["evict"] = True
        await ctx.checkpoint()
        raise AssertionError("checkpoint should have raised")

    result, store, flags = await _drive_with_eviction(step)
    stored = await store.get("i1")
    assert stored.status == "running"  # never marked failed
    assert stored.completed_steps == []
    assert flags == {"terminal": 0, "parked": 0}
    assert result.id == "i1"


async def test_ownership_loss_at_poststep_checkpoint_yields_quietly():
    async def step(ctx):
        ctx.state["evict"] = True  # the write after the step loses the fence

    result, store, flags = await _drive_with_eviction(
        step, extra_steps=(Step("gate", wait=True),)
    )
    stored = await store.get("i1")
    assert stored.status == "running"
    assert stored.completed_steps == []
    assert flags == {"terminal": 0, "parked": 0}


async def test_ownership_loss_at_park_write_skips_park_callbacks():
    from openloop.workflows.engine import WorkflowPark

    async def step(ctx):
        ctx.state["evict"] = True
        raise WorkflowPark("dynamic-gate")

    result, store, flags = await _drive_with_eviction(step)
    stored = await store.get("i1")
    assert stored.status == "running"
    assert stored.waiting_on is None
    assert flags == {"terminal": 0, "parked": 0}


async def test_ownership_loss_at_failure_write_does_not_mark_failed():
    async def step(ctx):
        ctx.state["evict"] = True
        raise RuntimeError("step blew up after eviction")

    result, store, flags = await _drive_with_eviction(step)
    stored = await store.get("i1")
    assert stored.status == "running"
    assert stored.error is None
    assert flags == {"terminal": 0, "parked": 0}


class _EvictOnReleaseStore(InMemoryWorkflowStore):
    """Fails only release writes, so the completion write loses in isolation
    while the preceding post-step checkpoint still lands."""

    async def fenced_write(self, instance, gen, *, release=False):
        if release and instance.state.get("evict_release"):
            return False
        return await super().fenced_write(instance, gen, release=release)


async def test_ownership_loss_at_completion_write_skips_terminal_callbacks():
    store = _EvictOnReleaseStore()
    terminal: list[str] = []

    async def step(ctx):
        ctx.state["evict_release"] = True

    wf = Workflow("t", [Step("s", step)])
    engine = WorkflowEngine(store, {"t": wf})
    engine.add_terminal_callback(lambda inst: _record(terminal, inst))

    result = await engine.start("t", "i1", {})

    stored = await store.get("i1")
    assert stored.completed_steps == ["s"]  # the post-step checkpoint landed
    assert stored.status == "running"  # the completed write lost the fence
    assert terminal == []
    assert result.id == "i1"
