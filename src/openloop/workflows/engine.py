"""A minimal durable-workflow engine (the Architecture "Later" runtime item).

A :class:`Workflow` is an ordered list of named :class:`Step`s. The engine runs
them in order, persisting the :class:`WorkflowInstance` after each one, so a crash
resumes from the last completed step. A step marked ``wait`` parks the
instance: the engine persists ``status="waiting"`` and returns, and a later
:meth:`WorkflowEngine.send_event` delivers the awaited event and drives the rest.

This is how approval stops being a special case in ``ToolGateway.resolve``:
approval is just a wait node, and ``resolve`` becomes a thin adapter that emits
the approval event. Steps must be **idempotent** — a crash between a step's side
effect and its checkpoint write means the step re-runs on resume. State persists
only at explicit checkpoints (step completion, park, terminal, and mid-step
:meth:`WorkflowContext.checkpoint` calls); the lease-renewal ticker renews the
lease without committing state, so replay always resumes from a deliberate
checkpoint.

Wakeups are in-process today (``send_event`` can drive synchronously or only
record the wake, and a startup reconciler re-drives stale running instances).
Redis pub/sub can later make wakeups cross-process without changing the workflow
contract.

Drive arbitration is a **DB atomic claim fenced by ``drive_gen``**: a drive
starts by winning ``claim_drive`` (server-clock lease check, gen bump), and
every subsequent write is fenced by that gen, so across any number of replicas
there is **at most one durable workflow writer** per instance. Losing the fence
(:class:`DriveOwnershipLost`) stops the drive without further writes and
cancels the running step task — best-effort curtailment: work already inside a
thread, subprocess, or container survives task cancellation, so an evicted
step's external side effects can outlive its ownership until per-adapter abort
contracts exist. Event consumption (``claim_event``) and cancellation
(``cancel_instance``) are their own atomic claims; the startup-recovery lock
around :meth:`resume_incomplete` is an efficiency layer, not the correctness
mechanism. Best-effort UI writes (the Slack "still working…" status) are
drained per-instance from this process's task set on the terminal transition;
across an ownership handoff a straggler from the previous owner can still land
late — level-triggered reconciliation against shared instance state would
close that, and remains future work.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial

from openloop.workflows.store import (
    TERMINAL,
    WorkflowInstance,
    WorkflowStore,
)

logger = logging.getLogger(__name__)

DEFAULT_LEASE_SECONDS = 30.0


class DriveOwnershipLost(Exception):
    """This drive's fenced write lost to a newer claim; the instance moved on."""


@dataclass(slots=True)
class WorkflowContext:
    """What a step sees: the live instance (mutate ``state`` / ``result``)."""

    instance: WorkflowInstance
    _checkpoint: Callable[[WorkflowInstance], Awaitable[None]] | None = None

    @property
    def state(self) -> dict:
        return self.instance.state

    async def checkpoint(self) -> None:
        """Persist current state under this drive's ownership fence."""
        if self._checkpoint is not None:
            await self._checkpoint(self.instance)


StepFn = Callable[[WorkflowContext], Awaitable[None]]


@dataclass(slots=True)
class Step:
    """One named step. ``wait`` nodes park the instance until an event arrives.

    ``resumable=False`` marks a step that must not be replayed (e.g. a chat turn's
    non-idempotent model call). A crash is recoverable only once every
    non-resumable step has completed; before that the instance is abandoned rather
    than re-driven into the non-resumable step. Idempotent steps stay resumable.
    """

    name: str
    run: StepFn | None = None
    wait: bool = False
    resumable: bool = True


@dataclass(slots=True)
class Workflow:
    name: str
    steps: list[Step]


class WorkflowPark(Exception):
    """A running step reached a durable, dynamically named wait boundary."""

    def __init__(self, event: str) -> None:
        super().__init__(event)
        self.event = event


class WorkflowEngine:
    """Runs workflows durably: checkpoint per step, park on wait, resume on event."""

    def __init__(
        self,
        store: WorkflowStore,
        workflows: dict[str, Workflow] | None = None,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> None:
        self.store = store
        self.workflows = workflows or {}
        self.lease_seconds = lease_seconds
        self._background: dict[str, asyncio.Task] = {}
        self._terminal_callbacks: list[
            Callable[[WorkflowInstance], Awaitable[None]]
        ] = []
        self._progress_callbacks: list[
            Callable[[WorkflowInstance], Awaitable[None]]
        ] = []
        self._park_callbacks: list[
            Callable[[WorkflowInstance], Awaitable[None]]
        ] = []
        # Strong refs to in-flight progress notifications, keyed by instance so a
        # terminal transition can drain just that instance's stragglers (not other
        # workers') before delivering the final answer. Each self-removes on done.
        self._progress_tasks: dict[str, set[asyncio.Task]] = {}

    def register(self, workflow: Workflow) -> None:
        self.workflows[workflow.name] = workflow

    def add_terminal_callback(
        self, callback: Callable[[WorkflowInstance], Awaitable[None]]
    ) -> None:
        """Run ``callback`` after a workflow reaches a terminal state."""
        self._terminal_callbacks.append(callback)

    def add_progress_callback(
        self, callback: Callable[[WorkflowInstance], Awaitable[None]]
    ) -> None:
        """Run ``callback`` on each mid-step checkpoint of a running instance.

        Best-effort UI signal (e.g. a Slack "still working…" status): callbacks
        are fired as detached tasks from checkpoints and lease renewals, never
        awaited on those paths, so a slow callback can't delay a lease renewal.
        """
        self._progress_callbacks.append(callback)

    def add_park_callback(
        self, callback: Callable[[WorkflowInstance], Awaitable[None]]
    ) -> None:
        """Run after a dynamic wait is durable (used for decision delivery)."""
        self._park_callbacks.append(callback)

    async def start(
        self, workflow: str, instance_id: str, initial_state: dict
    ) -> WorkflowInstance:
        """Create a new instance and drive it to its first park/terminal.

        Idempotent on the instance id: creation is an atomic
        ``INSERT … ON CONFLICT DO NOTHING``, so a racing ``start`` returns the
        existing instance as-is, never re-driven — that could replay a
        non-resumable step. Resuming is the job of :meth:`send_event` (waits)
        and :meth:`resume_incomplete` (crashes), which both apply the
        resumability rules.
        """
        instance = WorkflowInstance(
            id=instance_id, workflow=workflow, state=dict(initial_state)
        )
        if not await self.store.create(instance):
            existing = await self.store.get(instance_id)
            return existing if existing is not None else instance
        return await self._drive(instance)

    async def send_event(
        self,
        instance_id: str,
        event: str,
        payload: dict | None = None,
        *,
        drive: bool = True,
    ) -> WorkflowInstance | None:
        """Deliver an awaited event, optionally driving the instance inline.

        Event consumption and drive ownership are separate atomic claims: the
        claimed instance is ``running`` with no lease, and whichever drive
        attempt reaches ``claim_drive`` first — this one, a
        :meth:`drive_background` task, or a recovery sweep — wins cleanly.
        """
        instance = await self.store.claim_event(instance_id, event, payload or {})
        if instance is None:
            # Idempotent: missing, already consumed, or waiting on a newer
            # decision. Most importantly, a losing replica never drives.
            return await self.store.get(instance_id)
        if not drive:
            return instance
        return await self._drive(instance)

    def drive_background(self, instance_id: str) -> asyncio.Task:
        """Drive a running instance in a background task, coalesced per process."""
        existing = self._background.get(instance_id)
        if existing is not None and not existing.done():
            return existing
        task = asyncio.create_task(self._drive_from_store(instance_id))
        self._background[instance_id] = task

        def _forget(done: asyncio.Task) -> None:
            self._background.pop(instance_id, None)
            try:
                done.result()
            except Exception:
                logger.exception("background workflow drive failed for %s", instance_id)

        task.add_done_callback(_forget)
        return task

    async def wait_background(self, instance_id: str) -> WorkflowInstance | None:
        """Test/helper hook: await an in-process background drive if one exists."""
        task = self._background.get(instance_id)
        if task is None:
            return await self.store.get(instance_id)
        return await task

    async def cancel(self, instance_id: str, reason: str = "") -> WorkflowInstance | None:
        """Cancel a parked/running instance (e.g. its approval was denied).

        Atomic across replicas: the winning cancel bumps ``drive_gen``, so a
        live driver's next fenced write fails and it stops. Terminal callbacks
        fire only from the winning transition (at-most-once).
        """
        cancelled = await self.store.cancel_instance(instance_id, reason)
        if cancelled is None:
            # Already terminal, or no such instance — nothing to notify.
            return await self.store.get(instance_id)
        await self._notify_terminal(cancelled)
        return cancelled

    async def resume_incomplete(self) -> list[str]:
        """Re-drive instances left ``running`` by a crash. Call once at startup.

        ``waiting`` instances stay parked (their event hasn't arrived);
        ``completed`` / ``failed`` are terminal. Per-instance mutual exclusion
        is the drive claim itself — a live-leased instance simply loses the
        claim and is skipped. The app lifespan runs this under the
        ``startup-recovery`` :class:`~openloop.coordination.DistributedLock`,
        which only keeps replicas from redundantly sweeping; correctness does
        not depend on it.
        """
        resumed: list[str] = []
        for instance in await self.store.recent(limit=1000):
            if instance.status != "running":
                continue
            workflow = self.workflows.get(instance.workflow)
            if workflow is None:
                # Its workflow isn't registered in this process; leave it be.
                continue
            claimed = await self.store.claim_drive(
                instance.id, lease_seconds=self.lease_seconds
            )
            if claimed is None:
                logger.debug(
                    "workflow %s is claimed elsewhere; skipping resume",
                    instance.id,
                )
                continue
            if _has_pending_non_resumable_step(workflow, claimed):
                # Decide from the authoritative claimed row, not the earlier
                # recent() snapshot: the previous owner may have checkpointed
                # the non-resumable step between the scan and this claim.
                claimed.status = "abandoned"
                claimed.error = "interrupted before a non-resumable step completed"
                try:
                    await self._write_owned(claimed, claimed.drive_gen, release=True)
                except DriveOwnershipLost:
                    continue
                await self._notify_terminal(claimed)
                continue
            logger.info("resuming workflow %s (%s)", instance.id, instance.workflow)
            await self._drive_owned(workflow, claimed)
            resumed.append(instance.id)
        return resumed

    async def _drive(self, instance: WorkflowInstance) -> WorkflowInstance:
        """Claim drive ownership and run steps; a lost claim is a no-op."""
        workflow = self.workflows.get(instance.workflow)
        if workflow is None:
            raise KeyError(f"unknown workflow {instance.workflow!r}")
        claimed = await self.store.claim_drive(
            instance.id, lease_seconds=self.lease_seconds
        )
        if claimed is None:
            # Another driver holds a live lease, or the instance is parked or
            # terminal — either way this attempt has nothing to do.
            current = await self.store.get(instance.id)
            return current if current is not None else instance
        return await self._drive_owned(workflow, claimed)

    async def _drive_owned(
        self, workflow: Workflow, instance: WorkflowInstance
    ) -> WorkflowInstance:
        """Run steps while holding the claim; the ownership-loss boundary.

        This outer catch must wrap the *entire* owned drive: ownership can be
        lost at the post-step checkpoint, at the park write inside the
        ``WorkflowPark`` handler, at the failed-terminal write inside the
        generic handler, or at the final completed write — and an exception
        raised inside an ``except`` block escapes its sibling handlers.
        """
        gen = instance.drive_gen
        try:
            return await self._run_steps(workflow, instance, gen)
        except DriveOwnershipLost:
            logger.warning(
                "workflow %s drive (gen %d) lost ownership; yielding to the "
                "new owner without further writes",
                instance.id,
                gen,
            )
            current = await self.store.get(instance.id)
            return current if current is not None else instance

    async def _run_steps(
        self, workflow: Workflow, instance: WorkflowInstance, gen: int
    ) -> WorkflowInstance:
        if instance.status in TERMINAL:
            return instance
        for step in workflow.steps:
            if step.name in instance.completed_steps:
                continue
            if step.wait:
                instance.status = "waiting"
                instance.waiting_on = step.name
                await self._write_owned(instance, gen, release=True)
                return instance
            ctx = WorkflowContext(instance, partial(self._checkpoint_owned, gen=gen))
            try:
                await self._run_step_with_lease(step, ctx, gen)
            except WorkflowPark as park:
                instance.status = "waiting"
                instance.waiting_on = park.event
                await self._write_owned(instance, gen, release=True)
                for callback in list(self._park_callbacks):
                    try:
                        await callback(instance)
                    except Exception:
                        logger.exception(
                            "workflow park callback failed for %s", instance.id
                        )
                return instance
            except DriveOwnershipLost:
                # Re-raise past the generic handler: an eviction mid-step is
                # not a step failure, and a fenced failure-write would be
                # doomed anyway.
                raise
            except Exception as exc:  # noqa: BLE001 — record failure, don't crash caller
                instance.status = "failed"
                instance.error = str(exc)
                await self._write_owned(instance, gen, release=True)
                logger.exception("workflow %s failed at step %s", instance.id, step.name)
                await self._notify_terminal(instance)
                return instance
            instance.completed_steps.append(step.name)
            await self._write_owned(instance, gen)  # checkpoint after each step
        instance.status = "completed"
        await self._write_owned(instance, gen, release=True)
        await self._notify_terminal(instance)
        return instance

    async def _drive_from_store(self, instance_id: str) -> WorkflowInstance | None:
        instance = await self.store.get(instance_id)
        if instance is None:
            return None
        return await self._drive(instance)

    async def _write_owned(
        self, instance: WorkflowInstance, gen: int, *, release: bool = False
    ) -> None:
        if not await self.store.fenced_write(instance, gen, release=release):
            raise DriveOwnershipLost(instance.id)
        if release:
            # Release is defined by both stores as one generation bump plus
            # lease clearance. Keep the caller-visible snapshot aligned with
            # the durable row without adding a terminal/park read round-trip.
            instance.drive_gen = gen + 1
            instance.leased_until = None

    async def _checkpoint_owned(
        self, instance: WorkflowInstance, *, gen: int
    ) -> None:
        """Mid-step checkpoint: fenced state write plus the progress signal."""
        await self._write_owned(instance, gen)
        if instance.status == "running":
            self._fire_progress(instance)

    async def _run_step_with_lease(
        self, step: Step, ctx: WorkflowContext, gen: int
    ) -> None:
        assert step.run is not None
        step_task = asyncio.create_task(step.run(ctx))
        ticker = asyncio.create_task(
            self._renew_lease_while_running(ctx.instance, gen, step_task)
        )
        try:
            await step_task
        except asyncio.CancelledError:
            # The ticker cancels the step on ownership loss; surface that as
            # DriveOwnershipLost. Any other cancellation propagates as-is.
            if ticker.done() and not ticker.cancelled():
                exc = ticker.exception()
                if isinstance(exc, DriveOwnershipLost):
                    raise exc from None
            raise
        finally:
            ticker.cancel()
            # If the ticker already ended with DriveOwnershipLost but the step
            # finished first, don't let the finally clause mask the step's
            # outcome — the next fenced write will surface the loss anyway.
            with contextlib.suppress(asyncio.CancelledError, DriveOwnershipLost):
                await ticker

    async def _renew_lease_while_running(
        self, instance: WorkflowInstance, gen: int, step_task: asyncio.Task
    ) -> None:
        interval = max(self.lease_seconds / 3, 0.01)
        while True:
            await asyncio.sleep(interval)
            if instance.status != "running":
                return
            try:
                renewed = await self.store.renew_lease(
                    instance.id, gen, lease_seconds=self.lease_seconds
                )
            except Exception:
                # A store hiccup isn't an eviction; keep trying. If the lease
                # truly lapses, another claimant's gen bump fails our next
                # renewal or fenced write, and that path evicts us cleanly.
                logger.exception(
                    "lease renewal errored for workflow %s; retrying", instance.id
                )
                continue
            if not renewed:
                # Evicted: stop the step's side effects as fast as we can.
                # Best-effort — work inside threads/subprocesses/containers
                # survives task cancellation.
                step_task.cancel()
                raise DriveOwnershipLost(instance.id)
            self._fire_progress(instance)

    async def _notify_terminal(self, instance: WorkflowInstance) -> None:
        if instance.status not in TERMINAL:
            return
        # Cancel in-flight progress writes before the terminal callbacks post the
        # final answer, so no stale "still working…" status lands after it.
        await self._drain_progress(instance.id)
        for callback in list(self._terminal_callbacks):
            try:
                await callback(instance)
            except Exception:
                logger.exception(
                    "workflow terminal callback failed for %s", instance.id
                )

    def _fire_progress(self, instance: WorkflowInstance) -> None:
        """Schedule progress callbacks as detached tasks (never blocks the caller)."""
        for callback in self._progress_callbacks:
            task = asyncio.create_task(self._run_progress(callback, instance))
            self._progress_tasks.setdefault(instance.id, set()).add(task)
            task.add_done_callback(
                lambda t, iid=instance.id: self._discard_progress(iid, t)
            )

    def _discard_progress(self, instance_id: str, task: asyncio.Task) -> None:
        tasks = self._progress_tasks.get(instance_id)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                self._progress_tasks.pop(instance_id, None)

    async def _drain_progress(self, instance_id: str) -> None:
        """Cancel and await this instance's in-flight progress tasks.

        Called on the terminal transition *before* the final answer is posted, so
        a straggler ``set_progress_status`` can't land after the post and leave a
        stale "still working…" status behind (the final post clears it, and no
        later status write re-sets it). Best-effort and process-local: it can't
        reach a previous owner's stragglers after a cross-process handoff — a
        request already physically sent to the surface can't be un-sent. This
        closes the common window, not the theoretical one.
        """
        tasks = self._progress_tasks.pop(instance_id, None)
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _run_progress(
        self,
        callback: Callable[[WorkflowInstance], Awaitable[None]],
        instance: WorkflowInstance,
    ) -> None:
        try:
            await callback(instance)
        except Exception:
            logger.exception(
                "workflow progress callback failed for %s", instance.id
            )


def _has_pending_non_resumable_step(
    workflow: Workflow, instance: WorkflowInstance
) -> bool:
    """True if a non-resumable step has not yet completed.

    Once every non-resumable step is done, only idempotent steps remain and the
    instance is safe to re-drive on resume.
    """
    done = set(instance.completed_steps)
    return any(
        not step.resumable and step.name not in done for step in workflow.steps
    )
