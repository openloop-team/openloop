"""A minimal durable-workflow engine (the Architecture "Later" runtime item).

A :class:`Workflow` is an ordered list of named :class:`Step`s. The engine runs
them in order, persisting the :class:`WorkflowInstance` after each one, so a crash
resumes from the last completed step. A step marked ``wait`` parks the
instance: the engine persists ``status="waiting"`` and returns, and a later
:meth:`WorkflowEngine.send_event` delivers the awaited event and drives the rest.

This is how approval stops being a special case in ``ToolGateway.resolve``:
approval is just a wait node, and ``resolve`` becomes a thin adapter that emits
the approval event. Steps must be **idempotent** — a crash between a step's side
effect and its checkpoint write means the step re-runs on resume.

Wakeups are in-process today (``send_event`` can drive synchronously or only
record the wake, and a startup reconciler re-drives stale running instances).
Redis pub/sub can later make wakeups cross-process without changing the workflow
contract.

Coordination is **single-process**: the drive is guarded against concurrent
re-drive by an in-memory marker (``_active_drives``) plus an advisory lease
(``leased_until``, wall-clock, renewed by a ticker), and in-flight progress
notifications are drained per-instance from this process's task set. This is
correct for one runtime process. **Before running more than one replica**,
replace the advisory lease + in-memory marker with a DB atomic-claim arbiter for
the drive (``UPDATE … WHERE status='running' AND (leased_until IS NULL OR
leased_until < now()) RETURNING`` — server-side ``now()`` kills clock skew and
gives true mutual exclusion), because a second concurrent drive means double
spend and a double push. Best-effort UI writes (the Slack "still working…"
status) want the opposite primitive — level-triggered reconciliation against the
shared instance state, which converges without a leader — not the drain, which
is process-local.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta

from openloop.workflows.store import (
    TERMINAL,
    WorkflowInstance,
    WorkflowStore,
    _now,
)

logger = logging.getLogger(__name__)

DEFAULT_LEASE_SECONDS = 30.0


@dataclass(slots=True)
class WorkflowContext:
    """What a step sees: the live instance (mutate ``state`` / ``result``)."""

    instance: WorkflowInstance
    _checkpoint: Callable[[WorkflowInstance], Awaitable[None]] | None = None

    @property
    def state(self) -> dict:
        return self.instance.state

    async def checkpoint(self) -> None:
        """Persist current state and renew this run's lease."""
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
        self._active_drives: set[str] = set()
        self._terminal_callbacks: list[
            Callable[[WorkflowInstance], Awaitable[None]]
        ] = []
        self._progress_callbacks: list[
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
        are fired as detached tasks from :meth:`checkpoint`, never awaited on the
        checkpoint path, so a slow callback can't delay a lease renewal.
        """
        self._progress_callbacks.append(callback)

    async def checkpoint(self, instance: WorkflowInstance) -> None:
        """Persist mid-step state (e.g. after an idempotent write inside a step)."""
        if instance.status == "running":
            self._renew_lease(instance)
        await self.store.upsert(instance)
        if instance.status == "running":
            self._fire_progress(instance)

    async def start(
        self, workflow: str, instance_id: str, initial_state: dict
    ) -> WorkflowInstance:
        """Create a new instance and drive it to its first park/terminal.

        Idempotent on the instance id: if one already exists it is returned
        as-is, never re-driven — that could replay a non-resumable step. Resuming
        is the job of :meth:`send_event` (waits) and :meth:`resume_incomplete`
        (crashes), which both apply the resumability rules.
        """
        existing = await self.store.get(instance_id)
        if existing is not None:
            return existing
        instance = WorkflowInstance(
            id=instance_id, workflow=workflow, state=dict(initial_state)
        )
        await self.store.upsert(instance)
        return await self._drive(instance)

    async def send_event(
        self,
        instance_id: str,
        event: str,
        payload: dict | None = None,
        *,
        drive: bool = True,
    ) -> WorkflowInstance | None:
        """Deliver an awaited event, optionally driving the instance inline."""
        instance = await self.store.get(instance_id)
        if instance is None:
            return None
        if instance.status != "waiting" or instance.waiting_on != event:
            # Idempotent: the event was already consumed, or the instance moved
            # past this wait (e.g. a double-approve). Nothing to do.
            return instance
        instance.state.setdefault("events", {})[event] = payload or {}
        instance.completed_steps.append(event)
        instance.status = "running"
        instance.waiting_on = None
        self._renew_lease(instance)
        await self.store.upsert(instance)
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
        """Cancel a parked/running instance (e.g. its approval was denied)."""
        instance = await self.store.get(instance_id)
        if instance is None or instance.status in TERMINAL:
            return instance
        instance.status = "cancelled"
        instance.waiting_on = None
        instance.error = reason or None
        instance.leased_until = None
        await self.store.upsert(instance)
        await self._notify_terminal(instance)
        return instance

    async def resume_incomplete(self) -> list[str]:
        """Re-drive instances left ``running`` by a crash. Call once at startup.

        ``waiting`` instances stay parked (their event hasn't arrived);
        ``completed`` / ``failed`` are terminal. Idempotent; across replicas the
        app lifespan runs it under a ``startup-recovery``
        :class:`~openloop.coordination.DistributedLock` so only the leader sweeps.
        """
        resumed: list[str] = []
        now = _now()
        for instance in await self.store.recent(limit=1000):
            if instance.status != "running":
                continue
            if self._is_driving_in_process(instance.id):
                logger.debug(
                    "workflow %s is already being driven in this process; "
                    "skipping resume",
                    instance.id,
                )
                continue
            if instance.leased_until is not None and instance.leased_until > now:
                logger.debug(
                    "workflow %s is still leased until %s; skipping resume",
                    instance.id,
                    instance.leased_until,
                )
                continue
            workflow = self.workflows.get(instance.workflow)
            if workflow is None:
                # Its workflow isn't registered in this process; leave it be.
                continue
            if _has_pending_non_resumable_step(workflow, instance):
                # A non-resumable step (e.g. a chat turn's model call) hasn't
                # completed — resuming would replay it. Abandon instead.
                instance.status = "abandoned"
                instance.error = "interrupted before a non-resumable step completed"
                instance.leased_until = None
                await self.store.upsert(instance)
                await self._notify_terminal(instance)
                continue
            logger.info("resuming workflow %s (%s)", instance.id, instance.workflow)
            await self._drive(instance)
            resumed.append(instance.id)
        return resumed

    async def _drive(self, instance: WorkflowInstance) -> WorkflowInstance:
        """Run steps from where the instance left off, checkpointing each."""
        if instance.id in self._active_drives:
            return instance
        self._active_drives.add(instance.id)
        try:
            return await self._drive_active(instance)
        finally:
            self._active_drives.discard(instance.id)

    async def _drive_active(self, instance: WorkflowInstance) -> WorkflowInstance:
        """Run steps while this process owns the in-memory drive marker."""
        workflow = self.workflows.get(instance.workflow)
        if workflow is None:
            raise KeyError(f"unknown workflow {instance.workflow!r}")
        if instance.status in TERMINAL:
            return instance

        instance.status = "running"
        instance.waiting_on = None
        self._renew_lease(instance)
        await self.store.upsert(instance)

        for step in workflow.steps:
            if step.name in instance.completed_steps:
                continue
            if step.wait:
                instance.status = "waiting"
                instance.waiting_on = step.name
                instance.leased_until = None
                await self.store.upsert(instance)
                return instance
            ctx = WorkflowContext(instance, self.checkpoint)
            try:
                assert step.run is not None
                self._renew_lease(instance)
                await self.store.upsert(instance)
                await self._run_step_with_lease(step, ctx)
            except Exception as exc:  # noqa: BLE001 — record failure, don't crash caller
                instance.status = "failed"
                instance.error = str(exc)
                instance.leased_until = None
                await self.store.upsert(instance)
                logger.exception("workflow %s failed at step %s", instance.id, step.name)
                await self._notify_terminal(instance)
                return instance
            instance.completed_steps.append(step.name)
            self._renew_lease(instance)
            await self.store.upsert(instance)  # checkpoint after each step

        instance.status = "completed"
        instance.leased_until = None
        await self.store.upsert(instance)
        await self._notify_terminal(instance)
        return instance

    async def _drive_from_store(self, instance_id: str) -> WorkflowInstance | None:
        instance = await self.store.get(instance_id)
        if instance is None:
            return None
        return await self._drive(instance)

    def _renew_lease(self, instance: WorkflowInstance) -> None:
        instance.leased_until = _now() + timedelta(seconds=self.lease_seconds)

    async def _run_step_with_lease(
        self, step: Step, ctx: WorkflowContext
    ) -> None:
        ticker = asyncio.create_task(self._renew_lease_while_running(ctx.instance))
        try:
            assert step.run is not None
            await step.run(ctx)
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker

    async def _renew_lease_while_running(self, instance: WorkflowInstance) -> None:
        interval = max(self.lease_seconds / 3, 0.01)
        while True:
            await asyncio.sleep(interval)
            if instance.status != "running":
                return
            await self.checkpoint(instance)

    def _is_driving_in_process(self, instance_id: str) -> bool:
        if instance_id in self._active_drives:
            return True
        task = self._background.get(instance_id)
        return task is not None and not task.done()

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
        later status write re-sets it). Best-effort: a request already physically
        sent to the surface in the same instant can't be un-sent — this closes the
        common window, not the theoretical one against an unordered surface API.
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
