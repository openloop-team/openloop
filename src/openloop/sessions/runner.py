"""Session runner — binds one surface session to one ``agent_task`` workflow.

This is the delivery layer Phase D adds on top of Phase C's durable chat turn.
Given an inbound surface event the runner:

1. creates (or re-uses) a :class:`SurfaceSession`, idempotent on the event id;
2. sets a transient progress indicator and marks the session ``running``;
3. drives the turn via :meth:`Runtime.handle`, binding the workflow instance id
   to the session id so the two share one identity;
4. records the result/error on the session and asks the
   :class:`~openloop.sessions.delivery.SurfaceDelivery` to post the final answer.

Progress is coarse for this first pass (``queued`` → ``running`` → ``waiting`` /
``completed`` / ``failed``). Every durable delivery is guarded by a persisted
message id, so a duplicate event never posts a second final answer, and a retry
of a session that crashed *after* reaching a terminal state but *before* posting
re-delivers it once. The narrow window between a successful provider post and
recording its message id — where the persisted-id guard can't help — is covered
by a deterministic delivery key: every post is tagged with it and the recovery
path looks the message up by key instead of re-posting (best-effort; a surface
whose lookup can't run degrades back to at-least-once). One gap remains by
design: a session that crashed mid-turn is recovered by the startup reconciler
(Slice 6), not this inline path (it must not replay the model call). The
original request does not own the task's lifetime — the runner does, and it can
be awaited inline (tests) or scheduled in the background (Slack).

**Thread-bound continuation.** A session is one *turn*; a workspace task can span
many. When the thread already owns a durable task
(:mod:`openloop.tasks.binding`) and the reply is eligible to continue it
(:mod:`openloop.tasks.continuation`), the runner skips the model entirely and
runs that task's next turn instead: same task id, same branch and pull request,
same approval and spend attribution, delivered under this turn's own session. The
task is reconstructed from its durable record every time, so a continuation after
a restart is indistinguishable from one a second later — and while a task's turn
is in flight the thread's queued replies wait for it rather than racing a second
execution onto the same workspace.
"""

from __future__ import annotations

import logging
import time
import uuid

from openloop.deliverable import Artifact, Deliverable, Prose
from openloop.runtime import Runtime, Task
from openloop.runtime.pipeline import _result_content
from openloop.sessions.delivery import SurfaceDelivery
from openloop.sessions.store import (
    TERMINAL,
    SurfaceSession,
    SurfaceSessionStore,
    SurfaceTarget,
)
from openloop.sessions.threads import (
    ThreadRecordStore,
    TranscriptFragment,
    thread_scope_key,
)
from openloop.tasks.binding import BUSY, CLOSED, ThreadTask, ThreadTaskStore
from openloop.tasks.continuation import (
    ContinuationUnavailable,
    continuation_instance_id,
    continuation_state,
    may_continue,
)
from openloop.tasks.contract import (
    START_GATE_EVENT as _TASK_START_GATE,
    WORKSPACE_TASK_WORKFLOW as _TASK_WORKFLOW,
)
from openloop.workflows.store import TERMINAL as _WORKFLOW_TERMINAL

logger = logging.getLogger(__name__)

PROGRESS_STATUS_TEXT = "is thinking..."
# Slack's assistant-thread status is transient — it lapses if not re-asserted.
# Re-send the current phrase at least this often (even unchanged) so a long,
# single-phase run keeps showing "still working…" instead of going blank. Bursts
# of identical ticks within the window still collapse. Kept below the lease
# ticker's ~lease/3 cadence so each tick refreshes.
PROGRESS_REFRESH_SECONDS = 5.0
WAITING_TEXT = "⏳ Waiting for approval…"
# A workspace-task turn that parked mid-work (e.g. on an action decision). Only
# a repair path ever posts it — the decision card carries the real question.
TASK_WAITING_TEXT = "⏳ Working on this task…"
ERROR_TEXT = "⚠️ This task was interrupted and could not be completed."

# How many prior thread turns to replay as conversation history. A safety bound
# on context size, not a correctness limit — older turns fall back to recall.
HISTORY_TURN_LIMIT = 20

def _is_non_terminal_invocation(inv) -> bool:
    if inv.status in ("started", "approved"):
        # "approved" = a durable decision with no result yet (a losing
        # concurrent click on a direct tool, or an effect that can't run
        # here): update the card informationally, keep the session waiting,
        # and let the winner's real outcome deliver.
        return True
    data = getattr(getattr(inv, "result", None), "data", {}) or {}
    return data.get("status") in {"running", "waiting"}


def _prose_of(result: "Deliverable | str") -> str:
    """The replay-safe prose of a deliverable — what transcripts/history keep."""
    if isinstance(result, Artifact):
        return result.summary
    if isinstance(result, Prose):
        return result.text
    return result


def _deliverable_from_outcome_data(data) -> "Deliverable | None":
    """Rebuild a DIRECT-DELIVER outcome from a terminal result's ``outcome``
    block, or ``None`` for anything that must fall back to the existing
    prose/model-continuation path.

    ``data`` is a tool/workflow result's ``.data`` — untrusted at this
    boundary: it may not be a dict at all, its ``outcome`` may be missing or
    not itself a dict (e.g. a bare string), its ``kind`` may be unrecognized,
    or a recognized kind may be missing a required field (a still-evolving
    connector, hand-built test data, or a future strategy's partial payload).
    ANY of that degrades to ``None`` rather than raising: this is called from
    :meth:`SessionRunner._continue_session`, itself reachable from
    :meth:`SessionRunner.reconcile`'s per-session sweep — that loop has no
    per-iteration try/except, so one raise here would abort the *whole* sweep,
    and the recovery loop re-runs every interval, re-poisoning it forever.

    This single accessor also IS the "which outcomes deliver directly"
    policy, so callers need no separate kind check: only ``diagnosis`` and
    ``evidence_bundle`` ever produce a Deliverable here. ``pull_request``
    (the coding worker's outcome), ``failed``, and anything unrecognized
    always return ``None`` — even when perfectly well-formed — so that
    connector keeps today's unchanged model-continuation (M0b) behavior.
    """
    try:
        outcome = data.get("outcome") if isinstance(data, dict) else None
        if not isinstance(outcome, dict):
            return None
        kind = outcome.get("kind")
        if kind not in ("diagnosis", "evidence_bundle"):
            return None
        from openloop.tasks.outcomes import Diagnosis, EvidenceBundle, to_deliverable

        if kind == "diagnosis":
            built = Diagnosis(text=outcome["text"])
        else:
            built = EvidenceBundle(
                summary=outcome["summary"],
                findings=outcome["findings"],
                title=outcome.get("title", "Investigation findings"),
                filename=outcome.get("filename", "findings.md"),
            )
        return to_deliverable(built)
    except Exception:  # noqa: BLE001 — never poison the reconcile sweep
        logger.warning(
            "ignoring malformed outcome data; falling back to the prose/"
            "model-continuation path",
            exc_info=True,
        )
        return None


def _approval_id_for_instance(instance) -> str | None:
    state = getattr(instance, "state", {}) or {}
    if state.get("approval_id"):
        return state["approval_id"]
    event = (state.get("events") or {}).get("await_approval") or {}
    return event.get("approval_id")


def _task_id_of(instance) -> str | None:
    """The workspace task a ``workspace_task`` instance is a turn of.

    Reads the durable state's own identity, never the instance id: those are the
    same string only for a first turn, and conflating them is exactly the
    assumption continuation exists to remove.
    """
    state = getattr(instance, "state", {}) or {}
    return state.get("task_id") or state.get("job_id")


def _inbox_payload(task: Task, target: SurfaceTarget) -> dict:
    """Serialize just enough to reconstruct the task + delivery target at drain
    time. History is intentionally omitted — it's rebuilt from the (by then
    delivered) transcript when the turn actually runs."""
    return {
        "text": task.text,
        "user": task.user,
        "kind": task.kind,
        "surface": target.surface,
        "workspace": target.workspace,
        "agent": target.agent,
        "channel": target.channel,
        "thread": target.thread,
        "event_id": target.event_id,
    }


def _task_target_from_payload(p: dict) -> tuple[Task, SurfaceTarget]:
    task = Task(
        text=p["text"], surface=p["surface"], channel=p.get("channel"),
        user=p.get("user"), kind=p.get("kind"),
    )
    target = SurfaceTarget(
        surface=p["surface"], workspace=p["workspace"], agent=p["agent"],
        channel=p.get("channel"), thread=p.get("thread"),
        event_id=p.get("event_id"),
    )
    return task, target


class SessionRunner:
    """Runs a task as a background session and delivers the answer back."""

    def __init__(
        self,
        runtime: Runtime,
        sessions: SurfaceSessionStore,
        delivery: SurfaceDelivery,
        threads: "ThreadRecordStore | None" = None,
        tasks: "ThreadTaskStore | None" = None,
    ) -> None:
        self.runtime = runtime
        self.sessions = sessions
        self.delivery = delivery
        # The durable thread↔task bindings. When present, a reply in a thread
        # that already owns a workspace task continues THAT task — same id,
        # same branch, same PR, same authorization — instead of running a model
        # turn that would delegate a second one. Absent, every turn is a fresh
        # model turn (the pre-continuation behavior).
        self.tasks = tasks
        # Phase A: the thread-scoped delivered-transcript store. When present, a
        # follow-up turn's history is the real conversation (request→answer per
        # delivered turn) rather than the per-session summary scan; when absent the
        # runner falls back to SurfaceSessionStore.thread_history (old path).
        self.threads = threads
        # (phrase, last-sent monotonic) per session: collapse identical bursts,
        # but still re-assert periodically so Slack's transient status doesn't
        # lapse during a long single-phase run.
        self._progress_seen: dict[str, tuple[str, float]] = {}
        engine = getattr(runtime, "engine", None)
        if engine is not None and hasattr(engine, "add_terminal_callback"):
            # Several runners may share one engine in tests or multi-surface
            # wiring. All callbacks may fire; delivery stays correct because the
            # persisted final_message_id/key guards below make it idempotent.
            engine.add_terminal_callback(self._on_workflow_terminal)
            if hasattr(engine, "add_progress_callback"):
                engine.add_progress_callback(self._on_workflow_progress)
            if hasattr(engine, "add_park_callback"):
                engine.add_park_callback(self._on_workflow_parked)

    async def run(self, task: Task, target: SurfaceTarget) -> SurfaceSession:
        """Create/resume a session for ``task`` and deliver its outcome.

        Idempotent on ``target.event_id``: a duplicate inbound event reuses the
        existing session rather than starting a second turn. If that session
        reached a terminal state but crashed before its answer was posted, the
        retry re-delivers it (guarded by the persisted message id, so never
        twice). A session still mid-turn is left for the startup reconciler
        (Slice 6) — this inline retry path does not replay the model call.
        """
        existing = await self.sessions.get_by_event(target.event_id)
        if existing is not None:
            return await self._ensure_delivered(existing)

        # Phase B: tag the turn with its thread's warm-context key so a workflow-
        # backed tool (the coding worker) can reuse this thread's warm checkout.
        # Only threaded turns have warm context; a top-level turn stays cold.
        if target.thread is not None:
            task.thread_key = thread_scope_key(target)

        session = SurfaceSession(
            id=uuid.uuid4().hex,
            target=target,
            status="queued",
            # Persist the inbound text so a later turn in this thread can replay it
            # as conversation history (see _apply_thread_history).
            request_text=task.text,
        )
        # One session : one workflow instance — share the id so the approval
        # continuation / reconciler can map between them trivially.
        session.workflow_instance_id = session.id
        # Step 5: tag the turn with its session id so a workflow-backed tool
        # (the coding worker) attributes its spend to the originating session
        # (UsageRecord.session_id). Unlike thread_key, every turn has one.
        task.session_id = session.id
        try:
            await self.sessions.upsert(session)
        except Exception:  # noqa: BLE001 — a concurrent duplicate won the race
            # The event_id unique index rejected this insert: another delivery of
            # the same event created the session first. Defer to the winner.
            racer = await self.sessions.get_by_event(target.event_id)
            if racer is not None:
                return await self._ensure_delivered(racer)
            raise

        await self._set_progress_status(session)
        session.status = "running"
        await self.sessions.upsert(session)

        # Thread-bound continuation: if this thread already owns a workspace task
        # and the reply is eligible to continue it, this turn IS that task's next
        # turn — no model round, no second delegation, no new task identity.
        continued = await self._maybe_continue_task(session, task, target)
        if continued is not None:
            return continued

        # Replay earlier turns of this thread so the model has the conversation in
        # context, not just semantic recall. Done before handle() so the history
        # is baked into the workflow's persisted turn state (resume-safe).
        await self._apply_thread_history(task, session)
        # TEMP DEBUG (thread-isolation diagnosis): show exactly which thread this
        # turn resolved to and how many prior turns were replayed as history.
        logger.debug(
            "THREAD-DEBUG event=%s channel=%r thread=%r history_turns=%d session=%s",
            target.event_id,
            target.channel,
            target.thread,
            len(task.history) // 2,
            session.id,
        )

        try:
            response = await self.runtime.handle(
                task, instance_id=session.workflow_instance_id
            )
        except Exception as exc:  # noqa: BLE001 — record + deliver, don't crash caller
            logger.exception("session %s failed while handling the task", session.id)
            session.status = "failed"
            session.error = str(exc)
            await self.sessions.upsert(session)
            await self._post_error(session)
            return session

        return await self._deliver(session, response)

    async def run_threaded(self, task: Task, target: SurfaceTarget) -> None:
        """Serialize a thread's turns: enqueue this reply, then drain the thread's
        inbox one turn at a time.

        Two replies to the same thread must not run concurrently — the later one
        has to see the earlier's delivered answer as context, and racing them would
        also double-drive. So an inbound reply is appended to the durable inbox and
        then :meth:`_drain_thread` tries to become the thread's single drain leader.
        Falls back to a direct :meth:`run` when there is no thread store or no
        thread/event scope to serialize on.
        """
        if self.threads is None or target.thread is None or not target.event_id:
            await self.run(task, target)
            return
        await self.threads.append_inbox(
            target, target.event_id, _inbox_payload(task, target)
        )
        await self._drain_thread(target)

    async def _drain_thread(self, target: SurfaceTarget) -> None:
        """Drain a thread's queued replies, one turn at a time.

        The caller tries to become the thread's single drain leader
        (``try_begin_turn``, an atomic CAS). The winner drains every queued turn
        via :meth:`run` (itself idempotent on ``event_id``) until the inbox is
        empty, then releases; a loser simply returns, its reply left for the
        leader. The outer re-claim loop closes the window where a reply lands
        after the last dequeue but before the release.

        Stops — leaving everything queued — while the thread's bound workspace
        task is mid-turn: a reply must join the task it belongs to, not race a
        second execution of it onto the same branch. The task's own terminal
        transition drains the rest (see :meth:`_on_workflow_terminal`), and the
        startup sweep drains anything a crash left queued, so a held reply is
        never lost — only deferred.
        """
        while True:
            if await self._task_busy(target):
                return
            if not await self.threads.try_begin_turn(target):
                return
            try:
                while (item := await self.threads.next_inbox(target)) is not None:
                    turn_task, turn_target = _task_target_from_payload(item.payload)
                    await self.run(turn_task, turn_target)
                    if await self._task_busy(target):
                        break
            finally:
                await self.threads.end_turn(target)

    # --- thread-bound workspace-task continuation ---

    async def _bound_task(self, target: SurfaceTarget) -> "ThreadTask | None":
        """This thread's durable workspace-task binding, if it has one."""
        if self.tasks is None or target.thread is None:
            return None
        return await self.tasks.bound(thread_scope_key(target))

    async def _task_busy(self, target: SurfaceTarget) -> bool:
        """Whether this thread's bound task currently has a turn in flight.

        Two independent signals, because neither alone is trustworthy after a
        crash: the durable claim (a turn said it was driving) and the task's
        workflow instance (what is actually running). A task parked at its
        *start gate* is deliberately not busy — nothing is executing, it owns no
        workspace state yet, and a reply while the approval card is up must stay
        an ordinary turn rather than freezing the thread.
        """
        record = await self._bound_task(target)
        if record is None:
            return False
        if record.status == BUSY:
            return True
        return await self._instance_busy(record.instance_id)

    async def _instance_busy(self, instance_id: str | None) -> bool:
        engine = getattr(self.runtime, "engine", None)
        if engine is None or not instance_id:
            return False
        instance = await engine.store.get(instance_id)
        if instance is None or instance.status in _WORKFLOW_TERMINAL:
            return False
        if instance.status == "waiting" and instance.waiting_on == _TASK_START_GATE:
            return False
        return True

    async def _maybe_continue_task(
        self, session: SurfaceSession, task: Task, target: SurfaceTarget
    ) -> "SurfaceSession | None":
        """Run this turn as the next turn of the thread's bound task, or not.

        ``None`` means "not a continuation" — the caller runs the ordinary model
        turn, which can still delegate new work through the approval gate. Every
        refusal here is deliberate and fail-open in that direction: an ineligible
        reply, a task another turn already claimed, or a record that cannot be
        re-entered cold all fall back rather than block the user.
        """
        if self.tasks is None or target.thread is None:
            return None
        engine = getattr(self.runtime, "engine", None)
        if engine is None or _TASK_WORKFLOW not in getattr(engine, "workflows", {}):
            # No engine, or this process cannot run the task's workflow (the
            # connector is disabled here) — never pretend to continue it.
            return None
        record = await self._bound_task(target)
        if not may_continue(record, user=task.user):
            return None
        instance_id = continuation_instance_id(record.task_id, session.id)
        claimed = await self.tasks.claim(
            record.task_id, instance_id=instance_id, session_id=session.id
        )
        if claimed is None:
            logger.info(
                "workspace task %s is claimed elsewhere; running an ordinary turn",
                record.task_id,
            )
            return None
        try:
            state = continuation_state(
                claimed, request=task.text, session_id=session.id
            )
        except ContinuationUnavailable:
            logger.warning(
                "workspace task %s cannot be continued from its durable record",
                claimed.task_id,
                exc_info=True,
            )
            await self.tasks.release(claimed.task_id)
            return None

        session.task_id = claimed.task_id
        # The turn's recovery pointer is the task's instance, not a model turn's.
        session.workflow_instance_id = instance_id
        await self.sessions.upsert(session)
        logger.info(
            "continuing workspace task %s as turn %d (instance %s)",
            claimed.task_id,
            claimed.turns,
            instance_id,
        )
        try:
            instance = await engine.start(_TASK_WORKFLOW, instance_id, state)
        except Exception as exc:  # noqa: BLE001 — record + deliver, don't crash
            logger.exception(
                "failed to start continuation of workspace task %s", claimed.task_id
            )
            await self.tasks.release(claimed.task_id)
            session.status = "failed"
            session.error = str(exc)
            await self.sessions.upsert(session)
            await self._post_error(session)
            return session
        return await self._deliver_task_turn(session, instance)

    async def _deliver_task_turn(
        self, session: SurfaceSession, instance
    ) -> SurfaceSession:
        """Deliver one turn of a workspace task, idempotently.

        Reached twice for the same instance by design — once inline from the
        continuation that started it, once from the engine's terminal callback
        (and again from the startup reconciler after a crash). Every write is
        guarded on the session's persisted delivery state, so the second and
        third arrivals are no-ops.
        """
        fresh = await self.sessions.get(session.id)
        if fresh is not None:
            session = fresh
        status = getattr(instance, "status", None)
        if status not in _WORKFLOW_TERMINAL:
            # Parked mid-task (an OpenHands action decision, say). The park
            # callback posts the decision; this turn is delivered when the task
            # reaches a terminal state.
            if session.final_message_id is None and session.status not in TERMINAL:
                session.status = "waiting"
                session.result_summary = session.result_summary or TASK_WAITING_TEXT
                await self.sessions.upsert(session)
            return session
        await self._release_task(session.task_id, instance)
        self._progress_seen.pop(session.id, None)
        if session.final_message_id is not None or session.status in TERMINAL:
            return session
        if status == "completed":
            result = getattr(instance, "result", None) or {}
            session.status = "completed"
            session.result_summary = result.get("summary") or "Done."
            await self.sessions.upsert(session)
            await self._post_final(session, session.result_summary)
            return session
        session.status = "failed"
        session.error = getattr(instance, "error", None) or ERROR_TEXT
        await self.sessions.upsert(session)
        await self._post_error(session)
        return session

    async def _release_task(self, task_id: str | None, instance) -> None:
        """Refresh the durable task from the turn that just ended, and unclaim it.

        The refreshed blob is what a cold replica reconstructs the task from, so
        the delivered pull-request identity is folded in here — it lives on the
        workflow's result, and nothing else durable would carry it forward.
        """
        if self.tasks is None or not task_id:
            return
        state = dict(getattr(instance, "state", None) or {})
        result = getattr(instance, "result", None) or {}
        profile_state = state.get("profile_state")
        if isinstance(profile_state, dict):
            code = profile_state.get("code")
            if isinstance(code, dict):
                for key in ("branch", "pr_number", "pr_url"):
                    if result.get(key) is not None:
                        code[key] = result[key]
        try:
            if getattr(instance, "status", None) == "cancelled":
                # Cancelled means its authorization was withdrawn (a denial, or
                # an explicit cancel) — the task never continues.
                await self.tasks.release(
                    task_id, state=state or None, status=CLOSED, reason="workflow cancelled"
                )
                return
            await self.tasks.release(task_id, state=state or None)
        except Exception:  # noqa: BLE001 — delivery must never fail on bookkeeping
            logger.exception("failed to release workspace task %s", task_id)

    async def _resume_thread(self, target: SurfaceTarget) -> None:
        """Drain replies a busy task deferred. A no-op while a drain is active."""
        if self.threads is None or target.thread is None:
            return
        try:
            await self._drain_thread(target)
        except Exception:  # noqa: BLE001 — never fail a workflow callback
            logger.exception("failed to resume thread drain after a task turn")

    async def _recover_task_claims(self) -> list[str]:
        """Repair claims a crash orphaned; returns the repaired task ids.

        A turn that died mid-flight leaves its task claimed, which would wedge
        the thread forever. The durable workflow instance is the arbiter: if it
        is terminal (or gone), the claim is released and the task's record is
        refreshed from it; if it is still live, the engine's own resume drives
        it and its terminal transition releases the claim.
        """
        if self.tasks is None:
            return []
        engine = getattr(self.runtime, "engine", None)
        repaired: list[str] = []
        for record in await self.tasks.claimed():
            if record.instance_id and engine is not None:
                instance = await engine.store.get(record.instance_id)
                if instance is not None and instance.status not in _WORKFLOW_TERMINAL:
                    continue
                if instance is not None:
                    await self._release_task(record.task_id, instance)
                    repaired.append(record.task_id)
                    continue
            await self.tasks.release(record.task_id)
            repaired.append(record.task_id)
        return repaired

    async def _drain_pending_threads(self, sessions: list[SurfaceSession]) -> None:
        """Drain threads that still hold a queued reply nobody is draining.

        Replies deferred while a task was busy sit in the durable inbox with no
        leader once the process dies. This sweep runs on the recovery interval,
        so it asks the store which scopes actually have something queued rather
        than probing every thread it has ever seen — in the ordinary case that
        is one query returning nothing. A scope with no recent session to
        address it from is left for the next inbound event, which carries its
        own target.
        """
        if self.threads is None:
            return
        pending = getattr(self.threads, "pending_scopes", None)
        if pending is None:
            return
        try:
            scopes = set(await pending())
        except Exception:  # noqa: BLE001 — the sweep's other repairs still stand
            logger.exception("failed to list threads with queued replies")
            return
        if not scopes:
            return
        seen: set[str] = set()
        for session in sessions:
            target = session.target
            if target.thread is None:
                continue
            key = thread_scope_key(target)
            if key not in scopes or key in seen:
                continue
            seen.add(key)
            try:
                await self._drain_thread(target)
            except Exception:  # noqa: BLE001 — one thread must not poison the sweep
                logger.exception("failed to drain thread %s during recovery", key)

    async def _deliver(self, session: SurfaceSession, response) -> SurfaceSession:
        if response.model == "error":
            # The workflow was interrupted inside a non-resumable model step.
            session.status = "abandoned"
            session.error = response.text or ERROR_TEXT
            await self.sessions.upsert(session)
            await self._post_error(session)
            return session

        if response.approval_ids:
            # Parked on a human approval. Persist the approval ids so Slice 4 can
            # map a button click back to this session and post the eventual answer.
            session.status = "waiting"
            session.approval_ids = list(response.approval_ids)
            session.result_summary = response.text or WAITING_TEXT
            await self.sessions.upsert(session)
            # Post (or update) a durable approval card with buttons in-thread.
            requests = await self._approval_requests(session.approval_ids)
            await self._post_or_update_approval(
                session, response.text or WAITING_TEXT, requests
            )
            return session

        session.status = "completed"
        session.result_summary = response.text or "(no response)"
        await self.sessions.upsert(session)
        await self._post_final(session, session.result_summary)
        return session

    async def resolve_approval(
        self, approval_id: str, approver: str, *, approve: bool
    ) -> str:
        """Resolve an approval and continue the session that was waiting on it.

        Resolves the approval through the tool gateway. Immediate tools still
        deliver their outcome here; workflow-backed tools only return a started
        status, leave the session waiting, and deliver later from the terminal
        workflow callback or reconciler. Returns the status line for the
        button-click reply.

        Delivery failures never block the button reply and always leave the
        session in a repairable state: a session left ``waiting`` retries the
        whole continuation on the next click; one already flipped terminal but
        not yet delivered is repaired idempotently from its persisted outcome. So
        even if the tool side effect succeeds but a Slack post fails, a second
        click (or the startup reconciler) still delivers the answer.
        """
        from openloop.surfaces.approvals import resolution_message

        tools = getattr(self.runtime, "tools", None)
        if tools is None:
            return "⛔ Approvals are not available right now."
        inv = await tools.resolve(approval_id, approver, approve=approve)
        message = resolution_message(inv, approver)

        session = await self.sessions.get_by_approval(approval_id)
        if session is not None:
            try:
                if session.status == "waiting":
                    await self._continue_session(
                        session, inv, approver, message, approval_id=approval_id
                    )
                elif session.status in TERMINAL and session.final_message_id is None:
                    # A prior continuation flipped the session terminal but a Slack
                    # post failed before the answer landed — re-deliver it from the
                    # persisted outcome (idempotent; reuses result_summary).
                    await self._ensure_delivered(session)
            except Exception:  # noqa: BLE001 — leave it repairable, still reply
                logger.exception(
                    "failed to deliver approval outcome for session %s", session.id
                )
        return message

    async def reconcile(self) -> list[str]:
        """Repair delivery state for sessions left mid-flight by a crash.

        Call once at startup, **after** the workflow engine's own
        ``resume_incomplete`` has driven crashed turns to a terminal state. For
        each session:

        - ``waiting`` (parked on a human approval) or already-delivered → leave
          it alone;
        - terminal but with no final message (the turn finished but a Slack post
          failed, or it crashed between the status flip and the post) →
          re-deliver from the persisted outcome;
        - still ``queued`` / ``running`` (the turn crashed before it was
          delivered) → recover the answer from the now-terminal workflow instance
          and deliver it, or post an interrupted notice if it can't be recovered.

        Idempotent, so safe to run on every boot. Across replicas, the app lifespan
        runs it under a ``startup-recovery`` :class:`~openloop.coordination.\
        DistributedLock` so only the leader sweeps; delivery stays id-/key-guarded
        if two ever overlap.
        """
        repaired: list[str] = []
        # Before any session is judged: free tasks a crashed turn left claimed,
        # so the sweep (and every thread they wedge) can make progress.
        repaired.extend(await self._recover_task_claims())
        recent = await self.sessions.recent(limit=1000)
        for session in recent:
            if session.status == "waiting":
                if await self._deliver_terminal_approval(session):
                    repaired.append(session.id)
                    continue
                if session.progress_message_id is None and session.approval_ids:
                    requests = await self._approval_requests(session.approval_ids)
                    if requests:
                        await self._post_or_update_approval(
                            session,
                            session.result_summary or WAITING_TEXT,
                            requests,
                            recover=True,
                        )
                        repaired.append(session.id)
                continue
            if session.final_message_id is not None:
                continue
            if session.status in TERMINAL:
                await self._ensure_delivered(session)
                repaired.append(session.id)
                continue
            # queued / running — recover from the workflow the session is bound to.
            if session.task_id is not None:
                # A workspace-task turn: its answer comes from the task's own
                # instance, never from a model turn's state.
                if await self._recover_task_session(session):
                    repaired.append(session.id)
                continue
            found, response = await self._recover(session)
            if response is not None:
                await self._deliver(session, response)
            elif not found:
                # No recoverable workflow (missing instance / no engine) → notice.
                session.status = "abandoned"
                session.error = ERROR_TEXT
                await self.sessions.upsert(session)
                await self._post_error(session)
            else:
                # The workflow exists but isn't terminal yet — leave it for a later
                # restart rather than delivering a half-finished turn.
                continue
            repaired.append(session.id)
        # Replies a busy task deferred are still queued with nobody draining
        # them; a restart is exactly when that has to be picked back up.
        await self._drain_pending_threads(recent)
        return repaired

    async def _recover_task_session(self, session: SurfaceSession) -> bool:
        """Recover a crashed workspace-task turn; True when it was repaired."""
        engine = getattr(self.runtime, "engine", None)
        instance = (
            await engine.store.get(session.workflow_instance_id)
            if engine is not None and session.workflow_instance_id
            else None
        )
        if instance is None:
            await self._release_task_claim(session.task_id)
            session.status = "abandoned"
            session.error = ERROR_TEXT
            await self.sessions.upsert(session)
            await self._post_error(session)
            return True
        if instance.status not in _WORKFLOW_TERMINAL:
            # Live (or resumable): the engine's own resume drives it and its
            # terminal transition delivers this session.
            return False
        await self._deliver_task_turn(session, instance)
        return True

    async def _release_task_claim(self, task_id: str | None) -> None:
        if self.tasks is None or not task_id:
            return
        try:
            await self.tasks.release(task_id)
        except Exception:  # noqa: BLE001 — recovery must not fail on bookkeeping
            logger.exception("failed to release workspace task %s", task_id)

    async def _apply_thread_history(self, task: Task, session: SurfaceSession) -> None:
        """Populate ``task.history`` from earlier delivered turns in this thread.

        Rebuilds the conversation from the durable sessions — each prior delivered
        exchange contributes a ``user`` (its request) + ``assistant`` (its answer)
        pair, oldest-first — rather than re-fetching the surface's own transcript.
        That keeps it surface-agnostic and free of delivery scaffolding (progress
        notes, approval cards never appear). The store decides what's replayable
        (only completed, *delivered* exchanges — never an answer the user didn't
        see; see ``thread_history``), so this just maps them to messages. A caller
        that already supplied history is left untouched, and a session with no
        thread (or the thread's first turn) simply gets no history.
        """
        if task.history or session.target.thread is None:
            return
        turns: list[dict[str, str]] = []
        if self.threads is not None:
            # Phase A: read the thread-scoped delivered transcript (request→answer).
            for frag in await self.threads.replayable_transcript(
                session.target, exclude_turn_id=session.id, limit=HISTORY_TURN_LIMIT
            ):
                turns.append({"role": "user", "content": frag.request})
                turns.append({"role": "assistant", "content": frag.answer})
        else:
            # Fallback: reconstruct from the per-session delivered-turn scan.
            for s in await self.sessions.thread_history(
                session.target, exclude_id=session.id, limit=HISTORY_TURN_LIMIT
            ):
                turns.append({"role": "user", "content": s.request_text})
                turns.append({"role": "assistant", "content": s.result_summary})
        if turns:
            task.history = turns

    async def _recover(self, session: SurfaceSession) -> tuple[bool, object]:
        """``(found, response)`` for a session's workflow — see
        :meth:`Runtime.recover_response`."""
        instance_id = session.workflow_instance_id
        recover = getattr(self.runtime, "recover_response", None)
        if instance_id is None or recover is None:
            return False, None
        return await recover(instance_id)

    async def _continue_session(
        self, session: SurfaceSession, inv, approver: str, message: str,
        approval_id: str | None = None,
    ) -> None:
        """Apply an approval outcome without treating non-terminal work as final."""
        fresh = await self.sessions.get(session.id)
        if fresh is not None:
            session = fresh
        if _is_non_terminal_invocation(inv):
            if session.final_message_id is not None or session.status in TERMINAL:
                return
            session.status = "waiting"
            session.result_summary = (
                inv.result.summary if inv.result else (inv.message or message)
            )
            await self.sessions.upsert(session)
            try:
                await self._update_approval(session, message, [])
            except Exception:  # noqa: BLE001 — buttons going stale is cosmetic
                logger.exception(
                    "failed to mark approval started for session %s", session.id
                )
            return
        if inv.status == "executed":
            if session.final_message_id is not None:
                return
            # Stage 1 Phase 2: an evidence-bundle/diagnosis outcome IS the final
            # answer (findings/diagnosis prose) — deliver it directly, no model
            # re-run. _deliverable_from_outcome_data is the single, defensive
            # accessor: it IS the "which kinds direct-deliver" policy (only
            # evidence_bundle/diagnosis — a pull_request outcome, a result with
            # no outcome block, or malformed outcome data all return None), so
            # there's no separate kind check here, and nothing this call can
            # raise. None falls through to the existing M0b path below
            # unchanged, so the coding worker keeps today's model-continuation
            # behavior.
            outcome_data = inv.result.data if inv.result else None
            deliverable = _deliverable_from_outcome_data(outcome_data)
            if deliverable is not None:
                # Persist the outcome (so a failed post is repairable from
                # result_summary) before posting — mirrors the pattern below.
                session.status = "completed"
                session.result_summary = _prose_of(deliverable)
                await self.sessions.upsert(session)
                await self._post_final(session, deliverable)
                try:
                    await self._update_approval(session, message, [])
                except Exception:  # noqa: BLE001 — buttons going stale is cosmetic
                    logger.exception(
                        "failed to collapse approval card for session %s",
                        session.id,
                    )
                return
            # M0b: re-run the model with the approved result folded in, so the reply
            # is a fresh model answer — not the raw tool summary. Falls back to the
            # summary if the continuation can't be built (no engine / lost state).
            if approval_id and await self._continue_with_model(
                session, approval_id, inv, approver, message
            ):
                return
            detail = inv.result.summary if inv.result else (inv.message or "done")
            final_text = detail
        elif inv.status == "denied":
            # Name the canonical decider (the approval row's decided_by), not
            # the clicker — a losing/reconciler-driven denial still attributes
            # to whoever actually decided.
            final_text = f"🚫 Denied by {inv.decided_by or approver}."
        else:  # forbidden / not-an-approver / already resolved — leave it parked
            return
        # Persist the outcome (so a failed post is repairable from result_summary),
        # then deliver the ANSWER first — the approval card collapse is cosmetic and
        # must never block or lose the final reply.
        session.status = "completed"
        session.result_summary = final_text
        await self.sessions.upsert(session)
        await self._post_final(session, final_text)
        try:
            await self._update_approval(session, message, [])
        except Exception:  # noqa: BLE001 — buttons going stale is cosmetic
            logger.exception(
                "failed to collapse approval card for session %s", session.id
            )

    async def _continue_with_model(
        self, session: SurfaceSession, approval_id: str, inv, approver: str,
        message: str,
    ) -> bool:
        """Re-run the model with the approved tool result folded in, under the SAME
        session (M0b). Returns True if it drove a continuation, False if it could
        not (caller then falls back to delivering the tool summary).

        The continuation is a *new* ``agent_task`` instance under the same
        ``SurfaceSession`` — a deterministic id (``{session.id}:cont:{approval_id}``)
        so a re-spawn is idempotent — seeded with the original turn's message log
        after the approved call's held placeholder is replaced by the real result.
        The resume-aware loop then sees the round resolved and the next model call
        produces a fresh answer, delivered under the session's one delivery record.
        """
        runtime = self.runtime
        engine = getattr(runtime, "engine", None)
        cont = getattr(runtime, "continue_turn", None)
        if engine is None or cont is None or session.workflow_instance_id is None:
            return False
        prior = await engine.store.get(session.workflow_instance_id)
        if prior is None:
            return False
        messages = [dict(m) for m in (prior.state.get("messages") or [])]
        call_id = (prior.state.get("approval_calls") or {}).get(approval_id)
        result_content = _result_content(inv.result) if inv.result else "done"
        folded = False
        for m in messages:
            if m.get("role") == "tool" and m.get("tool_call_id") == call_id:
                m["content"] = result_content  # held placeholder -> real result
                folded = True
                break
        if not folded:
            return False

        task = Task(
            text=session.request_text or "",
            surface=session.target.surface,
            channel=session.target.channel,
            # Same thread → same warm context for any follow-on write.
            thread_key=(
                thread_scope_key(session.target)
                if session.target.thread is not None
                else None
            ),
            # Same session → same spend attribution if the continuation issues a
            # new write (step 5).
            session_id=session.id,
        )
        cont_id = f"{session.id}:cont:{approval_id}"
        response = await cont(task, messages, instance_id=cont_id)
        # The continuation is a new instance under the same session: repoint recovery
        # at it, then deliver the fresh answer through the normal path (which re-parks
        # on a *new* approval if the model asked for another write). Keep the resolved
        # approval id on the session so the second-click / reconciler repair path can
        # still map back to it (`_deliver` overwrites it only on a new approval).
        session.workflow_instance_id = cont_id
        await self.sessions.upsert(session)
        await self._deliver(session, response)
        try:
            await self._update_approval(session, message, [])
        except Exception:  # noqa: BLE001 — buttons going stale is cosmetic
            logger.exception(
                "failed to collapse approval card for session %s", session.id
            )
        return True

    async def _final_deliverable(self, session: SurfaceSession) -> "Deliverable | str":
        """What a (re-)delivery of this session's final answer should post."""
        return session.result_summary or "(no response)"

    async def _on_workflow_terminal(self, instance) -> None:
        is_task = getattr(instance, "workflow", None) == _TASK_WORKFLOW
        if is_task:
            # A continuation turn owns its own session and delivers directly:
            # there is no approval to fold in, because the task passed its start
            # gate turns ago and this turn ran under that same authorization.
            turn = await self._task_turn_session(instance)
            if turn is not None:
                await self._deliver_task_turn(turn, instance)
                await self._resume_thread(turn.target)
                return
            # A first turn: the approval path below delivers it, but the durable
            # binding is refreshed here — this is the moment the task acquires
            # the branch/PR identity a later reply continues.
            await self._release_task(_task_id_of(instance), instance)
        approval_id = _approval_id_for_instance(instance)
        if not approval_id:
            return
        session = await self.sessions.get_by_approval(approval_id)
        if session is None:
            return
        self._progress_seen.pop(session.id, None)
        await self._deliver_terminal_approval(session)
        if is_task:
            await self._resume_thread(session.target)

    async def _task_turn_session(self, instance) -> "SurfaceSession | None":
        """The session delivering this instance as a workspace-task turn."""
        if self.tasks is None:
            return None
        get_by_instance = getattr(self.sessions, "get_by_instance", None)
        if get_by_instance is None:
            return None
        session = await get_by_instance(getattr(instance, "id", "") or "")
        if session is None or session.task_id is None:
            return None
        return session

    async def _on_workflow_parked(self, instance) -> None:
        """Deliver an OpenHands decision only after its parked state is durable."""
        waiting_on = getattr(instance, "waiting_on", None) or ""
        if not waiting_on.startswith("openhands_decision:"):
            return
        state = getattr(instance, "state", {}) or {}
        decision = state.get("openhands_decision") or {}
        decision_id = decision.get("decision_id")
        summary = decision.get("summary")
        if not decision_id or not summary:
            return
        # A continuation turn parks under its own session; a first turn parks
        # under the session its approval card belongs to.
        session = await self._task_turn_session(instance)
        if session is not None:
            if session.status in TERMINAL:
                return
        else:
            approval_id = _approval_id_for_instance(instance)
            if not approval_id:
                return
            session = await self.sessions.get_by_approval(approval_id)
            if session is None or session.status != "waiting":
                return
        update = getattr(self.delivery, "update_openhands_decision", None)
        post = getattr(self.delivery, "post_openhands_decision", None)
        if session.progress_message_id is not None and update is not None:
            await update(
                session.target,
                session.progress_message_id,
                instance.id,
                decision_id,
                summary,
            )
            return
        if post is not None:
            session.progress_message_id = await post(
                session.target,
                instance.id,
                decision_id,
                summary,
                key=f"{session.id}:openhands:{decision_id}",
            )
            await self.sessions.upsert(session)

    async def resolve_openhands_decision(
        self,
        job_id: str,
        decision_id: str,
        *,
        kind: str,
        actor_id: str,
        event_id: str,
    ) -> str:
        """Authorize, durably record, and asynchronously drive a Slack action."""
        from openloop.tasks import WorkspaceTask
        from openloop.tools.openhands_resume import OpenHandsResumeState, ResumeDecision

        engine = getattr(self.runtime, "engine", None)
        if engine is None:
            return "⛔ This task can't be resumed right now."
        instance = await engine.store.get(job_id)
        event = f"openhands_decision:{decision_id}"
        if (
            instance is None
            or instance.status != "waiting"
            or instance.waiting_on != event
        ):
            return "⛔ That decision is stale or already resolved."
        # WorkspaceTask.from_dict rehydrates either layout: a NEW nested
        # instance's durable worker_state lives at
        # profile_state["code"]["worker_state"]; an OLD flat-layout instance
        # (parked before this contract-convergence rework shipped) is lifted
        # there by the same compat shim workflows/coding_worker.py relies on.
        task = WorkspaceTask.from_dict(instance.state or {})
        raw_worker = task.profile_state.get("code", {}).get("worker_state") or {}
        raw_resume = raw_worker.get("openhands_resume")
        try:
            resume = OpenHandsResumeState.from_dict(raw_resume)
        except Exception:
            return "⛔ The paused task state is invalid."
        if resume.decision_id != decision_id:
            return "⛔ That decision is stale."
        if actor_id != resume.slack_requester_id:
            return "⛔ Only the user who approved this task may decide."
        decision = ResumeDecision(
            kind=kind,
            decision_id=decision_id,
            event_id=event_id,
            actor_id=actor_id,
        )
        await engine.send_event(job_id, event, decision.to_dict(), drive=False)
        # The card belongs to whichever turn posted it: a continuation turn's own
        # session, or — for a first turn — the session holding the approval.
        session = await self._task_turn_session(instance)
        if session is None:
            approval_id = _approval_id_for_instance(instance)
            session = (
                await self.sessions.get_by_approval(approval_id)
                if approval_id is not None
                else None
            )
        if session is not None and session.progress_message_id is not None:
            label = "accepted" if kind == "accept" else "rejected"
            try:
                await self.delivery.update_approval(
                    session.target,
                    session.progress_message_id,
                    f"Action {label} by @{actor_id}; resuming…",
                    [],
                )
            except Exception:  # noqa: BLE001 — decision is already durable
                logger.warning("failed to collapse OpenHands decision card", exc_info=True)
        engine.drive_background(job_id)
        return "✅ Decision recorded; resuming work."

    async def _on_workflow_progress(self, instance) -> None:
        """Relay a running workflow's progress phrase as a transient status.

        Best-effort UI: maps the instance back to its waiting session via the
        approval id and pushes ``instance.state['progress']`` to the surface,
        deduped so an unchanged phrase never re-hits the API.
        """
        # The instance is mutated in place by the drive, so a task scheduled
        # during the last step but running just after completion sees the terminal
        # status here and bails — the guard the engine's drain can't cover for a
        # task that already started running.
        if getattr(instance, "status", None) in _WORKFLOW_TERMINAL:
            return
        phrase = (getattr(instance, "state", {}) or {}).get("progress")
        if not phrase:
            return
        session = await self._task_turn_session(instance)
        if session is not None:
            if session.status in TERMINAL:
                return
        else:
            approval_id = _approval_id_for_instance(instance)
            if not approval_id:
                return
            session = await self.sessions.get_by_approval(approval_id)
            if session is None or session.status != "waiting":
                return
        last = self._progress_seen.get(session.id)
        now = time.monotonic()
        if last is not None:
            last_phrase, last_at = last
            if last_phrase == phrase and now - last_at < PROGRESS_REFRESH_SECONDS:
                return
        self._progress_seen[session.id] = (phrase, now)
        await self._set_progress_status(session, phrase)

    async def _deliver_terminal_approval(self, session: SurfaceSession) -> bool:
        """Deliver a waiting session whose approval reached a terminal outcome.

        Covers two crash shapes the decision reconciler leaves behind: an
        approved workflow that finished, and a denied request whose Slack
        session/card is still parked. The denied case needs no engine — its
        reconcile-side cancel already ran — so the engine requirement gates
        only the approved-workflow branch.
        """
        tools = getattr(self.runtime, "tools", None)
        if tools is None:
            return False
        engine = getattr(tools, "engine", None) or getattr(self.runtime, "engine", None)
        from openloop.surfaces.approvals import resolution_message
        from openloop.tools.base import Invocation
        from openloop.tools.gateway import _workflow_invocation

        for approval_id in session.approval_ids:
            request = await tools.approvals.get(approval_id)
            if request is None:
                continue
            approver = request.decided_by or "an approver"
            if request.status == "denied":
                inv = Invocation(
                    status="denied",
                    message="action denied",
                    decided_by=request.decided_by,
                )
                await self._continue_session(
                    session, inv, approver, resolution_message(inv, approver),
                    approval_id=approval_id,
                )
                return True
            if request.status != "approved":
                continue
            if engine is None:
                continue
            # Route on the durable execution marker, not the live tool shape: a
            # decided row's mode must never drift. _classify yields the trusted
            # instance id (the stamped workflow_instance_id, or _instance_id for
            # a legacy workflow row) — never a model-supplied args['job_id'],
            # which could name an unrelated live workflow.
            kind, instance_id = tools._classify(request)
            if kind != "workflow":
                continue
            instance = await engine.store.get(instance_id)
            if instance is None or instance.status not in _WORKFLOW_TERMINAL:
                continue
            inv = _workflow_invocation(instance)
            inv.decided_by = request.decided_by
            await self._continue_session(
                session, inv, approver, resolution_message(inv, approver),
                approval_id=approval_id,
            )
            return True
        return False

    async def _ensure_delivered(self, session: SurfaceSession) -> SurfaceSession:
        """Re-deliver an existing session's answer if it crashed before posting.

        Called for a duplicate event / retry. The ``_post_*`` helpers are guarded
        by ``final_message_id``, so a fully delivered session is returned
        untouched while a terminal-but-undelivered one finally gets its answer. A
        session still ``queued`` / ``running`` (a mid-turn crash) is returned
        as-is — recovering those is the reconciler's job, not this synchronous
        retry path (which must not replay the model call). A waiting session
        that lacks an approval card can repair that card from persisted approval
        ids.
        """
        if session.final_message_id is not None:
            return session
        # This is the retry path: the post may already have landed before its id
        # was persisted, so ask delivery to recover-or-post (recover=True) rather
        # than blindly re-posting and duplicating the answer. An artifact final is
        # re-materialized from its persisted ref (degrading to the prose summary).
        if session.status == "completed":
            await self._post_final(
                session, await self._final_deliverable(session), recover=True
            )
        elif session.status in ("failed", "abandoned"):
            await self._post_error(session, recover=True)
        elif session.status == "waiting" and session.progress_message_id is None:
            requests = await self._approval_requests(session.approval_ids)
            if requests:
                await self._post_or_update_approval(
                    session,
                    session.result_summary or WAITING_TEXT,
                    requests,
                    recover=True,
                )
        return session

    # --- idempotent delivery helpers (guarded by persisted message ids) ---

    @staticmethod
    def _delivery_key(session: SurfaceSession, role: str) -> str:
        """Deterministic dedup key for one of a session's posts.

        Stable across retries (keyed on the session id), so a recovery post can
        find the message a crashed first attempt already sent. One key per role so
        approval / final / error never collide.
        """
        return f"{session.id}:{role}"

    async def _set_progress_status(
        self, session: SurfaceSession, text: str = PROGRESS_STATUS_TEXT
    ) -> None:
        try:
            await self.delivery.set_progress_status(session.target, text)
        except Exception:  # noqa: BLE001 — status is transient UI polish
            logger.warning(
                "failed to set progress status for session %s",
                session.id,
                exc_info=True,
            )

    async def _update_approval(
        self, session: SurfaceSession, text: str, requests
    ) -> None:
        if session.progress_message_id is None:
            return
        await self.delivery.update_approval(
            session.target, session.progress_message_id, text, requests
        )

    async def _post_or_update_approval(
        self, session: SurfaceSession, text: str, requests, *, recover: bool = False
    ) -> None:
        if session.progress_message_id is not None:
            await self._update_approval(session, text, requests)
            return
        mid = await self.delivery.post_approval(
            session.target,
            text,
            requests,
            key=self._delivery_key(session, "approval"),
            recover=recover,
        )
        session.progress_message_id = mid
        await self.sessions.upsert(session)

    async def _approval_requests(self, approval_ids: list[str]) -> list:
        """Fetch the pending ApprovalRequest objects so delivery can render them."""
        tools = getattr(self.runtime, "tools", None)
        if tools is None:
            return []
        out = []
        for rid in approval_ids:
            req = await tools.approvals.get(rid)
            if req is not None and req.status == "pending":
                out.append(req)
        return out

    async def _post_final(
        self, session: SurfaceSession, result: "Deliverable | str", *,
        recover: bool = False,
    ) -> None:
        if session.final_message_id is not None:
            return  # already delivered — never post a second final answer
        mid = await self.delivery.post_final(
            session.target,
            result,
            key=self._delivery_key(session, "final"),
            recover=recover,
        )
        session.final_message_id = mid
        await self.sessions.upsert(session)
        # Post-delivery, commit the turn to the thread's delivered transcript so a
        # later turn replays it as real conversation. Idempotent on the session id,
        # so a redelivery/reconcile never double-appends; only after the answer
        # actually reached the thread (final_message_id recorded above). Only the
        # replay-safe prose is recorded — an artifact body never enters history.
        await self._record_transcript(session, _prose_of(result))

    async def _record_transcript(self, session: SurfaceSession, answer: str) -> None:
        if self.threads is None or session.target.thread is None:
            return
        if not session.request_text or not answer:
            return
        try:
            await self.threads.append_delivered_fragment(
                session.target,
                TranscriptFragment(
                    turn_id=session.id, request=session.request_text, answer=answer
                ),
            )
        except Exception:  # noqa: BLE001 — transcript is history, never block delivery
            logger.warning(
                "failed to record thread transcript for session %s",
                session.id,
                exc_info=True,
            )

    async def _post_error(
        self, session: SurfaceSession, *, recover: bool = False
    ) -> None:
        if session.final_message_id is not None:
            return
        mid = await self.delivery.post_error(
            session.target, session.error or ERROR_TEXT,
            key=self._delivery_key(session, "error"), recover=recover,
        )
        session.final_message_id = mid
        await self.sessions.upsert(session)
