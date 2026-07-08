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
"""

from __future__ import annotations

import logging
import time
import uuid

from openloop.runtime import Runtime, Task
from openloop.runtime.pipeline import _result_content
from openloop.sessions.delivery import SurfaceDelivery
from openloop.sessions.store import (
    TERMINAL,
    SurfaceSession,
    SurfaceSessionStore,
    SurfaceTarget,
)
from openloop.sessions.threads import ThreadRecordStore, TranscriptFragment
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
ERROR_TEXT = "⚠️ This task was interrupted and could not be completed."

# How many prior thread turns to replay as conversation history. A safety bound
# on context size, not a correctness limit — older turns fall back to recall.
HISTORY_TURN_LIMIT = 20


def _is_non_terminal_invocation(inv) -> bool:
    if inv.status == "started":
        return True
    data = getattr(getattr(inv, "result", None), "data", {}) or {}
    return data.get("status") in {"running", "waiting"}


def _approval_id_for_instance(instance) -> str | None:
    state = getattr(instance, "state", {}) or {}
    if state.get("approval_id"):
        return state["approval_id"]
    event = (state.get("events") or {}).get("await_approval") or {}
    return event.get("approval_id")


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
    ) -> None:
        self.runtime = runtime
        self.sessions = sessions
        self.delivery = delivery
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
        then the caller tries to become the thread's single drain leader
        (``try_begin_turn``, an atomic CAS). The winner drains every queued turn via
        :meth:`run` (itself idempotent on ``event_id``) until the inbox is empty,
        then releases; a loser simply returns, its reply left for the leader. The
        outer re-claim loop closes the window where a reply lands after the last
        dequeue but before the release. Falls back to a direct :meth:`run` when
        there is no thread store or no thread/event scope to serialize on.
        """
        if self.threads is None or target.thread is None or not target.event_id:
            await self.run(task, target)
            return
        await self.threads.append_inbox(
            target, target.event_id, _inbox_payload(task, target)
        )
        while await self.threads.try_begin_turn(target):
            try:
                while (item := await self.threads.next_inbox(target)) is not None:
                    turn_task, turn_target = _task_target_from_payload(item.payload)
                    await self.run(turn_task, turn_target)
            finally:
                await self.threads.end_turn(target)

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
            return "⛔ approvals are not available"
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
        for session in await self.sessions.recent(limit=1000):
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
        return repaired

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
            final_text = f"🚫 Denied by {approver}."
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

    async def _on_workflow_terminal(self, instance) -> None:
        approval_id = _approval_id_for_instance(instance)
        if not approval_id:
            return
        session = await self.sessions.get_by_approval(approval_id)
        if session is None:
            return
        self._progress_seen.pop(session.id, None)
        await self._deliver_terminal_approval(session)

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
        """Deliver a waiting session whose approved workflow has finished."""
        tools = getattr(self.runtime, "tools", None)
        engine = getattr(tools, "engine", None) or getattr(self.runtime, "engine", None)
        if tools is None or engine is None:
            return False
        from openloop.surfaces.approvals import resolution_message
        from openloop.tools.gateway import _workflow_invocation

        for approval_id in session.approval_ids:
            request = await tools.approvals.get(approval_id)
            if request is None or request.status != "approved":
                continue
            tool = getattr(tools, "_tools", {}).get(request.tool)
            if not getattr(tool, "workflow", None):
                continue
            instance_id = request.args.get("job_id") or request.id
            instance = await engine.store.get(instance_id)
            if instance is None or instance.status not in _WORKFLOW_TERMINAL:
                continue
            inv = _workflow_invocation(instance)
            approver = request.decided_by or "an approver"
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
        # than blindly re-posting and duplicating the answer.
        if session.status == "completed":
            await self._post_final(
                session, session.result_summary or "(no response)", recover=True
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
        self, session: SurfaceSession, text: str, *, recover: bool = False
    ) -> None:
        if session.final_message_id is not None:
            return  # already delivered — never post a second final answer
        mid = await self.delivery.post_final(
            session.target,
            text,
            key=self._delivery_key(session, "final"),
            recover=recover,
        )
        session.final_message_id = mid
        await self.sessions.upsert(session)
        # Post-delivery, commit the turn to the thread's delivered transcript so a
        # later turn replays it as real conversation. Idempotent on the session id,
        # so a redelivery/reconcile never double-appends; only after the answer
        # actually reached the thread (final_message_id recorded above).
        await self._record_transcript(session, text)

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
