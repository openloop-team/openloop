"""Continuing a bound workspace task from a later reply in its thread.

A continuation is a **new turn of an existing task**, never a new task: the
task id, its branch and pull request, the approval that authorized it, and the
agent its spend attributes to are all read back from the durable
:class:`~openloop.tasks.binding.ThreadTask` and re-entered unchanged. Only the
per-turn facts change — the new request text, the session delivering this turn,
and the workflow instance driving it.

Two rules are decided here, not by whichever surface happens to deliver the
reply:

**Eligibility.** A reply continues the task only if the binding is open (not
closed, not claimed by another turn), the profile can actually be continued, and
the reply's author is the human who started the task. Anyone else's reply falls
through to an ordinary turn, which can still delegate work — through the
approval gate, as a task of its own. Fail-closed by construction: an unknown
initiator continues nothing.

**Re-gating.** The code profile's gate is ``Gate.START`` — the boundary sits at
the *task's* start, not at every turn. A continuation of an already-approved
task therefore carries that approval forward rather than raising a second card
for the same task, and the authorization envelope (approval id, agent identity,
initiating human) is preserved verbatim so spend and audit still trace to what
was authorized. A profile that needs per-effect gating expresses that as
``Gate.EFFECT``, which this path leaves untouched.
"""

from __future__ import annotations

from collections.abc import Callable

from openloop.tasks.binding import OPEN, ThreadTask
from openloop.tasks.contract import WorkspaceTask

# Marks a workflow instance as a continuation turn: its steps skip the start
# gate (already passed for this task) and it must never mint a new identity.
CONTINUATION = "continuation"

# How many earlier requests are replayed to the workspace agent as the task's
# own history. The thread transcript is the conversation; this is the task's.
REQUEST_HISTORY_LIMIT = 10


class ContinuationUnavailable(Exception):
    """This task cannot be continued from its durable record (yet, or ever)."""


def continuation_instance_id(task_id: str, session_id: str) -> str:
    """Deterministic instance id for one continuation turn.

    Derived from the task plus the turn's session, so a retried delivery of the
    same reply re-addresses the same instance instead of starting a second run
    of the same work.
    """
    return f"{task_id}:cont:{session_id}"


def may_continue(record: ThreadTask | None, *, user: str | None) -> bool:
    """Whether ``user``'s reply is eligible to continue ``record``'s task."""
    if record is None or record.status != OPEN:
        return False
    if not record.requested_by or not user:
        # An unknown initiator (or an unattributed reply) continues nothing —
        # the ordinary gated path still handles the request.
        return False
    if record.requested_by != user:
        return False
    return continuable(record)


def continuable(record: ThreadTask) -> bool:
    """Whether the durable record holds enough to re-enter the task cold."""
    builder = _BUILDERS.get(record.profile)
    if builder is None:
        return False
    try:
        _rebuild(record)
    except ContinuationUnavailable:
        return False
    return True


def continuation_state(
    record: ThreadTask, *, request: str, session_id: str
) -> dict:
    """The workflow initial state for one continuation turn of ``record``'s task.

    The returned dict is a full :class:`WorkspaceTask` (nested layout) plus the
    continuation marker — the same shape a first turn checkpoints, so the
    workflow reads it through exactly one contract.
    """
    task = _rebuild(record)
    builder = _BUILDERS[record.profile]
    builder(task, request)
    # Per-turn facts: this turn's delivery session, and a fresh step list.
    # Identity, authorization, and profile state stay as the durable record
    # left them.
    task.session_id = session_id
    task.progress = None
    task.completed_steps = []
    state = task.to_dict()
    state[CONTINUATION] = True
    state["turn"] = record.turns
    # The args-contract version the task was created under rides along, so a
    # consumer can still refuse a record written under an older contract.
    schema = (record.state or {}).get("args_schema")
    if schema is not None:
        state["args_schema"] = schema
    return state


def _rebuild(record: ThreadTask) -> WorkspaceTask:
    """Rehydrate the durable task, refusing a record that can't be re-entered."""
    builder = _BUILDERS.get(record.profile)
    if builder is None:
        raise ContinuationUnavailable(
            f"profile {record.profile!r} has no continuation"
        )
    try:
        task = WorkspaceTask.from_dict(record.state or {})
    except Exception as exc:  # noqa: BLE001 — a malformed row is never continued
        raise ContinuationUnavailable(f"unreadable task state: {exc}") from exc
    if not task.task_id:
        raise ContinuationUnavailable("the durable task has no identity")
    task.task_id = record.task_id
    _check(record.profile, task)
    return task


def _check_code(task: WorkspaceTask) -> None:
    code = task.profile_state.get("code")
    if not isinstance(code, dict):
        raise ContinuationUnavailable("the code task has no profile state")
    worker = code.get("worker_state")
    if not isinstance(worker, dict) or not worker.get("branch"):
        # The first turn has not produced branch identity yet (it is still
        # parked at its start gate, or it failed before provisioning). There is
        # nothing to build on, so a reply is an ordinary request.
        raise ContinuationUnavailable("the code task has no branch identity yet")
    if not code.get("repo"):
        raise ContinuationUnavailable("the code task has no repository")


def _continue_code(task: WorkspaceTask, request: str) -> None:
    """Re-enter a code task: same repo, same branch, same PR, new request.

    The continuation's base is the task's own branch once it has been pushed,
    so the new attempt starts from the work already on the pull request instead
    of from the original base — the diff grows, the PR head does not move.
    """
    code = task.profile_state["code"]
    worker = dict(code.get("worker_state") or {})
    branch = worker["branch"]
    pushed = "push" in (worker.get("completed_steps") or [])
    base = branch if pushed else (worker.get("base") or code.get("base") or "main")

    history = list(code.get("requests") or [])
    if not history and code.get("instruction"):
        history = [code["instruction"]]
    history.append(request)
    instruction = _code_instruction(
        history, repo=code.get("repo", ""), branch=branch, pushed=pushed
    )

    code["requests"] = history[-REQUEST_HISTORY_LIMIT:]
    code["instruction"] = instruction
    code["base"] = base
    # A fresh attempt over the same identity: progress, generated title/body and
    # any parked worker segment belong to the turn that produced them, never to
    # the next one. Branch, repo and job id are identity and stay put.
    worker.update(
        {
            "instruction": instruction,
            "base": base,
            "completed_steps": [],
            "title": None,
            "body": None,
            "openhands_resume": None,
        }
    )
    code["worker_state"] = worker


def _code_instruction(
    history: list[str], *, repo: str, branch: str, pushed: bool
) -> str:
    """Compose the workspace agent's brief for a continuation turn."""
    earlier = history[:-1][-REQUEST_HISTORY_LIMIT:]
    current = history[-1]
    lines = [
        f"You are continuing work you already started on branch {branch}"
        + (f" of {repo}" if repo else "")
        + ".",
    ]
    if pushed:
        lines.append(
            "That branch already carries your earlier commits and an open "
            "pull request. Build on what is there — do not start over, and do "
            "not undo earlier work unless the new request asks you to."
        )
    if earlier:
        lines.append("")
        lines.append("Earlier requests in this task:")
        lines.extend(f"{i}. {text}" for i, text in enumerate(earlier, start=1))
    lines.append("")
    lines.append("The new request for this task:")
    lines.append(current)
    return "\n".join(lines)


_BUILDERS: dict[str, Callable[[WorkspaceTask, str], None]] = {
    "code": _continue_code,
}

_CHECKS: dict[str, Callable[[WorkspaceTask], None]] = {
    "code": _check_code,
}


def _check(profile: str, task: WorkspaceTask) -> None:
    check = _CHECKS.get(profile)
    if check is not None:
        check(task)
