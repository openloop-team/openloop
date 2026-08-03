"""Unit tests for the durable thread↔task binding and the continuation builder.

Covers the store's claim semantics (a task is driven by one turn at a time, and
a closed task never returns), and the profile-neutral continuation rules:
eligibility, the authorization envelope carried forward, and the code profile's
branch/base handling.
"""

import pytest

from openloop.tasks import (
    BUSY,
    CLOSED,
    OPEN,
    ContinuationUnavailable,
    InMemoryThreadTaskStore,
    ThreadTask,
    WorkspaceTask,
    continuable,
    continuation_instance_id,
    continuation_state,
    may_continue,
)
from openloop.tasks.continuation import CONTINUATION

pytestmark = pytest.mark.unit

SCOPE = "slack\x1facme\x1fdev-platform\x1fC1\x1f100.1"


def _task_state(*, pushed=True, branch="openloop/job-j1", instruction="add retries"):
    """A serialized WorkspaceTask as a first turn leaves it behind."""
    steps = ["clone", "branch", "edit", "commit", "push"] if pushed else ["clone"]
    task = WorkspaceTask(
        task_id="j1",
        profile="code",
        entry_action="code:write",
        agent="dev-platform",
        agent_id="45006d4ce5c64d2c96ed1fe3277d7347",
        approval_id="ap-1",
        requester_id="maciag.artur",
        session_id="s1",
        warm_key=SCOPE,
        completed_steps=list(steps),
        profile_state={
            "code": {
                "repo": "acme/x",
                "instruction": instruction,
                "base": "main",
                "worker_state": {
                    "job_id": "j1",
                    "repo": "acme/x",
                    "instruction": instruction,
                    "base": "main",
                    "branch": branch,
                    "completed_steps": list(steps),
                    "title": "Add retries",
                    "body": "generated",
                    "openhands_resume": None,
                },
            }
        },
    )
    return task.to_dict()


def _record(**overrides) -> ThreadTask:
    fields = dict(
        task_id="j1",
        scope_key=SCOPE,
        profile="code",
        entry_action="code:write",
        agent="dev-platform",
        agent_id="45006d4ce5c64d2c96ed1fe3277d7347",
        approval_id="ap-1",
        requested_by="U1",
        instance_id="j1",
        state=_task_state(),
    )
    fields.update(overrides)
    return ThreadTask(**fields)


# --- store ---------------------------------------------------------------

async def test_bind_is_idempotent_on_task_id():
    store = InMemoryThreadTaskStore()
    first = await store.bind(_record())
    again = await store.bind(_record(scope_key="other-thread", state={}))

    assert again.scope_key == first.scope_key  # the birth record is written once
    assert (await store.get("j1")).state == first.state


async def test_bound_returns_the_threads_open_task_and_never_a_closed_one():
    store = InMemoryThreadTaskStore()
    await store.bind(_record())

    assert (await store.bound(SCOPE)).task_id == "j1"
    assert await store.bound("another-thread") is None

    await store.retire("j1", "approval denied")
    assert await store.bound(SCOPE) is None
    assert (await store.get("j1")).status == CLOSED


async def test_claim_is_exclusive_and_release_returns_the_task():
    store = InMemoryThreadTaskStore()
    await store.bind(_record())

    claimed = await store.claim("j1", instance_id="j1:cont:s2", session_id="s2")
    assert claimed is not None
    assert claimed.status == BUSY
    assert claimed.instance_id == "j1:cont:s2"
    assert claimed.turns == 2  # the binding is on its second turn

    # A second turn cannot drive the same task concurrently.
    assert await store.claim("j1", instance_id="j1:cont:s3", session_id="s3") is None
    assert [r.task_id for r in await store.claimed()] == ["j1"]

    released = await store.release("j1", state={"task_id": "j1", "profile": "code"})
    assert released.status == OPEN
    assert released.state["profile"] == "code"
    assert await store.claimed() == []


async def test_a_closed_task_never_reopens_but_still_records_state():
    store = InMemoryThreadTaskStore()
    await store.bind(_record())
    await store.retire("j1", "approval denied")

    await store.release("j1", state={"task_id": "j1", "note": "late write"})

    stored = await store.get("j1")
    assert stored.status == CLOSED
    assert stored.closed_reason == "approval denied"
    assert stored.state["note"] == "late write"
    assert await store.claim("j1", instance_id="x") is None


# --- eligibility ---------------------------------------------------------

async def test_only_the_human_who_started_the_task_continues_it():
    record = _record()
    assert may_continue(record, user="U1") is True
    assert may_continue(record, user="U2") is False
    assert may_continue(record, user=None) is False
    # An unattributed task is never continued by anyone (fail closed).
    assert may_continue(_record(requested_by=None), user="U1") is False
    assert may_continue(None, user="U1") is False


async def test_a_busy_or_closed_binding_is_not_continuable():
    assert may_continue(_record(status=BUSY), user="U1") is False
    assert may_continue(_record(status=CLOSED), user="U1") is False


async def test_a_task_without_branch_identity_is_not_continuable_yet():
    # Parked at its start gate: the durable state is still the approval's args,
    # so there is no branch to build on and a reply is an ordinary request.
    at_gate = _record(
        state={"job_id": "j1", "repo": "acme/x", "instruction": "add retries"}
    )
    assert continuable(at_gate) is False
    assert may_continue(at_gate, user="U1") is False
    with pytest.raises(ContinuationUnavailable):
        continuation_state(at_gate, request="also handle nulls", session_id="s2")


async def test_an_unknown_profile_is_not_continuable():
    assert continuable(_record(profile="investigate")) is False


# --- continuation state --------------------------------------------------

async def test_continuation_preserves_identity_authorization_and_branch():
    record = _record(turns=2)
    state = continuation_state(record, request="also handle nulls", session_id="s2")

    assert state[CONTINUATION] is True
    assert state["turn"] == 2
    # Task identity and the authorization envelope survive verbatim.
    assert state["task_id"] == "j1"
    assert state["approval_id"] == "ap-1"
    assert state["agent"] == "dev-platform"
    assert state["agent_id"] == "45006d4ce5c64d2c96ed1fe3277d7347"
    assert state["requester_id"] == "maciag.artur"
    assert state["warm_key"] == SCOPE
    # Only the per-turn facts move.
    assert state["session_id"] == "s2"
    assert state["completed_steps"] == []

    code = state["profile_state"]["code"]
    worker = code["worker_state"]
    assert worker["branch"] == "openloop/job-j1"  # same branch → same PR head
    assert worker["job_id"] == "j1"
    # The pushed branch is the new base, so the turn builds on the open PR.
    assert worker["base"] == "openloop/job-j1"
    assert code["base"] == "openloop/job-j1"
    # A fresh attempt: no leftover progress, title/body, or parked segment.
    assert worker["completed_steps"] == []
    assert worker["title"] is None
    assert worker["body"] is None
    assert worker["openhands_resume"] is None


async def test_continuation_briefs_the_worker_with_the_tasks_own_history():
    record = _record()
    state = continuation_state(record, request="also handle nulls", session_id="s2")
    code = state["profile_state"]["code"]

    assert code["requests"] == ["add retries", "also handle nulls"]
    instruction = code["instruction"]
    assert "openloop/job-j1" in instruction
    assert "add retries" in instruction  # the earlier request is context
    assert instruction.endswith("also handle nulls")  # the new one is the ask
    assert code["worker_state"]["instruction"] == instruction

    # A third turn keeps accumulating the task's own request history.
    record.state = state
    third = continuation_state(record, request="and add a test", session_id="s3")
    assert third["profile_state"]["code"]["requests"] == [
        "add retries",
        "also handle nulls",
        "and add a test",
    ]


async def test_an_unpushed_task_continues_from_its_original_base():
    # Nothing was pushed, so the branch does not exist on the remote yet and a
    # continuation must clone the original base instead.
    record = _record(state=_task_state(pushed=False))
    state = continuation_state(record, request="try again", session_id="s2")

    assert state["profile_state"]["code"]["worker_state"]["base"] == "main"
    assert state["profile_state"]["code"]["worker_state"]["branch"] == "openloop/job-j1"


async def test_continuation_instance_id_is_deterministic_per_turn():
    assert continuation_instance_id("j1", "s2") == "j1:cont:s2"
    assert continuation_instance_id("j1", "s2") != continuation_instance_id("j1", "s3")
