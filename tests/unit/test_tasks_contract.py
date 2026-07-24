from pydantic import BaseModel

from openloop.tasks.contract import Gate, TaskProfile, WorkspaceTask, profile_for


class _Args(BaseModel):
    repo: str


def test_gate_values_are_the_three_declared():
    assert {g.value for g in Gate} == {"start", "effect", "none"}


def test_profile_declares_its_gate_and_capabilities():
    p = TaskProfile(
        name="code",
        entry_action="code:write",
        args_model=_Args,
        gate=Gate.START,
        capabilities=frozenset({"repo:write"}),
    )
    assert p.gate is Gate.START
    assert p.capabilities == frozenset({"repo:write"})
    assert p.args_model is _Args


def test_workspace_task_roundtrips_and_holds_no_outcome_fields():
    t = WorkspaceTask(
        task_id="abc123",
        profile="investigate",
        entry_action="investigate:read",
        agent="dev-platform",
        session_id="s1",
        profile_state={"repo": "a/b", "question": "why"},
    )
    again = WorkspaceTask.from_dict(t.to_dict())
    assert again == t
    # PR-only fields live in profile_state, never on the core.
    assert not hasattr(again, "branch")
    assert again.profile_state["repo"] == "a/b"


def test_profile_registry_matches_legacy_gate_decisions():
    assert profile_for("code:write").gate is Gate.START
    assert profile_for("investigate:read").gate is Gate.NONE
    assert profile_for("bogus:perm") is None
