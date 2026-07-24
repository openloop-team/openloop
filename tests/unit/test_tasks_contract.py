import ast
from pathlib import Path

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


def test_from_dict_rehydrates_old_flat_coding_worker_layout():
    old = {
        "job_id": "j1", "repo": "a/b", "instruction": "x", "base": "main",
        "agent": "dev", "agent_id": "id1", "approval_id": "ap1",
        "session_id": "s1", "warm_key": "w1",
        "worker_state": {"job_id": "j1", "repo": "a/b", "branch": "openloop/job-j1"},
    }
    t = WorkspaceTask.from_dict(old)
    assert t.task_id == "j1"
    assert t.profile == "code"
    assert t.agent == "dev" and t.agent_id == "id1"
    assert t.profile_state["code"]["worker_state"]["branch"] == "openloop/job-j1"
    # The old flat layout's budget lived inside worker_state (carried
    # verbatim above); the compat shim itself has nothing to lift it from,
    # so the core field is left unset rather than guessed at.
    assert t.budget_usd is None
    # New layout round-trips too:
    again = WorkspaceTask.from_dict(t.to_dict())
    assert again == t


def test_workspace_task_budget_usd_roundtrips():
    t = WorkspaceTask(
        task_id="abc123",
        profile="code",
        entry_action="code:write",
        budget_usd=12.5,
    )
    assert t.to_dict()["budget_usd"] == 12.5
    again = WorkspaceTask.from_dict(t.to_dict())
    assert again.budget_usd == 12.5
    assert again == t
    # Default is unset, not silently zero.
    assert WorkspaceTask(
        task_id="x", profile="code", entry_action="code:write"
    ).budget_usd is None


def test_from_dict_rehydrates_old_direct_worker_checkpoint_layout():
    old_worker = {
        "job_id": "j2",
        "repo": "a/b",
        "instruction": "x",
        "base": "develop",
        "branch": "openloop/job-j2",
        "completed_steps": ["clone", "branch"],
        "agent": "dev",
        "agent_id": "id2",
        "budget_usd": 3.5,
        "openhands_resume": None,
    }

    task = WorkspaceTask.from_dict(old_worker)

    assert task.task_id == "j2"
    assert task.agent == "dev"
    assert task.agent_id == "id2"
    assert task.budget_usd == 3.5
    assert task.completed_steps == ["clone", "branch"]
    assert task.profile_state["code"]["worker_state"] == old_worker


def test_contract_types_are_consumed_by_runtime():
    """Guard the inverse of the gap that motivated contract convergence.

    The contract must remain a runtime dependency, not drift back into a set
    of types exercised only by tests: the connector constructs WorkspaceTask,
    the durable workflow rehydrates it, and the connector's gate consults the
    TaskProfile registry.
    """
    src = Path(__file__).parents[2] / "src" / "openloop"
    connector_tree = ast.parse(
        (src / "tools" / "coding_worker.py").read_text(encoding="utf-8")
    )
    workflow_tree = ast.parse(
        (src / "workflows" / "coding_worker.py").read_text(encoding="utf-8")
    )

    connector_calls = {
        node.func.id
        for node in ast.walk(connector_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    workflow_calls = {
        (node.func.value.id, node.func.attr)
        for node in ast.walk(workflow_tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        )
    }

    assert "WorkspaceTask" in connector_calls
    assert "profile_for" in connector_calls
    assert ("WorkspaceTask", "from_dict") in workflow_calls
