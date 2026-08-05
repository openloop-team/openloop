"""Unit tests for the workspace_task workflow's profile-dispatched steps."""

from openloop.testing import FakeGitHub, FakeWorkerOrchestrator
from openloop.workflows.coding_worker import (
    WORKFLOW_NAME,
    build_workspace_task_workflow,
)
from openloop.workflows.store import WorkflowInstance


def test_workflow_name_is_workspace_task():
    assert WORKFLOW_NAME == "workspace_task"


def test_code_profile_keeps_wait_node_and_pr_steps():
    wf = build_workspace_task_workflow(FakeWorkerOrchestrator(), FakeGitHub())
    inst = WorkflowInstance(id="i", workflow="workspace_task", state={"profile": "code"})
    steps = wf.steps_for(inst)
    assert [s.name for s in steps] == ["await_approval", "run_worker", "open_pr"]
    assert steps[0].wait is True


def test_default_profile_is_code():
    wf = build_workspace_task_workflow(FakeWorkerOrchestrator(), FakeGitHub())
    inst = WorkflowInstance(id="i", workflow="workspace_task", state={})
    assert [s.name for s in wf.steps_for(inst)] == ["await_approval", "run_worker", "open_pr"]
