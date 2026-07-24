"""Unit tests for the engine's optional state-driven step resolver."""

from openloop.workflows.engine import Step, Workflow
from openloop.workflows.store import WorkflowInstance


def test_fixed_steps_workflow_is_unaffected():
    wf = Workflow("w", [Step("a"), Step("b")])
    inst = WorkflowInstance(id="i", workflow="w")
    assert [s.name for s in wf.steps_for(inst)] == ["a", "b"]


def test_resolver_workflow_derives_steps_from_instance_state():
    def resolve(state):
        return [Step("x")] if state.get("profile") == "p" else [Step("y")]

    wf = Workflow("w", [], steps_resolver=resolve)
    inst = WorkflowInstance(id="i", workflow="w", state={"profile": "p"})
    assert [s.name for s in wf.steps_for(inst)] == ["x"]
