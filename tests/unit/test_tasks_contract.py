from pydantic import BaseModel

from openloop.tasks.contract import Gate, TaskProfile


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
