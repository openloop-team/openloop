"""Integration: Runtime.handle running as a durable `agent_task` workflow.

Phase C consumer #2 — every task runs through the explicitly supplied workflow
engine (prepare → run → persist), persisting turn state and writing usage/memory
idempotently.
"""

import inspect
from pathlib import Path
from openloop.agents import load_agent
from openloop.memory import InMemoryStore, scope_key_for
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore, UsageRecord, budget_scope_key
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine, WorkflowInstance
from openloop.testing import (
    FakeGitHub,
    ScriptedGateway,
    tool_call_response,
)

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


class CountingGateway:
    """Model gateway that records how many times it was called."""

    def __init__(self):
        self.calls = 0

    async def complete(self, model, messages, **kwargs):
        self.calls += 1
        return ModelResponse(text="hi", model="m")


def _agent():
    return load_agent(AGENT_YAML)


def _task(text="hi"):
    return Task(text=text, surface="slack", channel="#dev-platform", user="U1")


def _runtime(model_gateway, *, tools=None, usage=None, memory=None):
    store = InMemoryWorkflowStore()
    engine = WorkflowEngine(store)
    rt = Runtime(
        _agent(),
        gateway=model_gateway,
        tools=tools,
        usage=usage or InMemoryUsageStore(),
        memory=memory or InMemoryStore(),
        engine=engine,
    )
    return rt, engine, store


def test_runtime_requires_a_keyword_only_engine():
    parameter = inspect.signature(Runtime).parameters["engine"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


async def test_plain_chat_runs_through_the_workflow():
    rt, engine, store = _runtime(ScriptedGateway([ModelResponse(text="hello", model="m")]))
    res = await rt.handle(_task())

    assert res.text == "hello"
    inst = (await store.recent())[0]
    assert inst.workflow == rt.workflow_name
    assert inst.status == "completed"
    assert inst.completed_steps == ["prepare", "run", "persist"]
    # Turn state persisted: messages + received model output.
    assert inst.state["final_text"] == "hello"
    # M0a: the log is COMPLETE — the final assistant answer is appended (it used to
    # be dropped), so a resume/next turn sees the real answer, not a summary.
    assert inst.state["messages"][-1] == {"role": "assistant", "content": "hello"}
    assert inst.state["messages"][-2] == {"role": "user", "content": "hi"}


async def test_write_tool_call_held_for_approval_via_workflow():
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    rt, engine, store = _runtime(
        ScriptedGateway([
            tool_call_response("m", [("c1", "github_issues_write",
                                      {"repo": "acme/x", "title": "T"})]),
        ]),
        tools=tools, usage=usage,
    )

    res = await rt.handle(_task("open an issue"))

    assert res.model == "approval-gate"
    assert res.approval_ids
    assert github.created == []  # not executed yet
    inst = (await store.recent())[0]
    assert inst.status == "completed"
    # Approvals are part of the persisted turn state.
    assert inst.state["approval_ids"] == res.approval_ids
    assert len(usage.records) == 1  # usage written exactly once


async def test_budget_block_through_workflow():
    usage = InMemoryUsageStore()
    await usage.record(UsageRecord(
        scope_key=budget_scope_key(_agent()), workspace="acme", agent="dev-platform",
        model="m", cost_usd=1000.0, outcome="ok",
    ))
    rt, engine, store = _runtime(
        ScriptedGateway([ModelResponse(text="x", model="m")]), usage=usage
    )

    res = await rt.handle(_task())

    assert res.model == "budget-guard"
    inst = (await store.recent())[0]
    assert inst.state.get("blocked") is True
    assert any(r.outcome == "blocked" for r in usage.records)


async def test_turn_is_remembered_once():
    memory = InMemoryStore()
    rt, engine, store = _runtime(
        ScriptedGateway([ModelResponse(text="ok", model="m")]), memory=memory
    )
    await rt.handle(_task("remember this"))

    inst = (await store.recent())[0]
    assert inst.state.get("remembered") is True
    # The user's message was remembered for the channel scope.
    from openloop.memory import scope_key_for
    recalled = await memory.recall(scope_key_for(_agent(), "#dev-platform"), None, limit=5)
    assert any("remember this" in r.text for r in recalled)


async def test_crash_mid_run_resumes_without_replaying_committed_answer():
    # M0a: `run` is resumable. A turn that committed its final answer to the log
    # but crashed before the run step was marked complete RESUMES (no longer
    # abandoned) and reconstructs the answer from the log — without re-calling the
    # model. This is the crash window fixed by the terminal check.
    usage = InMemoryUsageStore()
    gateway = CountingGateway()  # any call here would be a replay bug
    rt, engine, store = _runtime(gateway, usage=usage)
    await store.create(WorkflowInstance(
        id="midrun", workflow=rt.workflow_name, status="running",
        completed_steps=["prepare"],  # run not yet complete
        state={
            "task": {"text": "hi", "surface": "slack", "channel": "#dev-platform",
                     "user": "U1"},
            "model": "m", "scope": scope_key_for(_agent(), "#dev-platform"),
            "query_embedding": None,
            # The final answer was committed to the log at (B); final_text was not
            # yet set when the crash landed.
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer"},
            ],
            "usage_total": {"model": "m", "prompt_tokens": 3,
                            "completion_tokens": 2, "cost_usd": 0.01},
        },
    ))

    resumed = await engine.resume_incomplete()

    assert "midrun" in resumed  # resumed, not abandoned
    inst = await store.get("midrun")
    assert inst.status == "completed"
    assert inst.state["final_text"] == "answer"  # reconstructed from the log
    assert gateway.calls == 0  # NO model replay
    assert len(usage.records) == 1  # usage recorded once by the persist tail


async def test_resume_after_committed_final_round_at_budget_does_not_recall():
    # The specific corner: the final answer was produced on the last permitted round
    # (rounds_used == MAX_TOOL_ITERS) and committed, but final_text wasn't persisted.
    # The terminal check must fire BEFORE the budget guard so the answer is
    # reconstructed, not overwritten by the "couldn't finish" fallback.
    from openloop.runtime.pipeline import MAX_TOOL_ITERS
    gateway = CountingGateway()
    rt, engine, store = _runtime(gateway)
    msgs = [{"role": "user", "content": "hi"}]
    # MAX_TOOL_ITERS assistant rounds already in the log; the last is a final answer.
    for i in range(MAX_TOOL_ITERS - 1):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "x", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    msgs.append({"role": "assistant", "content": "the answer"})  # MAX-th round, final

    await store.create(WorkflowInstance(
        id="atbudget", workflow=rt.workflow_name, status="running",
        completed_steps=["prepare"],
        state={
            "task": {"text": "hi", "surface": "slack", "channel": "#dev-platform",
                     "user": "U1"},
            "model": "m", "scope": scope_key_for(_agent(), "#dev-platform"),
            "query_embedding": None, "messages": msgs,
        },
    ))

    await engine.resume_incomplete()

    inst = await store.get("atbudget")
    assert inst.status == "completed"
    assert inst.state["final_text"] == "the answer"  # reconstructed, not the fallback
    assert gateway.calls == 0


async def test_followup_turn_calls_the_model_not_echo():
    # THE cursor-scoping bug (harmful): a follow-up turn's messages are prefixed
    # with prior delivered history ending in the previous turn's assistant answer.
    # The replay cursor must be TURN-SCOPED — otherwise the terminal check sees the
    # old answer at turn entry and echoes it without ever calling the model.
    gateway = ScriptedGateway([ModelResponse(text="fresh answer", model="m")])
    rt, engine, store = _runtime(gateway)
    task = Task(
        text="and then?", surface="slack", channel="#dev-platform", user="U1",
        history=[{"role": "user", "content": "q1"},
                 {"role": "assistant", "content": "old answer"}],
    )

    res = await rt.handle(task)

    assert len(gateway.calls) == 1  # the model WAS called — no echo
    assert res.text == "fresh answer"
    assert res.text != "old answer"


class _CountingGitHub(FakeGitHub):
    """FakeGitHub that counts issue reads, to prove no re-execution on resume."""

    def __init__(self):
        super().__init__()
        self.reads = 0

    async def get_issue(self, repo, number):
        self.reads += 1
        return await super().get_issue(repo, number)


async def test_resume_skips_already_executed_tool_call():
    # A tool round with one call committed and one not: resume executes ONLY the
    # unresolved call — the committed one is never re-invoked (no double side effect).
    github = _CountingGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    gateway = ScriptedGateway([ModelResponse(text="done", model="m")])
    rt, engine, store = _runtime(gateway, tools=tools)
    await store.create(WorkflowInstance(
        id="toolresume", workflow=rt.workflow_name, status="running",
        completed_steps=["prepare"],
        state={
            "task": {"text": "read issues", "surface": "slack",
                     "channel": "#dev-platform", "user": "U1"},
            "model": "m", "scope": scope_key_for(_agent(), "#dev-platform"),
            "query_embedding": None,
            "messages": [
                {"role": "user", "content": "read issues"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {
                        "name": "github_issues_read",
                        "arguments": '{"repo": "acme/x", "number": 1}'}},
                    {"id": "c2", "type": "function", "function": {
                        "name": "github_issues_read",
                        "arguments": '{"repo": "acme/x", "number": 2}'}},
                ]},
                # c1 already executed (committed); c2 is still unresolved.
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ],
        },
    ))

    await engine.resume_incomplete()

    inst = await store.get("toolresume")
    assert inst.status == "completed"
    assert github.reads == 1  # only c2 ran on resume; c1 NOT re-invoked
    assert inst.state["final_text"] == "done"


async def test_crash_after_run_resumes_idempotent_persist_tail():
    # The recoverable case: run already completed (model output persisted), only
    # the idempotent persist tail remains — resume it instead of abandoning.
    usage = InMemoryUsageStore()
    gateway = CountingGateway()
    rt, engine, store = _runtime(gateway, usage=usage)
    await store.create(WorkflowInstance(
        id="midpersist", workflow=rt.workflow_name, status="running",
        completed_steps=["prepare", "run"],
        state={
            "task": {"text": "hi", "surface": "slack", "channel": "#dev-platform",
                     "user": "U1"},
            "model": "m", "scope": scope_key_for(_agent(), "#dev-platform"),
            "messages": [], "query_embedding": None, "final_text": "answer",
            "accounted": {"model": "m", "prompt_tokens": 3,
                          "completion_tokens": 2, "cost_usd": 0.01},
            "approval_ids": [],
        },
    ))

    resumed = await engine.resume_incomplete()

    assert "midpersist" in resumed
    assert (await store.get("midpersist")).status == "completed"
    assert len(usage.records) == 1  # the persist tail wrote usage on resume
    assert gateway.calls == 0  # no model replay
