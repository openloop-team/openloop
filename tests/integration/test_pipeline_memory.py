"""Integration: the runtime recalls memory into context and remembers tasks."""

from pathlib import Path
from openloop.agents import load_agent
from openloop.memory import InMemoryStore, MemoryRecord, scope_key_for
from openloop.runtime import Runtime, Task
from openloop.testing import FakeEmbedder, FakeGateway

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _agent():
    return load_agent(AGENT_YAML)


async def test_recalled_memory_is_injected_into_context():
    agent = _agent()
    store = InMemoryStore()
    scope = scope_key_for(agent, "#dev-platform")
    await store.remember(
        MemoryRecord(scope_key=scope, text="Use Redis Streams for ingestion v1.")
    )

    gateway = FakeGateway()
    runtime = Runtime(agent, gateway=gateway, memory=store, embedder=None)
    await runtime.handle(
        Task(text="what did we pick for ingestion?", surface="slack",
             channel="#dev-platform")
    )

    system_text = " ".join(
        m["content"] for m in gateway.last_messages if m["role"] == "system"
    )
    assert "Redis Streams" in system_text


async def test_handle_remembers_the_task():
    agent = _agent()
    store = InMemoryStore()
    runtime = Runtime(agent, gateway=FakeGateway(), memory=store,
                      embedder=FakeEmbedder())

    await runtime.handle(
        Task(text="capture this decision", surface="slack",
             channel="#dev-platform", user="U1")
    )

    scope = scope_key_for(agent, "#dev-platform")
    recalled = await store.recall(scope)
    assert len(recalled) == 1
    assert recalled[0].text == "capture this decision"
    assert recalled[0].metadata["user"] == "U1"
    assert recalled[0].embedding is not None  # embedded once, reused


async def test_remember_disabled_stores_nothing():
    agent = _agent()
    store = InMemoryStore()
    runtime = Runtime(agent, gateway=FakeGateway(), memory=store, remember=False)
    await runtime.handle(Task(text="hi", surface="slack", channel="#x"))
    assert await store.recall(scope_key_for(agent, "#x")) == []


async def test_memory_does_not_leak_across_channels():
    agent = _agent()
    store = InMemoryStore()
    runtime = Runtime(agent, gateway=FakeGateway(), memory=store)

    await runtime.handle(Task(text="team A secret", surface="slack", channel="#a"))
    gateway_b = FakeGateway()
    runtime_b = Runtime(agent, gateway=gateway_b, memory=store)
    await runtime_b.handle(Task(text="hello", surface="slack", channel="#b"))

    system_text = " ".join(
        m["content"] for m in gateway_b.last_messages if m["role"] == "system"
    )
    assert "team A secret" not in system_text
