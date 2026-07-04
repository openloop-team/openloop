"""Tests for the in-memory store, scope isolation, and recall ranking."""

from pathlib import Path
import pytest

from openloop.agents import load_agent
from openloop.memory import InMemoryStore, MemoryRecord, scope_key_for
from openloop.memory.store import cosine_similarity

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


async def test_remember_and_recall_recency():
    store = InMemoryStore()
    for i in range(3):
        await store.remember(MemoryRecord(scope_key="s", text=f"m{i}"))
    recalled = await store.recall("s", limit=2)
    # Most recent first.
    assert [r.text for r in recalled] == ["m2", "m1"]


async def test_recall_is_scope_isolated():
    store = InMemoryStore()
    await store.remember(MemoryRecord(scope_key="team-a", text="secret a"))
    await store.remember(MemoryRecord(scope_key="team-b", text="secret b"))
    a = await store.recall("team-a")
    assert [r.text for r in a] == ["secret a"]
    assert await store.recall("team-c") == []


async def test_semantic_recall_orders_by_similarity():
    store = InMemoryStore()
    await store.remember(
        MemoryRecord(scope_key="s", text="far", embedding=[0.0, 1.0])
    )
    await store.remember(
        MemoryRecord(scope_key="s", text="near", embedding=[1.0, 0.0])
    )
    recalled = await store.recall("s", query_embedding=[0.9, 0.1], limit=2)
    assert recalled[0].text == "near"


def test_cosine_similarity_edge_cases():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_scope_key_respects_declared_scope():
    agent = load_agent(AGENT_YAML)  # memory.scope == "channel"
    k1 = scope_key_for(agent, "#dev-platform")
    k2 = scope_key_for(agent, "#other")
    assert k1 != k2
    assert "channel:#dev-platform" in k1
