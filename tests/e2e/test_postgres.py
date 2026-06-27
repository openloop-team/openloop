"""Happy-path end-to-end test against a real Postgres (pgvector).

Validates what the unit tests can't: the actual SQL, asyncpg type handling,
pgvector distance search, and approval persistence. The model, embedder, and
GitHub client are faked (no external credentials), but every store is real.

Runs only when a Postgres is reachable — set OPENLOOP_TEST_DATABASE_URL, or it
falls back to the docker-compose default. Skips cleanly otherwise so the normal
suite stays green without Docker.
"""

import os
import uuid

import pytest

from openloop.agents import load_agent
from openloop.approvals.postgres import PostgresApprovalStore
from openloop.memory.postgres import PostgresMemoryStore
from openloop.memory.store import MemoryRecord, scope_key_for
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import budget_scope_key
from openloop.usage.postgres import PostgresUsageStore
from openloop.testing import (
    EXAMPLE_AGENT,
    FakeEmbedder,
    FakeGitHub,
    ScriptedGateway,
    tool_call_response,
)

DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop_agents",
)

# 26-dim to match FakeEmbedder (the real default is 1536; dim is configurable).
EMBED_DIM = 26

pytestmark = [pytest.mark.e2e, pytest.mark.postgres]


async def _reachable() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(DSN, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def stores():
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")
    # Unique table-free isolation isn't possible (shared tables), so scope keys
    # are made unique per run instead.
    memory = PostgresMemoryStore(DSN, embedding_dim=EMBED_DIM)
    usage = PostgresUsageStore(DSN)
    approvals = PostgresApprovalStore(DSN)
    await memory.setup()
    await usage.setup()
    await approvals.setup()
    try:
        yield memory, usage, approvals
    finally:
        await memory.close()
        await usage.close()
        await approvals.close()


async def test_happy_path_end_to_end(stores):
    memory, usage, approvals = stores
    agent = load_agent(EXAMPLE_AGENT)
    run_id = uuid.uuid4().hex[:8]
    channel = f"#e2e-{run_id}"  # unique scope so the run is isolated
    scope = scope_key_for(agent, channel)

    # Seed a prior decision into channel memory (real pgvector insert).
    embedder = FakeEmbedder()
    seed_vec = (await embedder.embed(["Use Redis Streams for ingestion v1."]))[0]
    await memory.remember(MemoryRecord(
        scope_key=scope, text="Use Redis Streams for ingestion v1.",
        embedding=seed_vec))

    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)], approvals=approvals)

    # The model recalls context, then asks to open a GitHub issue (write action).
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_issues_write",
                                  {"repo": "acme/ingestion",
                                   "title": "Track: Redis Streams for v1"})]),
    ])
    runtime = Runtime(agent, gateway=gateway, memory=memory, embedder=embedder,
                      usage=usage, tools=tools)

    # --- the turn: write action is held for approval ---
    result = await runtime.handle(Task(
        text="open an issue to track the ingestion decision",
        surface="slack", channel=channel, user="U_requester"))

    assert result.model == "approval-gate"
    assert len(result.approval_ids) == 1
    approval_id = result.approval_ids[0]

    # Recall worked against pgvector: the seeded memory reached the model.
    system_text = " ".join(
        m["content"] for m in gateway.calls[0]["messages"]
        if m["role"] == "system")
    assert "Redis Streams" in system_text

    # The approval is persisted as pending in Postgres.
    pending = await approvals.pending(agent="dev-platform")
    assert any(p.id == approval_id for p in pending)
    assert github.created == []  # nothing executed yet

    # --- a human approves; the action executes and persists ---
    inv = await tools.resolve(approval_id, "@priya", approve=True)
    assert inv.status == "executed"
    assert github.created  # the issue was created on approval

    stored = await approvals.get(approval_id)
    assert stored.status == "approved"
    assert stored.decided_by == "@priya"

    # Usage was recorded to the real audit trail, and the turn was remembered.
    spent_records = await usage.recent(limit=200)
    assert any(r.channel == channel for r in spent_records)
    assert await usage.monthly_total(budget_scope_key(agent)) >= 0.0

    recalled = await memory.recall(scope, seed_vec, limit=5)
    texts = [r.text for r in recalled]
    assert "Use Redis Streams for ingestion v1." in texts
    # The requester's message was remembered this turn.
    assert any("open an issue to track" in t for t in texts)
