"""Deterministic runtime integration flow with no external services.

This is the cheapest full-stack runtime path: real runtime, memory, usage, tool
gateway, approval store, and GitHub connector; fake model, fake embedder, and
fake GitHub client.
It should run in the normal suite and catch broken orchestration contracts
without network, provider credentials, or Docker.
"""

import pytest

from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.memory import InMemoryStore, MemoryRecord, scope_key_for
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore, budget_scope_key
from openloop.testing import EXAMPLE_AGENT, FakeEmbedder, FakeGitHub, ScriptedGateway
from openloop.testing import tool_call_response

pytestmark = pytest.mark.integration


async def test_runtime_memory_tool_approval_usage_flow():
    agent = load_agent(EXAMPLE_AGENT)
    memory = InMemoryStore()
    usage = InMemoryUsageStore()
    approvals = InMemoryApprovalStore()
    embedder = FakeEmbedder()

    scope = scope_key_for(agent, "#dev-platform")
    seed_vec = (await embedder.embed(["Use Redis Streams for ingestion v1."]))[0]
    await memory.remember(
        MemoryRecord(
            scope_key=scope,
            text="Use Redis Streams for ingestion v1.",
            kind="decision",
            metadata={"source": "seed"},
            embedding=seed_vec,
        )
    )

    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)], approvals=approvals)

    read_response = tool_call_response(
        "m",
        [
            (
                "read-1",
                "github_issues_read",
                {"repo": "acme/ingestion", "number": 17},
            )
        ],
    )
    read_response.prompt_tokens = 10
    read_response.completion_tokens = 3
    read_response.cost_usd = 0.01

    write_response = tool_call_response(
        "m",
        [
            (
                "write-1",
                "github_issues_write",
                {
                    "repo": "acme/ingestion",
                    "title": "Track: Redis Streams for v1",
                    "body": "Follow up on the channel decision.",
                },
            )
        ],
    )
    write_response.prompt_tokens = 12
    write_response.completion_tokens = 4
    write_response.cost_usd = 0.02

    runtime = Runtime(
        agent,
        gateway=ScriptedGateway([read_response, write_response]),
        memory=memory,
        embedder=embedder,
        usage=usage,
        tools=tools,
    )

    result = await runtime.handle(
        Task(
            text="Check issue 17 and open a follow-up for the ingestion decision.",
            surface="slack",
            channel="#dev-platform",
            user="U_requester",
        )
    )

    assert result.model == "approval-gate"
    assert result.approval_ids
    approval_id = result.approval_ids[0]
    assert "approval required" in result.text.lower()
    assert github.created == []

    first_call_messages = runtime.gateway.calls[0]["messages"]
    system_text = " ".join(
        m["content"] for m in first_call_messages if m["role"] == "system"
    )
    assert "Redis Streams" in system_text

    second_call_messages = runtime.gateway.calls[1]["messages"]
    tool_text = " ".join(
        m["content"] for m in second_call_messages if m["role"] == "tool"
    )
    assert "read issue #17 in acme/ingestion" in tool_text

    pending = await approvals.pending(agent="dev-platform")
    assert [p.id for p in pending] == [approval_id]
    assert pending[0].requested_by == "U_requester"
    assert pending[0].args["title"] == "Track: Redis Streams for v1"

    resolved = await tools.resolve(approval_id, "@priya", approve=True)
    assert resolved.status == "executed"
    assert github.created == [
        {
            "number": 1,
            "repo": "acme/ingestion",
            "title": "Track: Redis Streams for v1",
        }
    ]

    stored = await approvals.get(approval_id)
    assert stored is not None
    assert stored.status == "approved"
    assert stored.decided_by == "@priya"

    records = await usage.recent(limit=10)
    assert len(records) == 1
    assert records[0].surface == "slack"
    assert records[0].channel == "#dev-platform"
    assert records[0].user == "U_requester"
    assert records[0].prompt_tokens == 22
    assert records[0].completion_tokens == 7
    assert records[0].cost_usd == pytest.approx(0.03)
    assert await usage.monthly_total(budget_scope_key(agent)) == pytest.approx(0.03)

    recalled = await memory.recall(scope, seed_vec, limit=5)
    texts = [r.text for r in recalled]
    assert "Use Redis Streams for ingestion v1." in texts
    assert any("open a follow-up" in t for t in texts)
