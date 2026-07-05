"""Deterministic runtime integration flow with no external services.

This is the cheapest full-stack runtime path: real runtime, memory, usage, tool
gateway, approval store, and GitHub connector; fake model, fake embedder, and
fake GitHub client.
It should run in the normal suite and catch broken orchestration contracts
without network, provider credentials, or Docker.
"""

from pathlib import Path
import pytest

from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.memory import InMemoryStore, MemoryRecord, scope_key_for
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore, budget_scope_key
from openloop.testing import FakeEmbedder, FakeGitHub, ScriptedGateway
from openloop.testing import tool_call_response

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"

pytestmark = pytest.mark.integration


async def test_runtime_memory_tool_approval_usage_flow():
    agent = load_agent(AGENT_YAML)
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

    resolved = await tools.resolve(approval_id, "@maciag.artur", approve=True)
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
    assert stored.decided_by == "@maciag.artur"

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


def test_surface_hint_shapes_system_prompt():
    # The Slack hint shapes content (no tables, no heading-heavy replies) for
    # what Slack's server-side Markdown rendering can't express. It must never
    # ask for mrkdwn syntax — the delivery layer owns rendering.
    agent = load_agent(AGENT_YAML)
    runtime = Runtime(agent, gateway=ScriptedGateway([]))

    slack = runtime._build_messages(Task(text="hi", surface="slack"), [])
    system = slack[0]["content"]
    assert "Slack thread" in system
    assert "Markdown tables" in system
    assert "mrkdwn" not in system

    # Other surfaces get the base prompt untouched.
    web = runtime._build_messages(Task(text="hi", surface="webhook"), [])
    assert "Slack" not in web[0]["content"]


def test_tool_facts_ground_system_prompt():
    # A tool-bearing agent is told the three facts it can't infer: tools are
    # its only external reach, no fitting tool means saying so (not inventing
    # data), and the loop's turn budget. The stated budget must track the
    # loop's real cap so the two can't drift apart.
    from openloop.runtime.pipeline import MAX_TOOL_ITERS

    agent = load_agent(AGENT_YAML)
    tools = ToolGateway(tools=[GitHubConnector(FakeGitHub())])
    runtime = Runtime(agent, gateway=ScriptedGateway([]), tools=tools)

    system = runtime._build_messages(Task(text="hi", surface="webhook"), [])[0][
        "content"
    ]
    assert "never invent" in system
    assert f"at most {MAX_TOOL_ITERS} model turns" in system

    # Without tools, no tool facts — a capability claim about tools that
    # don't exist would itself invite invention.
    bare = Runtime(agent, gateway=ScriptedGateway([]))
    bare_system = bare._build_messages(Task(text="hi", surface="webhook"), [])[0][
        "content"
    ]
    assert "never invent" not in bare_system
    assert "tool" not in bare_system.lower()
