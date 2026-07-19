"""Regression: resolving an approval persists the decision, not by identity.

The in-memory store is now snapshot-isolated — ``get`` returns a fresh copy per
call, like a row-backed store — so a decision is only visible later if the
gateway wrote it through a store op (``claim_decision``), never by mutating a
shared object. These tests lock that in.
"""

from pathlib import Path

from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.testing import FakeGitHub

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


async def test_resolution_persists_through_get_copies():
    agent = load_agent(AGENT_YAML)
    github = FakeGitHub()
    store = InMemoryApprovalStore()
    gw = ToolGateway(tools=[GitHubConnector(github)], approvals=store)

    pending = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "executed"
    assert github.created  # executed
    # The persisted record reflects the decision (only true if claim_decision ran).
    stored = await store.get(pending.approval.id)
    assert stored.status == "approved"
    assert stored.decided_by == "@maciag.artur"


async def test_denied_resolution_persists():
    agent = load_agent(AGENT_YAML)
    store = InMemoryApprovalStore()
    gw = ToolGateway(tools=[GitHubConnector(FakeGitHub())], approvals=store)

    pending = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    await gw.resolve(pending.approval.id, "@maciag.artur", approve=False)

    stored = await store.get(pending.approval.id)
    assert stored.status == "denied"
