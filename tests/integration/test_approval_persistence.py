"""Regression: resolving an approval must persist via update(), not identity.

The in-memory store returns the same object from get(), so in-place mutation
"works" by accident. A durable store (Postgres) returns a fresh object per
get(), so the decision is only persisted if the gateway calls update(). This
double mimics that to lock in the fix.
"""

import copy

from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.testing import EXAMPLE_AGENT, FakeGitHub


class CopyingApprovalStore(InMemoryApprovalStore):
    """Returns detached copies from get(), like a row-backed store would."""

    async def get(self, request_id: str):
        original = self._by_id.get(request_id)
        return copy.deepcopy(original) if original is not None else None


async def test_resolution_persists_through_get_copies():
    agent = load_agent(EXAMPLE_AGENT)
    github = FakeGitHub()
    store = CopyingApprovalStore()
    gw = ToolGateway(tools=[GitHubConnector(github)], approvals=store)

    pending = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    resolved = await gw.resolve(pending.approval.id, "@priya", approve=True)

    assert resolved.status == "executed"
    assert github.created  # executed
    # The persisted record reflects the decision (only true if update() ran).
    stored = await store.get(pending.approval.id)
    assert stored.status == "approved"
    assert stored.decided_by == "@priya"


async def test_denied_resolution_persists():
    agent = load_agent(EXAMPLE_AGENT)
    store = CopyingApprovalStore()
    gw = ToolGateway(tools=[GitHubConnector(FakeGitHub())], approvals=store)

    pending = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    await gw.resolve(pending.approval.id, "@priya", approve=False)

    stored = await store.get(pending.approval.id)
    assert stored.status == "denied"
