"""End-to-end HTTP test of the tool + approval endpoints."""

import pytest
from fastapi.testclient import TestClient

from openloop.app import create_app
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.testing import FakeGitHub


@pytest.fixture
def client():
    # Inject a gateway with a fake GitHub client (no network).
    fake = FakeGitHub()
    app = create_app(
        compose_overrides={
            "tools_factory": lambda stores: ToolGateway(
                tools=[GitHubConnector(fake)], approvals=stores.approvals
            )
        }
    )
    with TestClient(app) as c:
        c.fake_github = fake  # type: ignore[attr-defined]
        yield c


def test_read_action_executes(client):
    resp = client.post(
        "/tools/invoke",
        json={"action": "github.issues:read", "args": {"repo": "a/b", "number": 3}},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "executed"


def test_write_requires_approval_then_executes(client):
    # 1. Invoke a write action — held for approval.
    resp = client.post(
        "/tools/invoke",
        json={
            "action": "github.issues:write",
            "args": {"repo": "acme/x", "title": "Track decision"},
            "requested_by": "U1",
        },
    )
    body = resp.json()
    assert body["status"] == "pending_approval"
    approval_id = body["approval_id"]
    assert approval_id

    # 2. It shows up as pending.
    listing = client.get("/approvals").json()
    assert any(r["id"] == approval_id for r in listing)
    assert client.fake_github.created == []

    # 3. A non-approver is rejected.
    bad = client.post(
        f"/approvals/{approval_id}/resolve",
        json={"approver": "@random", "approve": True},
    )
    assert bad.status_code == 403

    # 4. An approver approves — the issue is created.
    ok = client.post(
        f"/approvals/{approval_id}/resolve",
        json={"approver": "@maciag.artur", "approve": True},
    )
    assert ok.json()["status"] == "executed"
    assert client.fake_github.created[0]["title"] == "Track decision"


def test_disallowed_action_forbidden(client):
    resp = client.post(
        "/tools/invoke", json={"action": "github.repos:delete", "args": {}}
    )
    assert resp.json()["status"] == "forbidden"
