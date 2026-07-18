"""Tests for the /usage and /audit read-only observability endpoints."""

import pytest
from fastapi.testclient import TestClient

from openloop.app import create_app
from openloop.usage import InMemoryUsageStore, UsageRecord


@pytest.fixture
def client():
    usage = InMemoryUsageStore()
    app = create_app(compose_overrides={"usage": usage})
    with TestClient(app) as c:
        c.usage = usage  # type: ignore[attr-defined]
        yield c


def test_usage_summary_reports_budget(client):
    body = client.get("/usage").json()
    assert body["agent"] == "dev-platform"
    assert body["monthly_budget_usd"] == 50
    assert body["per_task_budget_usd"] == 0.5
    assert body["month_to_date_usd"] == 0


def test_audit_lists_recent_records(client):
    # record() just appends; seed the list directly to stay synchronous.
    client.usage.records.append(UsageRecord(
        scope_key="ws:acme:agent:dev-platform", workspace="acme",
        agent="dev-platform", model="gpt-4o-mini", channel="#dev-platform",
        cost_usd=0.004, outcome="ok"))

    records = client.get("/audit").json()
    assert len(records) == 1
    assert records[0]["agent"] == "dev-platform"
    assert records[0]["cost_usd"] == 0.004
    assert records[0]["outcome"] == "ok"


def test_audit_respects_limit(client):
    resp = client.get("/audit?limit=10")
    assert resp.status_code == 200
