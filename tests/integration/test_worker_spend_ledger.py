"""Integration: the Phase 4 spend ledger fails closed on BOTH durable paths.

Per the roadmap's leading invariant there is no single "coding-worker path":
the checkpoint-connector fallback and the workflow-backed path must both run
attempts through the one shared :class:`GitWorkspaceOrchestrator`, so wiring
the ledger there caps them both — including an engine-less deploy. These tests
drive each path end to end with a real orchestrator (git subprocesses faked)
and assert the over-budget attempt is recorded, failed, and never becomes a
push or a PR.
"""

import pytest

from openloop.checkpoints import InMemoryCheckpointStore
from openloop.credentials import EnvCredentialResolver
from openloop.tools.coding_worker import (
    CodingWorkerConnector,
    GitWorkspaceOrchestrator,
)
from openloop.usage import InMemoryUsageStore, WorkerSpendLedger
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.coding_worker import build_coding_worker_workflow
from openloop.testing import FakeCodingWorker, FakeGitHub


def _orchestrator(monkeypatch, usage, *, cost_usd, per_task_usd=0.50):
    ledger = WorkerSpendLedger(
        usage=usage,
        scope_key="ws:acme:agent:dev-platform",
        workspace="acme",
        agent="dev-platform",
        model="m",
        per_task_usd=per_task_usd,
    )
    orch = GitWorkspaceOrchestrator(
        FakeCodingWorker(title="t", body="b", cost_usd=cost_usd),
        EnvCredentialResolver({"github": "tok"}),
        ledger=ledger,
    )

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    return orch


async def test_connector_path_fails_closed_over_cap(monkeypatch):
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    connector = CodingWorkerConnector(
        _orchestrator(monkeypatch, usage, cost_usd=0.75),
        github,
        checkpoints=InMemoryCheckpointStore(),
    )

    result = await connector.execute(
        "pr:write", {"repo": "acme/x", "instruction": "x", "job_id": "j1"}
    )

    assert not result.ok
    assert result.data["status"] == "failed"
    assert "per-task budget" in result.data["error"]
    assert github.pulls == []  # fail-closed: no PR from an over-budget run
    assert usage.records[0].outcome == "over_task_budget"
    # "failed" is terminal — the startup reconciler must NOT re-drive the job
    # and spend the budget again on every recovery pass.
    assert await connector.resume_incomplete() == []
    assert len(usage.records) == 1


async def test_workflow_path_fails_closed_over_cap(monkeypatch):
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    store = InMemoryWorkflowStore()
    engine = WorkflowEngine(store)
    engine.register(
        build_coding_worker_workflow(
            _orchestrator(monkeypatch, usage, cost_usd=0.75), github
        )
    )

    await engine.start(
        "coding_worker", "j1",
        {"job_id": "j1", "repo": "acme/x", "instruction": "x"},
    )
    instance = await engine.send_event("j1", "await_approval", {})

    assert instance.status == "failed"
    assert "per-task budget" in instance.error
    assert github.pulls == []
    assert usage.records[0].outcome == "over_task_budget"


async def test_within_budget_run_is_recorded_and_ships(monkeypatch):
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    connector = CodingWorkerConnector(
        _orchestrator(monkeypatch, usage, cost_usd=0.25), github
    )

    result = await connector.execute(
        "pr:write", {"repo": "acme/x", "instruction": "x", "job_id": "j2"}
    )

    assert result.ok
    assert len(github.pulls) == 1
    (record,) = usage.records
    assert record.outcome == "ok"
    assert record.task_kind == "coding_worker"
    # Worker spend now counts against the same monthly scope /usage reports.
    assert await usage.monthly_total("ws:acme:agent:dev-platform") == 0.25
