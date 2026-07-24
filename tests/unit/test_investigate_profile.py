"""Unit tests for the workspace_task ``investigate:read`` profile (Task 12).

Covers the connector-level surface only: ``investigate:read`` has
``gate=Gate.NONE`` (see ``openloop.tasks.contract``), so in Stage 1 it never
becomes an approval or a durable workflow instance — it always runs through
``CodingWorkerConnector.execute()`` synchronously, and its result flows back
through the model tool-loop. These tests exercise exactly that path.
"""

import pytest

from openloop.agents.schema import Budget
from openloop.credentials import EnvCredentialResolver
from openloop.models.gateway import ModelResponse
from openloop.tasks import WorkspaceTask
from openloop.tasks.investigation import INVESTIGATE_ARGS_VERSION, RepoInvestigator
from openloop.testing import FakeCodingWorker, FakeGitHub, FakeWorkerOrchestrator
from openloop.tools.coding_worker import CodingWorkerConnector, GitWorkspaceOrchestrator
from openloop.usage import (
    InMemoryUsageStore,
    WorkerBudgetExceeded,
    WorkerSpendLedger,
    budget_scope_key,
)
from tests.support.agents import make_agent


class _FakeGateway:
    """A network-free stand-in for the model gateway RepoInvestigator calls."""

    def __init__(self, text: str, *, cost_usd: float = 0.02) -> None:
        self._text = text
        self._cost_usd = cost_usd
        self.calls: list = []

    async def complete(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages})
        return ModelResponse(
            text=self._text,
            model=model,
            cost_usd=self._cost_usd,
            prompt_tokens=11,
            completion_tokens=7,
        )


def _investigator(
    text: str = "SUMMARY: it returns None\nFINDINGS:\n- parse() at a.py:1 returns None\n",
    *,
    cost_usd: float = 0.02,
) -> RepoInvestigator:
    return RepoInvestigator("m", gateway=_FakeGateway(text, cost_usd=cost_usd))


def _connector(*, investigator=None, runner=None, github=None) -> CodingWorkerConnector:
    return CodingWorkerConnector(
        runner or FakeWorkerOrchestrator(),
        github or FakeGitHub(),
        investigator=investigator,
    )


def test_supported_permissions_includes_investigate_when_investigator_set():
    conn = _connector(investigator=_investigator())
    assert conn.supported_permissions() == {"code:write", "investigate:read"}


def test_supported_permissions_excludes_investigate_without_investigator():
    conn = _connector()
    assert conn.supported_permissions() == {"code:write"}
    assert "investigate:read" not in conn.supported_permissions()


def test_investigate_describe_has_typed_args():
    spec = _connector(investigator=_investigator()).describe("investigate:read")
    assert spec.version == 1 == INVESTIGATE_ARGS_VERSION
    assert spec.model is not None
    assert "question" in spec.parameters["properties"]


async def test_execute_investigate_returns_evidence_bundle_and_opens_no_pr():
    github = FakeGitHub()
    # execute() now delegates to orchestrator.run_investigation, which owns
    # provisioning + the investigator call — the fake's run_investigation
    # returns its OWN canned bundle (it never echoes the investigator's text,
    # see FakeWorkerOrchestrator.run_investigation), so cost_usd is configured
    # here and the summary assertion below reads the fake's own canned value
    # rather than the investigator's.
    runner = FakeWorkerOrchestrator(cost_usd=0.02)
    conn = _connector(investigator=_investigator(), runner=runner, github=github)

    args = conn.prepare_args(
        "investigate:read", {"repo": "a/b", "question": "why does parse return None?"}
    )
    assert args["profile"] == "investigate"
    assert args.get("job_id")

    result = await conn.execute("investigate:read", args)

    assert result.ok is True
    outcome = result.data["outcome"]
    assert outcome["kind"] == "evidence_bundle"
    assert outcome["findings"].strip() != ""
    assert outcome["summary"] == runner.investigation_summary
    assert result.data["cost_usd"] == 0.02
    # Never opens a PR, never pushes — this is a read-only profile.
    assert github.pulls == []


async def test_execute_investigate_provisions_a_readonly_workspace_and_cleans_up():
    runner = FakeWorkerOrchestrator()
    conn = _connector(investigator=_investigator(), runner=runner)

    args = conn.prepare_args(
        "investigate:read", {"repo": "a/b", "question": "?", "ref": "dev"}
    )
    result = await conn.execute("investigate:read", args)

    assert result.ok is True
    assert runner.readonly_provisions == [("a/b", "dev")]


async def test_execute_investigate_without_investigator_fails_closed():
    conn = _connector()  # no investigator configured
    args = conn.prepare_args(
        "investigate:read", {"repo": "a/b", "question": "why?"}
    )
    result = await conn.execute("investigate:read", args)

    assert result.ok is False
    assert result.data["status"] == "failed"


async def test_execute_investigate_with_empty_args_returns_failed_result():
    conn = _connector(investigator=_investigator())
    result = await conn.execute("investigate:read", {})

    assert result.ok is False
    assert result.data["status"] == "failed"
    assert "repo" in result.data["error"] or "required" in result.data["error"]


async def test_execute_investigate_missing_repo_returns_failed_result():
    conn = _connector(investigator=_investigator())
    result = await conn.execute("investigate:read", {"question": "why?"})

    assert result.ok is False
    assert result.data["status"] == "failed"


async def test_execute_investigate_missing_question_returns_failed_result():
    conn = _connector(investigator=_investigator())
    result = await conn.execute("investigate:read", {"repo": "a/b"})

    assert result.ok is False
    assert result.data["status"] == "failed"


def test_code_write_permission_and_behavior_are_unchanged():
    conn = _connector(investigator=_investigator())
    spec = conn.describe("code:write")
    assert spec.parameters["properties"].keys() >= {"repo", "instruction"}


# --- GitWorkspaceOrchestrator.run_investigation: the ledger bracket ---
#
# Closes the untracked-spend gap (contract-convergence #4): investigation
# model spend must go through the SAME WorkerSpendLedger the coding worker
# brackets in run_attempt, attributed to the invoking agent, fail-closed over
# the per-task cap. Mirrors the ledger setup in tests/unit/test_worker_ledger.py.


def _investigation_orchestrator(
    monkeypatch, ledger: WorkerSpendLedger
) -> GitWorkspaceOrchestrator:
    orch = GitWorkspaceOrchestrator(
        FakeCodingWorker(),
        EnvCredentialResolver({"github": "tok"}),
        ledger=ledger,
    )

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    return orch


def _investigation_task(*, agent: str, agent_id: str) -> WorkspaceTask:
    return WorkspaceTask(
        task_id="t1",
        profile="investigate",
        entry_action="investigate:read",
        agent=agent,
        agent_id=agent_id,
        profile_state={
            "investigate": {
                "repo": "a/b",
                "question": "why does parse return None?",
                "ref": None,
            }
        },
    )


async def test_run_investigation_records_spend_for_the_invoking_agent(monkeypatch):
    usage = InMemoryUsageStore()
    agent = make_agent("dev-platform", "acme", budget=Budget(per_task_usd=0.50))
    ledger = WorkerSpendLedger(
        usage=usage, model="m", agents={"dev-platform": agent},
        default_agent="dev-platform",
    )
    orch = _investigation_orchestrator(monkeypatch, ledger)
    task = _investigation_task(agent="dev-platform", agent_id=agent.metadata.id)

    bundle, resp = await orch.run_investigation(task, _investigator(cost_usd=0.02))

    # The evidence bundle is returned on the happy path.
    assert bundle.summary == "it returns None"
    assert resp.cost_usd == 0.02
    # A usage entry lands under the invoking agent's scope — the spend is no
    # longer untracked.
    (record,) = usage.records
    assert record.agent == "dev-platform"
    assert record.scope_key == budget_scope_key(agent)
    assert record.outcome == "ok"
    assert record.cost_usd == 0.02


async def test_run_investigation_fails_closed_over_the_per_task_cap(monkeypatch):
    usage = InMemoryUsageStore()
    agent = make_agent("dev-platform", "acme", budget=Budget(per_task_usd=0.10))
    ledger = WorkerSpendLedger(
        usage=usage, model="m", agents={"dev-platform": agent},
        default_agent="dev-platform",
    )
    orch = _investigation_orchestrator(monkeypatch, ledger)
    task = _investigation_task(agent="dev-platform", agent_id=agent.metadata.id)

    with pytest.raises(WorkerBudgetExceeded):
        await orch.run_investigation(task, _investigator(cost_usd=0.75))

    # The over-cap spend is still on the audit trail (fail-closed, not
    # silent) — but no evidence bundle was ever returned as a success.
    assert usage.records[-1].outcome == "over_task_budget"
    assert usage.records[-1].cost_usd == 0.75


# --- Conv Task 3: connector execute() runs on a WorkspaceTask, no WorkerState ---
#
# _execute_investigate must build a real WorkspaceTask and delegate to
# orchestrator.run_investigation instead of calling provision_readonly +
# investigator.investigate directly (no ledger bracket, no WorkspaceTask).
# FakeWorkerOrchestrator.run_investigation returns FULLY CANNED output (it
# never echoes the investigator's own text) and records every task it's
# handed in self.investigations — so the only way this test's canned
# assertions can pass is if execute() actually delegated to
# run_investigation, and the only way to check what the connector built is
# to inspect the captured WorkspaceTask, not to rely on the fake echoing
# inputs.


async def test_execute_investigate_runs_on_workspacetask_no_workerstate():
    runner = FakeWorkerOrchestrator(
        cost_usd=0.03,
        investigation_summary="canned summary from the orchestrator",
        investigation_findings="- (canned) finding one",
    )
    conn = _connector(investigator=_investigator(), runner=runner)
    agent = make_agent("dev-platform", "acme")

    args = conn.prepare_args(
        "investigate:read",
        {"repo": "a/b", "question": "why does parse return None?", "ref": "dev"},
        agent=agent,
        session_id="sess-1",
    )

    result = await conn.execute("investigate:read", args)

    # The ToolResult carries the ORCHESTRATOR's canned bundle unchanged —
    # this can only be true if execute() delegated to run_investigation
    # (the old direct-call path would surface the investigator's own text,
    # "it returns None", instead).
    assert result.ok is True
    outcome = result.data["outcome"]
    assert outcome["kind"] == "evidence_bundle"
    assert outcome["summary"] == "canned summary from the orchestrator"
    assert outcome["findings"] == "- (canned) finding one"
    assert result.data["cost_usd"] == 0.03
    assert result.data["job_id"] == args["job_id"]

    # The orchestrator received exactly one call, carrying a real
    # WorkspaceTask — never a WorkerState — whose profile_state holds ONLY
    # the investigate inputs (no "code" key, no worker_state key), with the
    # same identity fields the code:write path threads into WorkerState.
    (task,) = runner.investigations
    assert isinstance(task, WorkspaceTask)
    assert task.task_id == args["job_id"]
    assert task.profile == "investigate"
    assert task.entry_action == "investigate:read"
    assert task.agent == "dev-platform"
    assert task.agent_id == agent.metadata.id
    assert task.session_id == "sess-1"
    assert set(task.profile_state) == {"investigate"}
    assert task.profile_state["investigate"] == {
        "repo": "a/b",
        "question": "why does parse return None?",
        "ref": "dev",
    }
