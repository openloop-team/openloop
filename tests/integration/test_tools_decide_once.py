"""Decide-once: atomic-claim race matrix and the decision reconciler.

Exercises ``ToolGateway.resolve``'s decide-first ordering and
``reconcile_decisions`` against a real workflow engine (the coding worker) and a
direct tool (GitHub), pinning the invariants the design names: one arbiter, one
effect, the winner's identity everywhere, and crash-window healing.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from openloop.agents import load_agent
from openloop.approvals import ApprovalRequest, InMemoryApprovalStore
from openloop.tools import ToolGateway
from openloop.tools.coding_worker import CodingWorkerConnector
from openloop.tools.gateway import _workflow_initial_state
from openloop.tools.github import GitHubConnector
from openloop.testing import FakeGitHub, FakeWorkerOrchestrator
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _engine(orchestrator=None, github=None):
    orchestrator = orchestrator or FakeWorkerOrchestrator()
    github = github or FakeGitHub()
    engine = WorkflowEngine(InMemoryWorkflowStore())
    from openloop.workflows.coding_worker import build_coding_worker_workflow

    engine.register(build_coding_worker_workflow(orchestrator, github))
    return engine, orchestrator, github


def _workflow_gateway(*, approvals=None, github=None, orchestrator=None):
    engine, orchestrator, github = _engine(orchestrator, github)
    gw = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(orchestrator, github)],
        approvals=approvals or InMemoryApprovalStore(),
        engine=engine,
    )
    return gw, engine, github, orchestrator


def _direct_gateway(*, approvals=None, github=None):
    github = github or FakeGitHub()
    gw = ToolGateway(
        tools=[GitHubConnector(github)],
        approvals=approvals or InMemoryApprovalStore(),
    )
    return gw, github


async def _seed_workflow_request(
    gw,
    approvers,
    *,
    park=True,
    job_id="job1",
    repo="acme/x",
    workflow_backed=True,
    **overrides,
):
    """A stamped workflow-backed approval, its instance optionally parked."""
    req = ApprovalRequest(
        agent="a",
        action="coding_worker.pr:write",
        tool="coding_worker",
        permission="pr:write",
        args={"job_id": job_id, "repo": repo, "instruction": "do it"},
        approvers=list(approvers),
        summary="run coding worker",
        workflow_backed=workflow_backed,
        workflow_instance_id=job_id,
        **overrides,
    )
    await gw.approvals.create(req)
    if park and gw.engine is not None:
        await gw.engine.start("coding_worker", job_id, _workflow_initial_state(req))
    return req


async def _seed_direct_request(
    gw, approvers, *, args=None, tool="github", workflow_backed=False, **overrides
):
    req = ApprovalRequest(
        agent="a",
        action="github.issues:write",
        tool=tool,
        permission="issues:write",
        args=args if args is not None else {"repo": "acme/x", "title": "T"},
        approvers=list(approvers),
        summary="create issue",
        workflow_backed=workflow_backed,
        **overrides,
    )
    await gw.approvals.create(req)
    return req


async def _drive(engine, instance_id):
    """Await any background drive the resolve kicked off, then read the instance."""
    await engine.wait_background(instance_id)
    return await engine.store.get(instance_id)


# --------------------------------------------------------------------------- #
# invoke-time stamping
# --------------------------------------------------------------------------- #


async def test_invoke_stamps_workflow_marker_and_instance_id():
    agent = load_agent(AGENT_YAML)
    gw, _, _, _ = _workflow_gateway()
    pending = await gw.invoke(
        agent, "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"}
    )
    req = pending.approval
    assert req.workflow_backed is True
    assert req.workflow_instance_id == req.args["job_id"]


async def test_invoke_stamps_direct_marker_false():
    agent = load_agent(AGENT_YAML)
    gw, _ = _direct_gateway()
    pending = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    assert pending.approval.workflow_backed is False
    assert pending.approval.workflow_instance_id is None


# --------------------------------------------------------------------------- #
# race matrix
# --------------------------------------------------------------------------- #


async def test_two_approvers_one_wins_one_wake_identity_is_winners():
    gw, engine, _, orchestrator = _workflow_gateway()
    req = await _seed_workflow_request(gw, ["@a", "@b"])

    a, b = await asyncio.gather(
        gw.resolve(req.id, "@a", approve=True),
        gw.resolve(req.id, "@b", approve=True),
    )

    stored = await gw.approvals.get(req.id)
    assert stored.status == "approved"
    winner = stored.decided_by
    assert winner in ("@a", "@b")
    # Both invocations report the winner's identity, never the caller's.
    assert a.decided_by == winner and b.decided_by == winner
    await _drive(engine, req.workflow_instance_id)
    # Exactly one run of the worker (one wake).
    assert len(orchestrator.runs) == 1
    # The wake's recorded approver is the winner.
    inst = await engine.store.get(req.workflow_instance_id)
    assert inst.state["events"]["await_approval"]["approver"] == winner


async def test_approve_races_deny_one_effect():
    gw, engine, _, orchestrator = _workflow_gateway()
    req = await _seed_workflow_request(gw, ["@a", "@b"])

    await asyncio.gather(
        gw.resolve(req.id, "@a", approve=True),
        gw.resolve(req.id, "@b", approve=False),
    )
    await _drive(engine, req.workflow_instance_id)

    stored = await gw.approvals.get(req.id)
    inst = await engine.store.get(req.workflow_instance_id)
    if stored.status == "approved":
        assert len(orchestrator.runs) == 1  # woken, not cancelled
        assert inst.status != "cancelled"
    else:
        assert stored.status == "denied"
        assert orchestrator.runs == []  # cancelled, never ran
        assert inst.status == "cancelled"


async def test_non_approver_rejected_before_any_claim():
    gw, _ = _direct_gateway()
    req = await _seed_direct_request(gw, ["@a"])
    inv = await gw.resolve(req.id, "@intruder", approve=True)
    assert inv.status == "forbidden"
    assert (await gw.approvals.get(req.id)).status == "pending"


async def test_non_approver_on_decided_row_is_forbidden_no_effect():
    gw, engine, _, orchestrator = _workflow_gateway()
    req = await _seed_workflow_request(gw, ["@a"])
    await gw.resolve(req.id, "@a", approve=True)
    await _drive(engine, req.workflow_instance_id)
    runs_after_decision = len(orchestrator.runs)

    inv = await gw.resolve(req.id, "@intruder", approve=True)
    assert inv.status == "forbidden"
    # No re-ensure effect ran for the non-approver.
    assert len(orchestrator.runs) == runs_after_decision


async def test_winner_approve_wakes_and_returns_invocation():
    gw, engine, github, orchestrator = _workflow_gateway()
    req = await _seed_workflow_request(gw, ["@a"])
    inv = await gw.resolve(req.id, "@a", approve=True)
    assert inv.decided_by == "@a"
    await _drive(engine, req.workflow_instance_id)
    assert len(orchestrator.runs) == 1
    assert github.pulls  # a PR was opened


async def test_winner_deny_cancels_workflow():
    gw, engine, _, orchestrator = _workflow_gateway()
    req = await _seed_workflow_request(gw, ["@a"])
    inv = await gw.resolve(req.id, "@a", approve=False)
    assert inv.status == "denied"
    assert inv.decided_by == "@a"
    inst = await engine.store.get(req.workflow_instance_id)
    assert inst.status == "cancelled"
    # Row marked reconciled — its effect (cancel) is done.
    assert (await gw.approvals.get(req.id)).effect_at is not None


async def test_lost_claim_approve_reensures_parked_workflow():
    gw, engine, github, orchestrator = _workflow_gateway()
    req = await _seed_workflow_request(gw, ["@a", "@b"])
    # First click wins and wakes.
    await gw.resolve(req.id, "@a", approve=True)
    # Second click loses the claim; it must re-ensure, not report stale state.
    inv = await gw.resolve(req.id, "@b", approve=True)
    assert inv.decided_by == "@a"  # winner's identity
    await _drive(engine, req.workflow_instance_id)
    assert len(orchestrator.runs) == 1


async def test_lost_claim_deny_cancels_still_parked_workflow():
    gw, engine, _, orchestrator = _workflow_gateway()
    req = await _seed_workflow_request(gw, ["@a", "@b"])
    await gw.resolve(req.id, "@a", approve=False)
    inv = await gw.resolve(req.id, "@b", approve=False)
    assert inv.status == "denied"
    assert inv.decided_by == "@a"
    inst = await engine.store.get(req.workflow_instance_id)
    assert inst.status == "cancelled"


# --------------------------------------------------------------------------- #
# availability gating (approve only), never reclassification
# --------------------------------------------------------------------------- #


async def test_unregistered_tool_approve_refused_pre_claim():
    gw, _ = _direct_gateway()
    # A row whose tool nobody registered (connector disabled/drifted); legacy
    # None marker so it classifies "unknown".
    req = await _seed_direct_request(
        gw, ["@a"], tool="ghost", workflow_backed=None
    )
    inv = await gw.resolve(req.id, "@a", approve=True)
    assert inv.status == "forbidden"
    assert "not registered" in inv.message
    assert (await gw.approvals.get(req.id)).status == "pending"


async def test_unregistered_tool_deny_claims_and_reports_legacy_unmarked():
    gw, _ = _direct_gateway()
    req = await _seed_direct_request(
        gw, ["@a"], tool="ghost", workflow_backed=None
    )
    inv = await gw.resolve(req.id, "@a", approve=False)
    assert inv.status == "denied"
    stored = await gw.approvals.get(req.id)
    assert stored.status == "denied"
    # Legacy unknown-tool denial can't be classified — left for the sweep.
    assert stored.effect_at is None


async def test_engine_less_approve_gating_on_stamped_workflow_row():
    # Connector present, but no engine — the workflow effect can't run here.
    github = FakeGitHub()
    orchestrator = FakeWorkerOrchestrator()
    gw = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(orchestrator, github)],
        approvals=InMemoryApprovalStore(),
        engine=None,
    )
    req = await _seed_workflow_request(gw, ["@a"], park=False)
    inv = await gw.resolve(req.id, "@a", approve=True)
    assert inv.status == "forbidden"
    assert (await gw.approvals.get(req.id)).status == "pending"


async def test_engine_less_already_approved_reports_approved_unmarked():
    github = FakeGitHub()
    orchestrator = FakeWorkerOrchestrator()
    gw = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(orchestrator, github)],
        approvals=InMemoryApprovalStore(),
        engine=None,
    )
    # An already-approved stamped row (decided elsewhere) on an engine-less gw.
    req = await _seed_workflow_request(
        gw, ["@a"], park=False, status="approved", decided_by="@a"
    )
    inv = await gw.resolve(req.id, "@a", approve=True)
    assert inv.status == "approved"
    assert inv.decided_by == "@a"
    assert (await gw.approvals.get(req.id)).effect_at is None


async def test_workflow_capability_drift_workflow_name_absent():
    # Engine present, but the tool's workflow name isn't registered on it.
    engine = WorkflowEngine(InMemoryWorkflowStore())  # no coding_worker registered
    github = FakeGitHub()
    orchestrator = FakeWorkerOrchestrator()
    gw = ToolGateway(
        tools=[GitHubConnector(github), CodingWorkerConnector(orchestrator, github)],
        approvals=InMemoryApprovalStore(),
        engine=engine,
    )
    req = await _seed_workflow_request(gw, ["@a"], park=False)
    inv = await gw.resolve(req.id, "@a", approve=True)
    assert inv.status == "forbidden"
    assert (await gw.approvals.get(req.id)).status == "pending"


async def test_duplicate_click_decided_row_connector_absent_reports_decision():
    # A decided workflow row on a gateway missing the connector: report the
    # durable decision, never "tool unavailable".
    engine = WorkflowEngine(InMemoryWorkflowStore())
    gw = ToolGateway(
        tools=[GitHubConnector(FakeGitHub())],  # coding_worker connector absent
        approvals=InMemoryApprovalStore(),
        engine=engine,
    )
    req = await _seed_workflow_request(
        gw, ["@a"], park=False, status="approved", decided_by="@a"
    )
    inv = await gw.resolve(req.id, "@a", approve=True)
    assert inv.status == "approved"
    assert inv.decided_by == "@a"


# --------------------------------------------------------------------------- #
# id-collision + mode drift
# --------------------------------------------------------------------------- #


async def test_deny_direct_request_never_cancels_colliding_workflow():
    gw, engine, _, orchestrator = _workflow_gateway()
    # Park a real workflow named "live".
    live = await _seed_workflow_request(gw, ["@a"], job_id="live")
    # A direct request whose model-supplied args carry job_id="live".
    direct = await _seed_direct_request(
        gw, ["@a"], args={"repo": "acme/x", "title": "T", "job_id": "live"}
    )
    await gw.resolve(direct.id, "@a", approve=False)
    # The colliding live workflow is untouched.
    inst = await engine.store.get("live")
    assert inst.status not in ("cancelled",)


async def test_false_row_direct_executes_no_instance_created():
    gw, engine, github, _ = _workflow_gateway()
    req = await _seed_direct_request(gw, ["@a"])  # workflow_backed=False
    inv = await gw.resolve(req.id, "@a", approve=True)
    assert inv.status == "executed"
    assert github.created  # the direct tool ran
    # No workflow instance was fabricated for the direct request.
    assert await engine.store.get(req.id) is None


# --------------------------------------------------------------------------- #
# marking containment
# --------------------------------------------------------------------------- #


async def test_mark_failure_after_direct_execute_is_contained(caplog):
    class FlakyMark(InMemoryApprovalStore):
        async def mark_reconciled(self, request_id):
            raise RuntimeError("mark boom")

    gw, github = _direct_gateway(approvals=FlakyMark())
    req = await _seed_direct_request(gw, ["@a"])
    with caplog.at_level(logging.ERROR):
        inv = await gw.resolve(req.id, "@a", approve=True)
    assert inv.status == "executed"  # the ToolResult survived
    assert github.created
    stored = await gw.approvals.get(req.id)
    assert stored.status == "approved"
    assert stored.effect_at is None  # unmarked — the sweep will retry


# --------------------------------------------------------------------------- #
# decision reconciler
# --------------------------------------------------------------------------- #


async def test_reconcile_wakes_approved_parked_instance_no_session():
    gw, engine, github, orchestrator = _workflow_gateway()
    # Approved row, its instance still parked (crash after claim, before wake).
    req = await _seed_workflow_request(
        gw, ["@a"], status="approved", decided_by="@a"
    )
    healed = await gw.reconcile_decisions()
    assert healed == 1
    await _drive(engine, req.workflow_instance_id)
    assert len(orchestrator.runs) == 1
    assert (await gw.approvals.get(req.id)).effect_at is not None


async def test_reconcile_recreates_missing_instance():
    gw, engine, github, orchestrator = _workflow_gateway()
    # Approved row whose instance was never created (crash before start).
    req = await _seed_workflow_request(
        gw, ["@a"], park=False, status="approved", decided_by="@a"
    )
    assert await engine.store.get(req.workflow_instance_id) is None
    healed = await gw.reconcile_decisions()
    assert healed == 1
    await _drive(engine, req.workflow_instance_id)
    assert len(orchestrator.runs) == 1


async def test_reconcile_cancels_denied_non_terminal_instance():
    gw, engine, _, orchestrator = _workflow_gateway()
    # Denied row, instance still parked (crash after deny claim, before cancel).
    req = await _seed_workflow_request(
        gw, ["@a"], status="denied", decided_by="@a"
    )
    healed = await gw.reconcile_decisions()
    assert healed == 1
    inst = await engine.store.get(req.workflow_instance_id)
    assert inst.status == "cancelled"
    assert (await gw.approvals.get(req.id)).effect_at is not None


async def test_reconcile_direct_row_retired_on_sight():
    gw, github = _direct_gateway()
    req = await _seed_direct_request(gw, ["@a"], status="approved", decided_by="@a")
    healed = await gw.reconcile_decisions()
    assert healed == 1
    # No automatic re-execute; the row is retired.
    assert github.created == []
    assert (await gw.approvals.get(req.id)).effect_at is not None


async def test_reconcile_tristate_direct_vs_legacy_unknown():
    gw, _ = _direct_gateway()  # only github registered
    # Post-migration direct row (False), connector absent → marked, no cancel.
    direct = await _seed_direct_request(
        gw, ["@a"], tool="ghost", workflow_backed=False,
        status="approved", decided_by="@a",
    )
    # Legacy row (None), unknown tool → deferred unmarked.
    legacy = await _seed_direct_request(
        gw, ["@a"], tool="ghost", workflow_backed=None,
        status="approved", decided_by="@a",
    )
    await gw.reconcile_decisions()
    assert (await gw.approvals.get(direct.id)).effect_at is not None
    assert (await gw.approvals.get(legacy.id)).effect_at is None


async def test_reconcile_missing_connector_defers_then_heals():
    engine = WorkflowEngine(InMemoryWorkflowStore())
    github = FakeGitHub()
    orchestrator = FakeWorkerOrchestrator()
    from openloop.workflows.coding_worker import build_coding_worker_workflow

    engine.register(build_coding_worker_workflow(orchestrator, github))
    approvals = InMemoryApprovalStore()
    # Gateway WITHOUT the coding_worker connector registered.
    gw = ToolGateway(
        tools=[GitHubConnector(github)], approvals=approvals, engine=engine
    )
    req = await _seed_workflow_request(
        gw, ["@a"], park=False, status="approved", decided_by="@a",
        workflow_backed=None,  # legacy → classified by registry (unknown here)
    )
    healed = await gw.reconcile_decisions()
    assert healed == 0
    assert (await gw.approvals.get(req.id)).effect_at is None
    # Register the connector; the next sweep heals it.
    gw.register(CodingWorkerConnector(orchestrator, github))
    healed2 = await gw.reconcile_decisions()
    assert healed2 == 1
    await _drive(engine, req.workflow_instance_id)
    assert len(orchestrator.runs) == 1


async def test_reconcile_stamped_denied_connector_absent_cancels_immediately():
    engine = WorkflowEngine(InMemoryWorkflowStore())
    github = FakeGitHub()
    orchestrator = FakeWorkerOrchestrator()
    from openloop.workflows.coding_worker import build_coding_worker_workflow

    engine.register(build_coding_worker_workflow(orchestrator, github))
    approvals = InMemoryApprovalStore()
    gw = ToolGateway(
        tools=[GitHubConnector(github)], approvals=approvals, engine=engine
    )
    # Stamped True denied row, connector absent — cancel needs no connector.
    req = await _seed_workflow_request(
        gw, ["@a"], status="denied", decided_by="@a"
    )
    healed = await gw.reconcile_decisions()
    assert healed == 1
    inst = await engine.store.get(req.workflow_instance_id)
    assert inst.status == "cancelled"
    assert (await gw.approvals.get(req.id)).effect_at is not None


async def test_reconcile_poison_head_pagination_heals_younger_rows():
    gw, engine, github, orchestrator = _workflow_gateway()
    # A batch of unavailable (unknown-tool, legacy) approved rows at the head,
    # then a healable direct row behind them. Small limit forces pagination.
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 7, 19, tzinfo=timezone.utc)
    for i in range(5):
        await _seed_direct_request(
            gw, ["@a"], tool="ghost", workflow_backed=None,
            status="approved", decided_by="@a",
            id=f"poison{i}", created_at=base + timedelta(minutes=i),
        )
    healable = await _seed_direct_request(
        gw, ["@a"], status="approved", decided_by="@a",
        id="healable", created_at=base + timedelta(minutes=10),
    )
    # Shrink the batch size so the poison rows fill the first page.
    gw.approvals.decided_unreconciled = _small_limit(gw.approvals, 3)
    await gw.reconcile_decisions()
    assert (await gw.approvals.get("healable")).effect_at is not None


async def test_reconcile_per_row_isolation_continues_past_raise():
    gw, engine, github, orchestrator = _workflow_gateway()
    good1 = await _seed_direct_request(gw, ["@a"], status="approved", decided_by="@a", id="g1")
    # A row whose classification will raise inside _reconcile_decision.
    boom = await _seed_workflow_request(
        gw, ["@a"], job_id="boom", status="approved", decided_by="@a"
    )
    good2 = await _seed_direct_request(gw, ["@a"], status="approved", decided_by="@a", id="g2")

    original = gw._ensure_approved_workflow

    async def _raising(request):
        if request.id == boom.id:
            raise RuntimeError("heal boom")
        return await original(request)

    gw._ensure_approved_workflow = _raising
    await gw.reconcile_decisions()
    # The direct rows around the poison row were still retired.
    assert (await gw.approvals.get("g1")).effect_at is not None
    assert (await gw.approvals.get("g2")).effect_at is not None
    assert (await gw.approvals.get(boom.id)).effect_at is None


async def test_reconcile_repeated_sweep_drains_to_empty():
    gw, engine, github, orchestrator = _workflow_gateway()
    await _seed_direct_request(gw, ["@a"], status="approved", decided_by="@a", id="d1")
    await _seed_workflow_request(gw, ["@a"], status="approved", decided_by="@a")
    await gw.reconcile_decisions()
    # A repeated sweep is a no-op and the unreconciled set is empty.
    assert await gw.reconcile_decisions() == 0
    assert await gw.approvals.decided_unreconciled() == []


def _small_limit(store, limit):
    original = store.decided_unreconciled

    async def _wrapped(limit=200, after=None, _forced=limit):
        return await original(limit=_forced, after=after)

    return _wrapped
