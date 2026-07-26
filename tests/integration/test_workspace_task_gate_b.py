"""Gate B: a second real profile uses the shared workspace-task contract with
no escape hatch, plus a structural proof that outcome type is decoupled from
the entry action (design decision 2).

``investigate:read`` has ``gate=Gate.NONE`` (see ``openloop.tasks.contract``),
so the gateway never creates an approval or a workflow instance for it — every
invocation falls straight through to ``CodingWorkerConnector.execute()``
synchronously (never a durable workflow), and its result (an evidence bundle)
flows back to the model through the ordinary tool-loop rather than a hosted
Artifact delivery (that path is for gated outcomes only — out of Stage 1 scope
for the ungated investigation). This file proves two things:

1. The outcome union (``TaskOutcome``/``to_deliverable``) is a property of the
   task core, never derived from the entry action — a task entered as
   "investigate" can hold a ``PullRequest`` outcome with no type error. That is
   the Gate F affordance (a future in-task profile transition) proven
   structurally, without Stage 1 ever exercising a runtime transition.
2. Invoking ``workspace_task.investigate:read`` through the exact same
   ``CodingWorkerConnector``/``ToolGateway`` that serves ``code:write`` — via
   the real production ``build_tool_gateway`` wiring, built the same way
   ``test_worker_backend_wiring.py`` builds it — reaches ``execute()``
   (ungated), returns an ``evidence_bundle`` outcome, and opens no PR. No
   investigate-specific escape hatch exists in the execution core: the only
   seams faked here are the three network-touching ones (model completion,
   git clone, GitHub API), monkeypatched at the class level exactly like
   Task 13's own wiring test does it — ``build_tool_gateway`` has no injection
   point for a fake investigator/orchestrator/github client, so there is no
   "cleaner" seam available without changing production code.
"""

from pathlib import Path

from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.deliverable import Artifact, Prose
from openloop.models.gateway import ModelResponse
from openloop.tasks import EvidenceBundle, PullRequest, WorkspaceTask, to_deliverable
from openloop.tasks.investigation import RepoInvestigator
from openloop.testing import FakeGitHub, FakeWorkerOrchestrator
from openloop.tools.coding_worker import GitWorkspaceOrchestrator
from openloop.tools.openhands_worker import OpenHandsCodingWorker
from openloop.usage import InMemoryUsageStore
from openloop.wiring import builders as appmod
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from tests.support.settings import IsolatedSettings as Settings

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def test_outcome_type_not_welded_to_profile():
    """Decision 2: ``to_deliverable`` dispatches on the outcome's own runtime
    type only — its signature doesn't even accept a task or a profile — so a
    ``WorkspaceTask`` entered via ``investigate:read`` (``profile="investigate"``)
    can hold either outcome shape in its ``profile_state`` with no type gate.
    """
    task = WorkspaceTask(
        task_id="t", profile="investigate", entry_action="investigate:read"
    )

    pr = PullRequest(repo="a/b", branch="x", pr_number=1, pr_url="u", summary="s")
    eb = EvidenceBundle(summary="s", findings="f")

    # Both construct and map to a deliverable with no reference to
    # task.profile anywhere in the call — to_deliverable structurally cannot
    # key off the entry action, because it never sees the task at all.
    assert isinstance(to_deliverable(pr), Prose)
    assert isinstance(to_deliverable(eb), Artifact)

    # The task itself is agnostic too: profile_state holds either outcome
    # shape even though this task entered life as "investigate" and never
    # transitions profile.
    task.profile_state["outcome"] = pr
    assert isinstance(to_deliverable(task.profile_state["outcome"]), Prose)
    task.profile_state["outcome"] = eb
    assert isinstance(to_deliverable(task.profile_state["outcome"]), Artifact)
    assert task.profile == "investigate"
    assert task.entry_action == "investigate:read"


def _settings(**kwargs):
    return Settings(
                coding_worker_enabled=True,
        github_token="t",
        **kwargs,
    )


def _gateway(settings, agents=None, usage=None):
    return appmod.build_tool_gateway(
        settings,
        agents if agents is not None else {"dev-platform": load_agent(AGENT_YAML)},
        InMemoryApprovalStore(),
        InMemoryCheckpointStore(),
        WorkflowEngine(InMemoryWorkflowStore()),
        usage=usage if usage is not None else InMemoryUsageStore(),
    )


async def test_investigation_produces_evidence_and_no_pr(monkeypatch):
    """Gate B end-to-end: ``investigate:read`` runs ungated over the SAME
    production wiring as ``code:write``, yields an ``evidence_bundle``
    outcome, and opens no PR — the second profile over the shared contract,
    no escape hatch. Only the three network-touching seams are faked, all at
    the class level (``build_tool_gateway`` has no injection point for any of
    them): ``RepoInvestigator.investigate`` (model completion),
    ``GitWorkspaceOrchestrator.provision_readonly`` (credentialed git clone —
    delegated to the real ``openloop.testing.FakeWorkerOrchestrator``, so
    "network-free" is a tested fact, not merely an assertion), and
    ``HttpGitHubClient`` (GitHub API) — the last one exists purely so we can
    assert an actual ``FakeGitHub`` recorded zero calls, not just that no
    exception happened to occur.
    """
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)

    fake_github = FakeGitHub()
    monkeypatch.setattr(appmod, "HttpGitHubClient", lambda *a, **k: fake_github)

    fake_orchestrator = FakeWorkerOrchestrator(
        seed_files={"parse.py": "def parse(s):\n    return None\n"}
    )

    async def fake_provision_readonly(self, repo, ref=None):
        # Delegates to the real fake — not a hand-rolled stand-in — so the
        # "network-free" claim is backed by openloop.testing, not just this
        # test file.
        return await fake_orchestrator.provision_readonly(repo, ref)

    monkeypatch.setattr(
        GitWorkspaceOrchestrator, "provision_readonly", fake_provision_readonly
    )

    async def fake_investigate(self, workspace, question, repo):
        return (
            EvidenceBundle(
                summary="parse() returns None on falsy input",
                findings="- parse.py:2 returns None",
            ),
            ModelResponse(
                text="", model=self.model, cost_usd=0.01,
                prompt_tokens=42, completion_tokens=9,
            ),
        )

    monkeypatch.setattr(RepoInvestigator, "investigate", fake_investigate)

    gateway = _gateway(_settings(coding_worker_backend="builtin"))

    # workspace_task is the SAME connector object that serves code:write — no
    # separate "investigate" tool or connector is registered.
    tool = gateway._tools["workspace_task"]
    assert tool.github is fake_github

    agent = load_agent(AGENT_YAML)
    for t in agent.spec.tools:
        if t.name == "workspace_task":
            t.permissions = [*t.permissions, "investigate:read"]
    assert "workspace_task.investigate:read" not in agent.spec.approvals.require_for

    inv = await gateway.invoke(
        agent,
        "workspace_task.investigate:read",
        args={"repo": "a/b", "question": "why does parse return None?"},
        requested_by="tester",
    )

    # Ungated: reached execute() directly, never parked for approval.
    assert inv.status == "executed"
    assert inv.result is not None
    assert inv.result.ok is True

    outcome = inv.result.data["outcome"]
    assert outcome["kind"] == "evidence_bundle"
    assert outcome["findings"].strip() != ""

    # Zero PRs opened, zero issues created — a read-only profile really is
    # read-only over the shared contract, not merely by convention.
    assert fake_github.pulls == []
    assert fake_github.created == []

    # Budget telemetry is sane (present and matches the faked model
    # response) — cheap to assert, and it shows the same cost/token plumbing
    # code:write uses also carries investigate:read's numbers through.
    assert inv.result.data["cost_usd"] == 0.01
    assert inv.result.data["prompt_tokens"] == 42
    assert inv.result.data["completion_tokens"] == 9

    # provision_readonly was actually invoked (not skipped) for the right repo.
    assert fake_orchestrator.readonly_provisions == [("a/b", None)]
