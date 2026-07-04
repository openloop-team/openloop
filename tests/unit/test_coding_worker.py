"""Unit tests for the coding-worker connector + orchestrator (network-free)."""

from pathlib import Path

import pytest

from openloop.credentials import EnvCredentialResolver
from openloop.tools.coding_worker import (
    STEPS,
    CodingWorkerConnector,
    BuiltinCodingWorker,
    GitWorkspaceOrchestrator,
    WorkerOutcome,
    WorkerState,
    _basic_auth,
    _parse_generation,
    _pr_body,
    _redact,
)
from openloop.testing import FakeCodingWorker, FakeGitHub, FakeWorkerOrchestrator


def _connector(runner=None, github=None):
    return CodingWorkerConnector(
        runner or FakeWorkerOrchestrator(), github or FakeGitHub()
    )


def _state(job_id="j1"):
    return WorkerState(
        job_id=job_id, repo="a/b", instruction="x", base="main",
        branch=f"openloop/job-{job_id}",
    )


def test_supported_permission():
    assert _connector().supported_permissions() == {"pr:write"}


def test_prepare_args_mints_job_id_once():
    conn = _connector()
    args = conn.prepare_args("pr:write", {"repo": "a/b", "instruction": "do x"})
    assert args["job_id"]
    # Idempotent: an existing job_id is preserved (replay across approval).
    again = conn.prepare_args("pr:write", {**args})
    assert again["job_id"] == args["job_id"]


def test_prepare_args_stamps_the_invoking_agent():
    from openloop.agents import load_agent
    from openloop.testing import EXAMPLE_AGENT

    conn = _connector()
    agent = load_agent(EXAMPLE_AGENT)
    # A model-supplied "agent" arg must never redirect spend attribution —
    # the gateway-passed identity wins unconditionally.
    args = conn.prepare_args(
        "pr:write",
        {"repo": "a/b", "instruction": "do x", "agent": "spoofed"},
        agent,
    )
    assert args["agent"] == "dev-platform"


def test_worker_state_roundtrips_agent_and_tolerates_old_checkpoints():
    state = _state()
    state.agent = "docs-bot"
    assert WorkerState.from_dict(state.to_dict()).agent == "docs-bot"
    # A pre-Phase 5 checkpoint has no agent key: attribution falls back.
    old = {k: v for k, v in state.to_dict().items() if k != "agent"}
    assert WorkerState.from_dict(old).agent is None


async def test_execute_runs_attempt_then_opens_draft_pr():
    runner = FakeWorkerOrchestrator(title="Add retries", body="Adds retry logic.")
    github = FakeGitHub()
    conn = _connector(runner, github)

    result = await conn.execute(
        "pr:write",
        {"repo": "acme/x", "instruction": "add retries", "job_id": "job123"},
    )

    assert result.ok
    # The attempt walked every named step.
    assert runner.runs[0].completed_steps == list(STEPS)
    # A draft PR was opened from the job branch.
    assert github.pulls == [
        {
            "number": 1,
            "repo": "acme/x",
            "head": "openloop/job-job123",
            "base": "main",
            "title": "Add retries",
            "body": _pr_body("Adds retry logic.", "job123"),
            "draft": True,
            "html_url": "https://github.com/acme/x/pull/1",
        }
    ]
    # job_id threads through the outcome + idempotency keys.
    assert result.data["job_id"] == "job123"
    assert result.data["pr_url"] == "https://github.com/acme/x/pull/1"
    assert result.data["idempotency_keys"] == {
        "push": "job123:push:openloop/job-job123",
        "open_pr": "job123:open_pr:acme/x:openloop/job-job123",
    }


async def test_pr_body_stamps_job_id():
    body = _pr_body("Some change", "abc123")
    assert "Some change" in body
    assert "job `abc123`" in body


async def test_attempt_failure_records_outcome_without_opening_pr():
    class BoomRunner:
        async def run_attempt(self, state, on_step=None):
            state.completed_steps.append("clone")
            raise RuntimeError("clone failed")

    github = FakeGitHub()
    conn = _connector(BoomRunner(), github)
    result = await conn.execute(
        "pr:write", {"repo": "a/b", "instruction": "x", "job_id": "j1"}
    )

    assert not result.ok
    assert result.data["status"] == "failed"
    assert result.data["error"] == "clone failed"
    assert result.data["completed_steps"] == ["clone"]
    assert github.pulls == []  # no PR on failure


async def test_open_pr_failure_records_outcome_without_crashing():
    # create_pull runs after resolve() marked the approval approved, so a GitHub
    # rejection must come back as a failed ToolResult, not bubble out of execute.
    class BoomGitHub(FakeGitHub):
        async def create_pull(self, *a, **k):
            raise RuntimeError("422 pull request already exists")

    runner = FakeWorkerOrchestrator()
    github = BoomGitHub()
    conn = _connector(runner, github)
    result = await conn.execute(
        "pr:write", {"repo": "a/b", "instruction": "x", "job_id": "j1"}
    )

    assert not result.ok
    assert result.data["status"] == "open_pr_failed"
    assert "422" in result.data["error"]
    # The attempt still ran (branch pushed); resume can reopen the PR later.
    assert result.data["branch"] == "openloop/job-j1"
    assert list(STEPS) == result.data["completed_steps"]


async def test_result_surfaces_worker_model_spend():
    runner = FakeWorkerOrchestrator(
        cost_usd=0.12, prompt_tokens=100, completion_tokens=50
    )
    conn = _connector(runner, FakeGitHub())
    result = await conn.execute(
        "pr:write", {"repo": "a/b", "instruction": "x", "job_id": "j2"}
    )
    assert result.data["cost_usd"] == 0.12
    assert result.data["prompt_tokens"] == 100
    assert result.data["completion_tokens"] == 50


def _orchestrator(worker=None, resolver=None):
    return GitWorkspaceOrchestrator(
        worker or FakeCodingWorker(),
        resolver or EnvCredentialResolver({"github": "secrettoken"}),
    )


async def test_orchestrator_run_redacts_secrets_from_command_and_stderr():
    orch = _orchestrator()
    with pytest.raises(RuntimeError) as excinfo:
        # stderr echoes the secret; the failing command also carries it.
        await orch._run(
            "python",
            "-c",
            "import sys; sys.stderr.write('fatal: url secrettoken'); sys.exit(1)",
            "secrettoken",
            redact="secrettoken",
        )
    message = str(excinfo.value)
    assert "secrettoken" not in message
    assert "***" in message


async def test_attempt_provisions_edits_commits_and_pushes(monkeypatch):
    """The orchestrator owns every git op; the worker only edits the prepared
    workspace. Auth rides an ephemeral header — never a token-in-URL — and the
    push re-resolves the credential (App tokens can expire mid-attempt)."""

    class RotatingResolver:
        def __init__(self):
            self.calls = 0

        async def resolve(self, scope):
            self.calls += 1
            return f"tok{self.calls}"

    worker = FakeCodingWorker(title="t", body="b")
    orch = GitWorkspaceOrchestrator(worker, RotatingResolver())
    commands = []

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        commands.append((cmd, redact))
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    state = _state()
    outcome = await orch.run_attempt(state)

    assert outcome.branch == "openloop/job-j1"
    assert outcome.title == "t"
    assert state.completed_steps == list(STEPS)

    clone_cmd, clone_redact = next(
        (cmd, r) for cmd, r in commands if "clone" in cmd
    )
    push_cmd, push_redact = next(
        (cmd, r) for cmd, r in commands if "push" in cmd
    )
    # No token-in-URL anywhere: the raw token appears in no command argument
    # except inside the one-shot auth header.
    for cmd, _ in commands:
        for arg in cmd:
            if arg.startswith("http.extraHeader="):
                continue
            assert "tok1" not in arg and "tok2" not in arg
    # Clone authenticates with the first mint, push with a fresh one.
    assert any(_basic_auth("tok1") in arg for arg in clone_cmd)
    assert any(_basic_auth("tok2") in arg for arg in push_cmd)
    assert "--force" in push_cmd  # idempotent retry to the job branch
    # Failure output scrubs both the raw token and its basic-auth encoding.
    assert clone_redact == ("tok1", _basic_auth("tok1"))
    assert push_redact == ("tok2", _basic_auth("tok2"))
    # Commit message comes from the worker's edit.
    commit_cmd = next(cmd for cmd, _ in commands if "commit" in cmd)
    assert "t" in commit_cmd


async def test_worker_sees_no_credential(monkeypatch):
    """Phase 2 exit criterion: nothing credential-shaped reaches the worker —
    not in its arguments and not via a token-in-URL clone."""

    class SpyWorker(FakeCodingWorker):
        def __init__(self):
            super().__init__()
            self.seen = None

        async def run(self, workspace, state, on_step=None):
            self.seen = (workspace, state, on_step)
            return await super().run(workspace, state, on_step)

    worker = SpyWorker()
    orch = GitWorkspaceOrchestrator(
        worker, EnvCredentialResolver({"github": "secrettoken"})
    )

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    await orch.run_attempt(_state())

    assert worker.seen is not None
    assert "secrettoken" not in repr(worker.seen)
    assert _basic_auth("secrettoken") not in repr(worker.seen)


async def test_workspace_root_is_honored(monkeypatch, tmp_path):
    """A containerized deploy pins workspaces to a host-shared dir so sibling
    sandbox containers can bind-mount them."""
    worker = FakeCodingWorker()
    root = tmp_path / "workspaces"
    orch = GitWorkspaceOrchestrator(
        worker,
        EnvCredentialResolver({"github": "t"}),
        workspace_root=root,
    )

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        return ""

    monkeypatch.setattr(orch, "_run", fake_run)
    await orch.run_attempt(_state())

    workspace, _ = worker.runs[0]
    assert workspace.parent == root
    assert not workspace.exists()  # removed after the attempt


def test_worker_holds_no_credential_attribute():
    """Phase 2 contract: the worker class has no credential to leak — the
    resolver lives only on the orchestrating boundary."""
    worker = BuiltinCodingWorker(model="m")
    assert "credential" not in repr(vars(worker)).lower()
    assert not hasattr(worker, "_credentials")

    orch = _orchestrator(worker)
    # The orchestrator stores the resolver seam, never a raw token.
    assert "secrettoken" not in repr(vars(orch))


def test_redact_scrubs_every_secret_in_a_tuple():
    text = "push to https://x:old@github.com failed; retried with new"
    assert _redact(text, ("old", "new")) == (
        "push to https://x:***@github.com failed; retried with ***"
    )
    assert _redact(text, None) == text


def test_worker_state_idempotency_keys_are_per_side_effect():
    state = _state()
    assert state.push_key() == "j1:push:openloop/job-j1"
    assert state.open_pr_key() == "j1:open_pr:a/b:openloop/job-j1"


async def test_git_worker_applies_diff_through_sandbox():
    class _StubCompleter:
        async def complete(self, model, messages, **kwargs):
            class R:
                text = (
                    "TITLE: t\nBODY: b\nDIFF:\n"
                    "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
                )
                cost_usd = 0.3
                prompt_tokens = 10
                completion_tokens = 5

            return R()

    class RecordingSandbox:
        def __init__(self):
            self.calls = []

        async def exec(self, workspace, *cmd, stdin=None):
            self.calls.append((workspace, cmd, stdin))
            return ""

    sandbox = RecordingSandbox()
    worker = BuiltinCodingWorker(model="m", gateway=_StubCompleter(), sandbox=sandbox)
    state = _state()
    edit = await worker.run(Path("/tmp/ws"), state)

    assert edit.title == "t" and edit.body == "b"
    assert edit.cost_usd == 0.3
    assert state.completed_steps == ["edit"]
    # The model-generated diff executed through the sandbox seam, nowhere else.
    (workspace, cmd, stdin) = sandbox.calls[0]
    assert str(workspace) == "/tmp/ws"
    assert cmd[:2] == ("git", "apply")
    assert stdin.startswith("--- a/x")


def test_parse_generation_splits_title_body_diff():
    text = (
        "TITLE: Fix typo\n"
        "BODY: corrects a spelling error\n"
        "DIFF:\n"
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-teh\n+the\n"
    )
    diff, title, body = _parse_generation(text)
    assert title == "Fix typo"
    assert body == "corrects a spelling error"
    assert diff.startswith("--- a/x")


def test_parse_generation_requires_a_diff():
    with pytest.raises(RuntimeError, match="no diff"):
        _parse_generation("TITLE: nothing\nBODY: empty\nDIFF:\n")
