"""Unit tests for the coding-worker connector (network-free, with fakes)."""

import pytest

from openloop.credentials import EnvCredentialResolver
from openloop.tools.coding_worker import (
    STEPS,
    CodingWorkerConnector,
    GitCodingWorker,
    WorkerOutcome,
    WorkerState,
    _parse_generation,
    _pr_body,
    _redact,
)
from openloop.testing import FakeCodingWorker, FakeGitHub


def _connector(worker=None, github=None):
    return CodingWorkerConnector(worker or FakeCodingWorker(), github or FakeGitHub())


def test_supported_permission():
    assert _connector().supported_permissions() == {"pr:write"}


def test_prepare_args_mints_job_id_once():
    conn = _connector()
    args = conn.prepare_args("pr:write", {"repo": "a/b", "instruction": "do x"})
    assert args["job_id"]
    # Idempotent: an existing job_id is preserved (replay across approval).
    again = conn.prepare_args("pr:write", {**args})
    assert again["job_id"] == args["job_id"]


async def test_execute_runs_worker_then_opens_draft_pr():
    worker = FakeCodingWorker(title="Add retries", body="Adds retry logic.")
    github = FakeGitHub()
    conn = _connector(worker, github)

    result = await conn.execute(
        "pr:write",
        {"repo": "acme/x", "instruction": "add retries", "job_id": "job123"},
    )

    assert result.ok
    # Worker walked every named step.
    assert worker.runs[0].completed_steps == list(STEPS)
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


async def test_worker_failure_records_outcome_without_opening_pr():
    class BoomWorker:
        async def run(self, state, on_step=None):
            state.completed_steps.append("clone")
            raise RuntimeError("clone failed")

    github = FakeGitHub()
    conn = _connector(BoomWorker(), github)
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

    worker = FakeCodingWorker()
    github = BoomGitHub()
    conn = _connector(worker, github)
    result = await conn.execute(
        "pr:write", {"repo": "a/b", "instruction": "x", "job_id": "j1"}
    )

    assert not result.ok
    assert result.data["status"] == "open_pr_failed"
    assert "422" in result.data["error"]
    # The worker still ran (branch pushed) — Phase A has no resume, so it's left.
    assert result.data["branch"] == "openloop/job-j1"
    assert list(STEPS) == result.data["completed_steps"]


async def test_result_surfaces_worker_model_spend():
    class CostingWorker:
        async def run(self, state, on_step=None):
            state.completed_steps.extend(STEPS)
            return WorkerOutcome(
                branch=state.branch, title="t", body="b",
                cost_usd=0.12, prompt_tokens=100, completion_tokens=50,
            )

    conn = _connector(CostingWorker(), FakeGitHub())
    result = await conn.execute(
        "pr:write", {"repo": "a/b", "instruction": "x", "job_id": "j2"}
    )
    assert result.data["cost_usd"] == 0.12
    assert result.data["prompt_tokens"] == 100
    assert result.data["completion_tokens"] == 50


async def test_git_run_redacts_token_from_command_and_stderr():
    worker = GitCodingWorker(EnvCredentialResolver({"github": "s"}), model="m")
    with pytest.raises(RuntimeError) as excinfo:
        # stderr echoes the token (as git does in remote URLs); the failing
        # command also carries it as an argument.
        await worker._run(
            "python",
            "-c",
            "import sys; sys.stderr.write('fatal: url secrettoken'); sys.exit(1)",
            "secrettoken",
            redact="secrettoken",
        )
    message = str(excinfo.value)
    assert "secrettoken" not in message
    assert "***" in message


async def test_push_re_resolves_token_and_bypasses_stale_origin(monkeypatch):
    """A long run can outlive the clone-time token (App tokens expire): the
    push must carry a freshly resolved token in an explicit URL, not rely on
    the stale one baked into origin, and redact both."""

    class RotatingResolver:
        def __init__(self):
            self.calls = 0

        async def resolve(self, scope):
            self.calls += 1
            return f"tok{self.calls}"

    class _StubCompleter:
        async def complete(self, model, messages, **kwargs):
            class R:
                text = (
                    "TITLE: t\nBODY: b\nDIFF:\n"
                    "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
                )
                cost_usd = 0.0
                prompt_tokens = 1
                completion_tokens = 1

            return R()

    worker = GitCodingWorker(
        RotatingResolver(), model="m", gateway=_StubCompleter()
    )
    commands = []

    async def fake_run(*cmd, cwd=None, stdin=None, redact=None):
        commands.append((cmd, redact))
        return ""

    monkeypatch.setattr(worker, "_run", fake_run)
    state = WorkerState(
        job_id="j1", repo="a/b", instruction="x", base="main",
        branch="openloop/job-j1",
    )
    await worker.run(state)

    clone_cmd, clone_redact = commands[0]
    assert "tok1" in clone_cmd[6]  # clone URL carries the first token
    push_cmd, push_redact = next(
        (cmd, redact) for cmd, redact in commands if cmd[1] == "push"
    )
    assert "origin" not in push_cmd
    assert "tok2" in push_cmd[3]  # explicit URL with the fresh token
    assert push_redact == ("tok1", "tok2")  # both tokens scrubbed on failure


def test_redact_scrubs_every_secret_in_a_tuple():
    text = "push to https://x:old@github.com failed; retried with new"
    assert _redact(text, ("old", "new")) == (
        "push to https://x:***@github.com failed; retried with ***"
    )
    assert _redact(text, None) == text


def test_worker_stores_no_raw_token_attribute():
    """Phase 1 contract: the credential lives behind the resolver, never on
    the worker."""
    worker = GitCodingWorker(
        EnvCredentialResolver({"github": "secrettoken"}), model="m"
    )
    assert "secrettoken" not in repr(vars(worker))


def test_worker_state_idempotency_keys_are_per_side_effect():
    state = WorkerState(
        job_id="j1", repo="a/b", instruction="x", base="main", branch="openloop/job-j1"
    )
    assert state.push_key() == "j1:push:openloop/job-j1"
    assert state.open_pr_key() == "j1:open_pr:a/b:openloop/job-j1"


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
