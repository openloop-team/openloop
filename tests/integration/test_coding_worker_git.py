"""Integration: the REAL orchestrator + worker against a local git remote.

Runs the actual git pipeline — shallow clone, branch, ``git apply``, commit,
force-push — against a ``file://`` bare repository, so the Phase 2 hardening
contract is verified mechanically, not just with fakes:

- the workspace handed to the worker contains **no credential anywhere**
  (``.git/config`` keeps the plain URL; auth rides a per-command header);
- the pushed branch lands in the remote with the worker's edit applied;
- a second attempt force-pushes idempotently.

Network-free: the model call is stubbed, the remote is on disk.
"""

import subprocess
from pathlib import Path

import pytest

from openloop.credentials import EnvCredentialResolver
from openloop.models.gateway import ModelResponse
from openloop.tools.coding_worker import (
    STEPS,
    BuiltinCodingWorker,
    GitWorkspaceOrchestrator,
    WorkerState,
)


def _git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture()
def remote(tmp_path: Path) -> Path:
    """A bare repo seeded with one commit on main, served over file://."""
    bare = tmp_path / "remotes" / "acme" / "x.git"
    bare.parent.mkdir(parents=True)
    _git("init", "--bare", "--initial-branch=main", str(bare))
    seed = tmp_path / "seed"
    _git("clone", str(bare), str(seed))
    (seed / "README.md").write_text("hello\n")
    _git("add", "-A", cwd=seed)
    _git("-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "seed", cwd=seed)
    _git("push", "origin", "HEAD:main", cwd=seed)
    return bare


class _StubCompleter:
    """Emits a slightly different edit per call, like a regenerated diff."""

    def __init__(self):
        self.calls = 0

    async def complete(self, model, messages, **kwargs):
        self.calls += 1
        greeting = f"hello world {self.calls}"
        diff = (
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1 +1 @@\n"
            "-hello\n"
            f"+{greeting}\n"
        )
        return ModelResponse(
            text=f"TITLE: Say hello world\nBODY: expands the greeting\nDIFF:\n{diff}",
            model="stub",
        )


def _orchestrator(remote: Path, worker=None):
    # remote is <base>/acme/x.git — repo "acme/x" resolves against file://<base>.
    remote_base = f"file://{remote.parent.parent}"
    return GitWorkspaceOrchestrator(
        worker or BuiltinCodingWorker(model="stub", gateway=_StubCompleter()),
        EnvCredentialResolver({"github": "secrettoken"}),
        remote_base=remote_base,
    )


def _state(job_id="j1"):
    return WorkerState(
        job_id=job_id, repo="acme/x", instruction="say hello world",
        base="main", branch=f"openloop/job-{job_id}",
    )


async def test_real_attempt_pushes_branch_with_edit(remote):
    orch = _orchestrator(remote)
    state = _state()

    outcome = await orch.run_attempt(state)

    assert outcome.title == "Say hello world"
    assert state.completed_steps == list(STEPS)
    # The branch landed in the remote with the worker's edit applied.
    assert "openloop/job-j1" in _git(
        "branch", "--list", "openloop/job-j1", cwd=remote
    )
    assert _git("show", "openloop/job-j1:README.md", cwd=remote) == "hello world 1\n"
    # Commit is authored by the worker bot identity.
    log = _git("log", "-1", "--format=%an <%ae>", "openloop/job-j1", cwd=remote)
    assert log.strip() == "OpenLoop coding worker <worker@openloop.team>"


async def test_workspace_handed_to_worker_holds_no_credential(remote):
    seen: dict = {}

    class SpyWorker(BuiltinCodingWorker):
        async def run(self, workspace, state, on_step=None):
            # Capture everything credential-shaped that COULD leak: the whole
            # git config visible from the workspace, plus every file in .git
            # that stores remote/auth settings.
            seen["config"] = _git("config", "--list", cwd=workspace)
            seen["origin"] = _git(
                "remote", "get-url", "origin", cwd=workspace
            ).strip()
            return await super().run(workspace, state, on_step)

    worker = SpyWorker(model="stub", gateway=_StubCompleter())
    orch = _orchestrator(remote, worker)

    await orch.run_attempt(_state("j2"))

    assert seen, "spy worker never ran"
    assert "secrettoken" not in seen["config"]
    assert "AUTHORIZATION" not in seen["config"]
    assert "extraheader" not in seen["config"].lower()
    # The origin URL is the plain remote — no token-in-URL clone.
    assert seen["origin"].startswith("file://")
    assert "secrettoken" not in seen["origin"]


async def test_second_attempt_force_pushes_idempotently(remote):
    orch = _orchestrator(remote)

    await orch.run_attempt(_state("j3"))
    first = _git("rev-parse", "openloop/job-j3", cwd=remote).strip()
    # A resumed job runs a fresh attempt; the force-push must not be rejected
    # as a non-fast-forward even though the branch already exists.
    await orch.run_attempt(_state("j3"))
    second = _git("rev-parse", "openloop/job-j3", cwd=remote).strip()

    assert first != second  # regenerated attempt replaced the branch
    assert _git("show", "openloop/job-j3:README.md", cwd=remote) == "hello world 2\n"


class _RecordingOrchestrator(GitWorkspaceOrchestrator):
    """Records every git subcommand so a test can prove warm reuse skips clone."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.commands: list[tuple[str, ...]] = []

    async def _run(self, *cmd, cwd=None, stdin=None, redact=None):
        self.commands.append(cmd)
        return await super()._run(*cmd, cwd=cwd, stdin=stdin, redact=redact)


def _count(commands, sub):
    return sum(1 for c in commands if sub in c)


async def test_warm_reuse_skips_clone_and_lands_edit(remote, tmp_path):
    from openloop.tools.workspace_pool import WarmHandle, WarmWorkspacePool

    refs: list = []

    async def sink(key, ref):
        refs.append((key, ref))

    pool = WarmWorkspacePool(root=tmp_path / "warm", on_change=sink)
    orch = _RecordingOrchestrator(
        BuiltinCodingWorker(model="stub", gateway=_StubCompleter()),
        EnvCredentialResolver({"github": "secrettoken"}),
        remote_base=f"file://{remote.parent.parent}",
        warm_pool=pool,
    )

    # Two turns of the SAME thread — new job each, but one warm checkout.
    s1 = _state("w1")
    s1.warm_key = "thread-A"
    await orch.run_attempt(s1)
    s2 = _state("w2")
    s2.warm_key = "thread-A"
    await orch.run_attempt(s2)

    # Exactly one clone (the first, cold) — the second reused via fetch.
    assert _count(orch.commands, "clone") == 1
    assert _count(orch.commands, "fetch") >= 1
    # Both attempts pushed their branch, and the warm tree was reset to base so
    # the second diff applied cleanly against `hello` (not the first edit).
    assert _git("show", "openloop/job-w1:README.md", cwd=remote) == "hello world 1\n"
    assert _git("show", "openloop/job-w2:README.md", cwd=remote) == "hello world 2\n"
    # The thread's warm context_ref was persisted (a live handle for the repo).
    live = [r for k, r in refs if k == "thread-A" and r is not None]
    assert live and WarmHandle.from_json(live[0]).repo == "acme/x"

    await pool.shutdown()


async def test_warm_discard_on_failure_cold_starts_next(remote, tmp_path):
    from openloop.tools.workspace_pool import WarmWorkspacePool

    # A worker that raises → the attempt fails → the checkout is discarded.
    class _Boom(BuiltinCodingWorker):
        async def run(self, workspace, state, on_step=None):
            raise RuntimeError("worker blew up")

    pool = WarmWorkspacePool(root=tmp_path / "warm")
    orch = _RecordingOrchestrator(
        _Boom(model="stub", gateway=_StubCompleter()),
        EnvCredentialResolver({"github": "secrettoken"}),
        remote_base=f"file://{remote.parent.parent}",
        warm_pool=pool,
    )
    s1 = _state("f1")
    s1.warm_key = "thread-B"
    with pytest.raises(RuntimeError):
        await orch.run_attempt(s1)

    # The failed checkout was dropped, so a good follow-up cold-clones again.
    orch2 = _RecordingOrchestrator(
        BuiltinCodingWorker(model="stub", gateway=_StubCompleter()),
        EnvCredentialResolver({"github": "secrettoken"}),
        remote_base=f"file://{remote.parent.parent}",
        warm_pool=pool,
    )
    s2 = _state("f2")
    s2.warm_key = "thread-B"
    await orch2.run_attempt(s2)
    assert _count(orch2.commands, "clone") == 1  # no warm entry to reuse
    assert _count(orch2.commands, "fetch") == 0
    await pool.shutdown()
