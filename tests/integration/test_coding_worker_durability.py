"""Integration: Phase B durability — per-step checkpoints + crash-resume.

The connector persists a checkpoint after each named step and resumes from it,
making the two durable side effects (branch push, PR open) idempotent so a replay
never duplicates them.
"""

from openloop.checkpoints import InMemoryCheckpointStore
from openloop.tools.coding_worker import (
    STEPS,
    CodingWorkerConnector,
    WorkerOutcome,
)
from openloop.testing import FakeGitHub


class CountingRunner:
    """Walks all attempt steps and records how many times it actually ran."""

    def __init__(self) -> None:
        self.runs = 0

    async def run_attempt(self, state, on_step=None):
        self.runs += 1
        for step in STEPS:
            state.completed_steps.append(step)
            if on_step is not None:
                await on_step(state)
        state.title, state.body = "Add retries", "Adds retry logic."
        return WorkerOutcome(branch=state.branch, title="Add retries", body="Adds retry logic.")


class RecordingStore(InMemoryCheckpointStore):
    """Records every upsert so per-step persistence can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[tuple[str, list[str]]] = []

    async def upsert(self, checkpoint) -> None:
        self.history.append((checkpoint.status, list(checkpoint.completed_steps)))
        await super().upsert(checkpoint)


def _args(job_id="j1"):
    return {"repo": "acme/x", "instruction": "add retries", "job_id": job_id}


async def test_checkpoint_persisted_after_each_step():
    store = RecordingStore()
    conn = CodingWorkerConnector(CountingRunner(), FakeGitHub(), checkpoints=store)

    result = await conn.execute("pr:write", _args())
    assert result.ok

    running = [steps for status, steps in store.history if status == "running"]
    # Started clean, then grew one step at a time up to the full set.
    assert running[0] == []
    assert ["clone"] in running
    assert list(STEPS) in running
    # Terminal record is the opened PR.
    assert store.history[-1][0] == "opened"
    final = await store.get("j1")
    assert final.status == "opened"
    assert final.pr_number == 1


async def test_reinvoke_after_open_is_idempotent_noop():
    store = InMemoryCheckpointStore()
    runner = CountingRunner()
    github = FakeGitHub()
    conn = CodingWorkerConnector(runner, github, checkpoints=store)

    first = await conn.execute("pr:write", _args())
    second = await conn.execute("pr:write", _args())

    assert first.ok and second.ok
    assert second.data.get("resumed") is True
    assert second.data["pr_number"] == first.data["pr_number"]
    # The worker ran once and exactly one PR exists.
    assert runner.runs == 1
    assert len(github.pulls) == 1


async def test_resume_after_crash_between_push_and_pr_open():
    class FlakyGitHub(FakeGitHub):
        def __init__(self):
            super().__init__()
            self.fail_next = True

        async def create_pull(self, *a, **k):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("network blip opening PR")
            return await super().create_pull(*a, **k)

    store = InMemoryCheckpointStore()
    runner = CountingRunner()
    github = FlakyGitHub()
    conn = CodingWorkerConnector(runner, github, checkpoints=store)

    # First attempt: worker pushes, but opening the PR fails.
    first = await conn.execute("pr:write", _args())
    assert not first.ok
    assert first.data["status"] == "open_pr_failed"
    assert github.pulls == []
    cp = await store.get("j1")
    assert cp.status == "open_pr_failed"
    assert "push" in cp.completed_steps

    # Resume: worker is NOT re-run (branch already pushed); the PR opens once.
    second = await conn.execute("pr:write", _args())
    assert second.ok
    assert runner.runs == 1  # not re-run
    assert len(github.pulls) == 1
    assert (await store.get("j1")).status == "opened"


def _seed_checkpoint(job_id, status, steps, *, base="main", branch=None):
    """Insert a checkpoint as if a prior run had reached `status`."""
    from openloop.checkpoints import WorkerCheckpoint

    branch = branch or f"openloop/job-{job_id}"
    cp = WorkerCheckpoint(
        job_id=job_id, repo="acme/x", instruction="add retries", base=base,
        branch=branch, status=status, completed_steps=list(steps),
        state_json={
            "job_id": job_id, "repo": "acme/x", "instruction": "add retries",
            "base": base, "branch": branch, "completed_steps": list(steps),
            "title": "Add retries", "body": "b",
        },
        title="Add retries", body="b",
    )
    return cp


async def test_resume_uses_checkpoint_base_not_args():
    # A job opened against a non-default base; resume passes only job_id.
    store = InMemoryCheckpointStore()
    runner = CountingRunner()
    github = FakeGitHub()
    conn = CodingWorkerConnector(runner, github, checkpoints=store)

    await store.upsert(
        _seed_checkpoint("j1", "open_pr_failed", STEPS, base="develop")
    )
    # Note: no "base" in args — must not fall back to "main".
    result = await conn.execute("pr:write", {"job_id": "j1"})

    assert result.ok
    assert runner.runs == 0  # branch already pushed
    assert github.pulls[0]["base"] == "develop"


async def test_resume_before_push_reruns_worker_and_completes():
    # P2 window: crash after push but before the "push" checkpoint write leaves the
    # checkpoint at "commit". Resume re-runs the worker (force-push makes the second
    # push safe) and completes the PR.
    store = InMemoryCheckpointStore()
    runner = CountingRunner()
    github = FakeGitHub()
    conn = CodingWorkerConnector(runner, github, checkpoints=store)

    await store.upsert(
        _seed_checkpoint("j1", "running", ["clone", "branch", "edit", "commit"])
    )
    result = await conn.execute("pr:write", {"job_id": "j1", "repo": "acme/x",
                                             "instruction": "add retries"})

    assert result.ok
    assert runner.runs == 1  # re-ran the local pipeline
    assert len(github.pulls) == 1
    assert (await store.get("j1")).status == "opened"


async def test_reconciler_resumes_only_non_terminal_jobs():
    # P1: resolve() won't re-invoke execute() after a crash, so the startup
    # reconciler is what actually triggers resume.
    store = InMemoryCheckpointStore()
    runner = CountingRunner()
    github = FakeGitHub()
    conn = CodingWorkerConnector(runner, github, checkpoints=store)

    await store.upsert(_seed_checkpoint("run1", "running", ["clone", "branch"]))
    await store.upsert(_seed_checkpoint("push1", "pushed", STEPS))
    await store.upsert(_seed_checkpoint("fail1", "open_pr_failed", STEPS))
    await store.upsert(_seed_checkpoint("done1", "opened", STEPS))
    await store.upsert(_seed_checkpoint("dead1", "failed", ["clone"]))

    resumed = await conn.resume_incomplete()

    assert set(resumed) == {"run1", "push1", "fail1"}  # terminal ones skipped
    # The two already-pushed jobs just open PRs; only the running one re-runs.
    assert runner.runs == 1
    assert {p["head"] for p in github.pulls} == {
        "openloop/job-run1", "openloop/job-push1", "openloop/job-fail1"
    }
    for job_id in ("run1", "push1", "fail1"):
        assert (await store.get(job_id)).status == "opened"
    assert (await store.get("done1")).status == "opened"  # untouched


async def test_existing_pr_is_reused_not_duplicated():
    # Checkpoint says branch pushed but the "opened" record was lost; a PR already
    # exists on GitHub. Resume must reuse it via find_pull, not open a second.
    store = InMemoryCheckpointStore()
    runner = CountingRunner()
    github = FakeGitHub()
    conn = CodingWorkerConnector(runner, github, checkpoints=store)

    # Seed a pushed-but-not-opened checkpoint and a pre-existing PR for the head.
    branch = "openloop/job-j1"
    await github.create_pull("acme/x", branch, "main", "Add retries", "b", True)
    from openloop.checkpoints import WorkerCheckpoint

    await store.upsert(
        WorkerCheckpoint(
            job_id="j1", repo="acme/x", instruction="add retries", base="main",
            branch=branch, status="open_pr_failed",
            completed_steps=list(STEPS),
            state_json={
                "job_id": "j1", "repo": "acme/x", "instruction": "add retries",
                "base": "main", "branch": branch, "completed_steps": list(STEPS),
                "title": "Add retries", "body": "b",
            },
            title="Add retries", body="b",
        )
    )

    result = await conn.execute("pr:write", _args())
    assert result.ok
    assert runner.runs == 0  # branch already pushed
    assert len(github.pulls) == 1  # reused, not duplicated
