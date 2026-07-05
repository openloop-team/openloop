"""The coding worker expressed as a durable workflow (Phase C consumer #1).

The whole approve → run-worker → open-draft-PR flow becomes three steps:

1. ``await_approval`` — a **wait node**. The instance is started (parked here)
   when the action is held for approval; the approval event wakes it.
2. ``run_worker`` — one **opaque replay unit** run by the shared
   :class:`~openloop.tools.coding_worker.GitWorkspaceOrchestrator`: provision an
   ephemeral workspace → credential-free worker edit → commit → force-push to
   the job-exclusive branch. Provision lives *inside* the unit, never as its own
   durable step — the engine skips completed steps on re-drive, and a workspace
   provisioned before a crash no longer exists. The step stays ``resumable``
   (the engine's ``resumable=False`` means *abandon before replay*): a resume
   runs a fresh attempt; the force-push keeps the retry idempotent.
3. ``open_pr`` — open the draft PR, reusing an existing one for the head branch.

This is the same logic as the connector's checkpoint fallback, but its durability
comes from the generic :class:`WorkflowEngine` (checkpoint per step, park/resume
on the approval event) instead of a worker-specific checkpoint + reconciler.
``job_id`` is the workflow instance id, so the one identity threads
approval → workflow → PR.
"""

from __future__ import annotations

from openloop.tools.coding_worker import (
    AttemptRunner,
    WorkerState,
    _branch_for,
    _pr_body,
)
from openloop.tools.github import GitHubClient
from openloop.workflows.engine import Step, Workflow, WorkflowContext

WORKFLOW_NAME = "coding_worker"

# The wait node's name doubles as the event the approval emits.
APPROVAL_EVENT = "await_approval"


def _worker_phase(completed_steps: list[str]) -> str:
    """A human 'still working' phrase for the worker's latest milestone.

    Reads the most-advanced step reached (the list only grows). During the long
    agent-edit phase the last completed step is still ``branch`` (the worker
    appends ``edit`` only once the agent finishes), so that maps to the generic
    "working on the changes" — the phase the user waits on longest.
    """
    steps = set(completed_steps)
    if "push" in steps:
        return "is pushing the branch…"
    if "commit" in steps:
        return "is committing the changes…"
    if "edit" in steps:
        return "is finalizing the changes…"
    if "branch" in steps:
        return "is working on the changes…"
    if "clone" in steps:
        return "is setting up the workspace…"
    return "is starting…"


def build_coding_worker_workflow(
    orchestrator: AttemptRunner, github: GitHubClient
) -> Workflow:
    """Build the coding-worker workflow bound to the shared orchestrator.

    Both durable paths call the same :class:`AttemptRunner` — no path invokes a
    worker that could hold git credentials (hardening Phase 2 invariant).
    """

    async def run_worker(ctx: WorkflowContext) -> None:
        s = ctx.state
        state = WorkerState(
            job_id=s["job_id"],
            repo=s["repo"],
            instruction=s["instruction"],
            base=s.get("base", "main"),
            branch=_branch_for(s["job_id"]),
            # The invoking agent, stamped into the approval args by the
            # gateway (Phase 5) — the ledger attributes spend to it.
            agent=s.get("agent"),
        )
        async def on_step(ws: WorkerState) -> None:
            # Record a human progress phrase so the surface can show "still
            # working…" without knowing worker-internal step names.
            s["progress"] = _worker_phase(ws.completed_steps)
            await ctx.checkpoint()

        outcome = await orchestrator.run_attempt(state, on_step=on_step)
        s["branch"] = outcome.branch
        s["title"] = outcome.title
        s["body"] = outcome.body
        s["cost_usd"] = outcome.cost_usd
        s["prompt_tokens"] = outcome.prompt_tokens
        s["completion_tokens"] = outcome.completion_tokens

    async def open_pr(ctx: WorkflowContext) -> None:
        s = ctx.state
        base = s.get("base", "main")
        existing = await github.find_pull(s["repo"], head=s["branch"])
        pull = existing or await github.create_pull(
            repo=s["repo"],
            head=s["branch"],
            base=base,
            title=s["title"],
            body=_pr_body(s["body"], s["job_id"]),
            draft=True,
        )
        ctx.instance.result = {
            "job_id": s["job_id"],
            "status": "opened",
            "branch": s["branch"],
            "pr_number": pull.get("number"),
            "pr_url": pull.get("html_url"),
            # Full spend telemetry, matching CodingWorkerConnector.execute()'s data.
            "cost_usd": s.get("cost_usd", 0.0),
            "prompt_tokens": s.get("prompt_tokens", 0),
            "completion_tokens": s.get("completion_tokens", 0),
            "summary": (
                f"opened draft PR #{pull.get('number')} in {s['repo']} "
                f"(job {s['job_id']})"
            ),
        }

    return Workflow(
        WORKFLOW_NAME,
        [
            Step(APPROVAL_EVENT, wait=True),
            # The opaque replay unit MUST stay resumable: resumable=False means
            # "abandon before replay" in this engine — the opposite of intent.
            Step("run_worker", run_worker, resumable=True),
            Step("open_pr", open_pr),
        ],
    )
