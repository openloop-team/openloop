"""The coding worker expressed as a durable workflow (Phase C consumer #1).

The whole approve → run-worker → open-draft-PR flow becomes three steps:

1. ``await_approval`` — a **wait node**. The instance is started (parked here)
   when the action is held for approval; the approval event wakes it.
2. ``run_worker`` — clone → model-edit → commit → push (idempotent: force-push to
   the job-exclusive branch makes a resumed re-run safe).
3. ``open_pr`` — open the draft PR, reusing an existing one for the head branch.

This is the same logic as the Phase B connector, but its durability now comes from
the generic :class:`WorkflowEngine` (checkpoint per step, park/resume on the
approval event) instead of a worker-specific checkpoint + reconciler. ``job_id``
is the workflow instance id, so the one identity threads approval → workflow → PR.
"""

from __future__ import annotations

from openloop.tools.coding_worker import (
    CodingWorker,
    WorkerState,
    _branch_for,
    _pr_body,
)
from openloop.tools.github import GitHubClient
from openloop.workflows.engine import Step, Workflow, WorkflowContext

WORKFLOW_NAME = "coding_worker"

# The wait node's name doubles as the event the approval emits.
APPROVAL_EVENT = "await_approval"


def build_coding_worker_workflow(
    worker: CodingWorker, github: GitHubClient
) -> Workflow:
    """Build the coding-worker workflow bound to a worker + GitHub client."""

    async def run_worker(ctx: WorkflowContext) -> None:
        s = ctx.state
        state = WorkerState(
            job_id=s["job_id"],
            repo=s["repo"],
            instruction=s["instruction"],
            base=s.get("base", "main"),
            branch=_branch_for(s["job_id"]),
        )
        outcome = await worker.run(state)
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
            Step("run_worker", run_worker),
            Step("open_pr", open_pr),
        ],
    )
