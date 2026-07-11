"""The sealed analysis worker expressed as a durable workflow (Phase 2).

The approve → sealed-run → report flow becomes three steps, the same shape as
the coding worker's workflow:

1. ``await_approval`` — a **wait node**. The instance is started (parked here)
   when the action is held for approval; the approval event wakes it.
2. ``run_analysis`` — one **opaque replay unit** run by the shared
   :class:`~openloop.tools.analysis_worker.SealedAnalysisOrchestrator`:
   monthly gate → materialize inputs → sealed run → per-task settle →
   size-capped read-out → artifact-store write. The step stays ``resumable``
   (``resumable=False`` means *abandon before replay* in this engine), and
   re-drive safety comes from the orchestrator's durable attempt accounting,
   not from replaying computation: the persisted ``attempt_id`` makes a
   re-driven step settle any known spend and then refuse re-execution, so a
   crash mid-run converges to a terminal, fully-accounted outcome instead of a
   silent second model call.
3. ``store_result`` — no I/O of its own. The report body was already written to
   the artifact store (once, by the orchestrator, inside ``run_analysis``);
   this step only persists ``instance.result`` as ``{artifact_ref,
   prose_summary, spend}``. The workflow never posts to a surface — delivery
   belongs to the session runner, which dereferences the ref at delivery time.

``job_id`` is the workflow instance id, so the one identity threads
approval → workflow → input store → artifact store → deliverable ref.
"""

from __future__ import annotations

from openloop.tools.analysis_worker import AnalysisAttemptRunner, AnalysisState
from openloop.workflows.engine import Step, Workflow, WorkflowContext

WORKFLOW_NAME = "analysis_worker"

# The wait node's name doubles as the event the approval emits.
APPROVAL_EVENT = "await_approval"

REPORT_FILENAME = "report.md"


def _analysis_phase(completed_steps: list[str]) -> str:
    """A human 'still working' phrase for the run's latest milestone.

    Keyed on the most-advanced step reached (the list only grows), phrased as
    what happens *next*: after ``materialize`` completes the model is writing
    the program, after ``generate`` the sealed execution is running, and so on.
    """
    steps = set(completed_steps)
    if "store_artifact" in steps:
        return "is finalizing the report…"
    if "read_out" in steps:
        return "is storing the report…"
    if "execute" in steps:
        return "is reading the results…"
    if "generate" in steps:
        return "is running the sealed analysis…"
    if "materialize" in steps:
        return "is writing the analysis program…"
    return "is provisioning the inputs…"


def build_analysis_worker_workflow(orchestrator: AnalysisAttemptRunner) -> Workflow:
    """Build the analysis workflow bound to the shared orchestrator.

    Both durable paths (this workflow and the connector's engine-less
    ``execute()`` fallback) call the same :class:`AnalysisAttemptRunner`, so no
    path can reach the input/artifact stores or the spend gates except through
    the one sealed boundary.
    """

    async def run_analysis(ctx: WorkflowContext) -> None:
        s = ctx.state
        state = AnalysisState(
            job_id=s["job_id"],
            input_ref=s.get("input_ref") or "",
            instruction=s.get("instruction") or "",
            # The invoking agent, stamped into the approval args by
            # prepare_args — the ledger attributes spend to it.
            agent=s.get("agent"),
            # Minted before approval; the durable attempt key that makes a
            # re-driven step accounting-safe instead of a free re-execution.
            attempt_id=s.get("attempt_id"),
        )

        async def on_step(astate: AnalysisState) -> None:
            # Record a human progress phrase so the surface can show "still
            # working…" without knowing orchestrator-internal step names.
            s["progress"] = _analysis_phase(astate.completed_steps)
            await ctx.checkpoint()

        result = await orchestrator.run_analysis(state, on_step=on_step)
        if not result.ok:
            # Spend (if any) was settled inside the orchestrator before this
            # surfaced; failing the step is what makes the instance terminal.
            raise RuntimeError(result.error or "sealed analysis failed")
        s["attempt_id"] = result.attempt_id
        s["artifact_ref"] = result.artifact_ref
        s["prose_summary"] = result.prose_summary
        if result.run is not None:
            s["cost_usd"] = result.run.cost_usd
            s["prompt_tokens"] = result.run.prompt_tokens
            s["completion_tokens"] = result.run.completion_tokens

    async def store_result(ctx: WorkflowContext) -> None:
        s = ctx.state
        ctx.instance.result = {
            "job_id": s["job_id"],
            "status": "reported",
            "input_ref": s.get("input_ref"),
            "attempt_id": s.get("attempt_id"),
            # The session runner recognizes this shape, dereferences the ref
            # from the artifact store, and posts the report as an Artifact
            # deliverable straight from the ref — no second model call (the
            # locked bypass-M0b decision), and never the body in this record.
            "deliverable": "artifact",
            "artifact_ref": s["artifact_ref"],
            "prose_summary": s.get("prose_summary"),
            "artifact_title": f"Analysis report (job {s['job_id']})",
            "artifact_filename": REPORT_FILENAME,
            "snippet_type": "markdown",
            # Full spend telemetry, matching AnalysisWorkerConnector.execute().
            "cost_usd": s.get("cost_usd", 0.0),
            "prompt_tokens": s.get("prompt_tokens", 0),
            "completion_tokens": s.get("completion_tokens", 0),
            # One line for the approval-button reply; the report itself rides
            # the ref and the prose summary rides prose_summary.
            "summary": f"sealed analysis completed; report ready (job {s['job_id']})",
        }

    return Workflow(
        WORKFLOW_NAME,
        [
            Step(APPROVAL_EVENT, wait=True),
            # The opaque replay unit MUST stay resumable: resumable=False means
            # "abandon before replay" in this engine — the opposite of intent.
            Step("run_analysis", run_analysis, resumable=True),
            Step("store_result", store_result),
        ],
    )
