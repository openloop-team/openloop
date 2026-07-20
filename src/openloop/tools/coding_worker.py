"""Native coding-worker connector — opens *draft* PRs from an instruction.

Exposes a single write action, ``coding_worker.pr:write``. On execution it runs
one **worker attempt** (provision a workspace → credential-free worker edit →
commit → push) and then opens a **draft** pull request from the pushed branch.

Two distinct gates, never conflated:

1. the human **approval** lets the worker **start**;
2. the **draft PR itself** is the review gate before **merge**.

There is no "approve the generated diff before opening the PR" step here — the
approval summary says *run worker + open draft PR* and must never imply diff
review.

``job_id`` is the stable thread through the whole system. It is minted in
:meth:`prepare_args` **before** the approval is created, so it is carried in the
approval args, the worker state, the branch name + idempotency keys, and the
final PR metadata — giving one identity through the whole system.

**Phase B — durability.** Given a :class:`CheckpointStore`, the connector
persists a checkpoint after each named step and resumes from it on a mid-flight
crash. The two durable side effects (branch push, PR open) are made idempotent so
a replay never duplicates them: a pushed branch is detected via ``completed_steps``
(the local workspace is ephemeral, so only the push survives a crash), and an
already-open PR is reused via :meth:`GitHubClient.find_pull`. Without a store the
connector behaves outcome-only (no resume).

**Hardening Phase 2 — credentials out of the worker.** The
:class:`CodingWorker` only edits a *prepared* workspace and holds no
credentials; every credential-bearing git operation (clone, branch, commit,
push) lives in :class:`GitWorkspaceOrchestrator` — the single orchestration
helper both durable paths (this connector's checkpoint fallback and the
workflow in :mod:`openloop.workflows.coding_worker`) call. Git auth rides an
ephemeral ``http.extraHeader`` on each command, so no token is ever written
into the workspace (no token-in-URL clone, nothing in ``.git/config``): the
worker sandbox is credential-free by construction. Legacy workers replay a
clean attempt before push; OpenHands cold resume restores only authenticated
pause/final boundaries and refuses to replay an interrupted active segment.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openloop.checkpoints.store import CheckpointStore, WorkerCheckpoint
from openloop.credentials import CredentialResolver, CredentialScope
from openloop.tools.base import ActionSpec, ToolResult
from openloop.tools.github import GitHubClient
from openloop.tools.openhands_resume import (
    OpenHandsResumeError,
    OpenHandsResumeState,
    ResumeDecision,
    WorkerPaused,
)

if TYPE_CHECKING:
    from openloop.sandbox import Sandbox
    from openloop.tools.openhands_resume import WorkspaceArtifactRef
    from openloop.tools.workspace_pool import WarmWorkspacePool
    from openloop.usage.ledger import WorkerSpendLedger

# Persist-after-each-step callback invoked after each completed step so a crash
# leaves an accurate mid-phase record (completed_steps + state_json), not just a
# status. The orchestrator owns the git-side steps; the worker reports its own.
StepCallback = Callable[["WorkerState"], Awaitable[None]]

logger = logging.getLogger(__name__)


class WorkerRunAborted(RuntimeError):
    """A worker stopped its own run early (in-run spend cap or deadline hit).

    Carries the spend accrued up to the abort so the orchestrator can still
    record it in the ledger before failing the attempt closed — the abort bounds
    spend to roughly the cap instead of running to completion, but whatever was
    spent before stopping is real and must land in the audit trail.
    """

    def __init__(
        self,
        reason: str,
        *,
        cost_usd: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cost_usd = cost_usd
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

CODING_WORKER_TOOL_NAME = "coding_worker"
CODING_WORKER_PR_WRITE = "pr:write"
CODING_WORKER_ARGS_VERSION = 1


class CodingWorkerPrArgs(BaseModel):
    """Model-facing args for ``coding_worker.pr:write`` (typed-tool-args §3).

    The declared schema is generated from this model and the gateway parses
    raw args through it, so normalization (strip) and constraints live in one
    artifact. Identity (job_id, agent, warm_key) is gateway-stamped after the
    parse and never appears here.
    """

    model_config = ConfigDict(extra="forbid")

    repo: str = Field(min_length=1, description="owner/repo, e.g. acme/ingestion")
    instruction: str = Field(
        min_length=1, description="what change the worker should make"
    )
    base: str | None = Field(
        default=None, description="branch to open the PR against (default main)"
    )

    @field_validator("repo", "instruction", mode="before")
    @classmethod
    def _strip(cls, value):
        return value.strip() if isinstance(value, str) else value

# Named steps one attempt walks through. The orchestrator owns the git-side
# steps (clone, branch, commit, push); the worker reports its own ("edit").
STEPS = ("clone", "branch", "edit", "commit", "push")


@dataclass(slots=True)
class WorkerState:
    """Worker progress for one job — serialized into a checkpoint's state_json.

    ``title`` / ``body`` are filled once the worker generates the change so they
    survive in the checkpoint; on resume after a push they let the PR be opened
    without re-running the worker.

    ``agent`` is the *invoking* agent's name (Phase 5), stamped by the gateway
    into the approval args and carried here so the spend ledger attributes the
    attempt — and enforces the budget — of whoever asked, on fresh runs and
    checkpoint resumes alike. ``None`` (pre-Phase 5 checkpoints) falls back to
    the ledger's default attribution.
    """

    job_id: str
    repo: str
    instruction: str
    base: str
    branch: str
    completed_steps: list[str] = field(default_factory=list)
    title: str | None = None
    body: str | None = None
    agent: str | None = None
    # Human who approved/initiated the durable worker on its surface. Cold
    # resume uses this stable identity to authorize later action decisions.
    requester_id: str | None = None
    # The approval that authorized this worker (attribution envelope, finding 4).
    # Gateway-stamped from the request id; carried so worker spend records trace
    # back to their authorization. ``None`` for pre-envelope checkpoints.
    approval_id: str | None = None
    # The originating surface session's id (attribution envelope, step 5).
    # Gateway-stamped from the invoking turn's session; carried so worker spend
    # records (UsageRecord.session_id) trace to the session that asked. ``None``
    # for sessionless paths (direct invoke, tests) and pre-envelope checkpoints.
    session_id: str | None = None
    # Warm-context key (Phase B): the requesting thread's durable scope key,
    # stamped by the gateway from the invoking turn. When a warm-workspace pool is
    # wired, the orchestrator reuses this thread's kept checkout instead of cloning
    # cold. Rides the checkpoint/workflow state so a resume re-warms the same
    # thread; ``None`` (a non-threaded turn, or warm context disabled) means the
    # unchanged ephemeral clone-and-discard path.
    warm_key: str | None = None
    # Per-attempt in-run spend ceiling (this agent's per-task cap), stamped by
    # the orchestrator before the worker runs so an agentic worker can stop
    # itself near the cap instead of blowing past it. Transient — recomputed each
    # attempt, deliberately not round-tripped through the checkpoint (from_dict).
    budget_usd: float | None = None
    # Versioned cold-resume facts. Legacy checkpoints omit this key entirely;
    # schema-first recovery must parse it before considering a legacy replay.
    openhands_resume: OpenHandsResumeState | None = None

    def push_key(self) -> str:
        """Idempotency key for the branch push — never a single global key."""
        return f"{self.job_id}:push:{self.branch}"

    def open_pr_key(self) -> str:
        """Idempotency key for opening the PR."""
        return f"{self.job_id}:open_pr:{self.repo}:{self.branch}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["openhands_resume"] = (
            self.openhands_resume.to_dict()
            if self.openhands_resume is not None
            else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerState":
        fields = {
            "job_id", "repo", "instruction", "base", "branch",
            "completed_steps", "title", "body", "agent", "warm_key",
            "requester_id", "approval_id", "session_id", "openhands_resume",
        }
        values = {k: v for k, v in data.items() if k in fields}
        resume = values.get("openhands_resume")
        if resume is not None:
            values["openhands_resume"] = OpenHandsResumeState.from_dict(resume)
        return cls(**values)


@dataclass(slots=True)
class WorkerOutcome:
    """What one worker attempt produced: a pushed branch ready for a draft PR.

    Carries the model spend so it is observable in the tool result. With a
    :class:`~openloop.usage.ledger.WorkerSpendLedger` wired into the
    orchestrator (Phases 4+5), the spend is also recorded to the usage store
    under the invoking agent's scope, fail-closed capped per task before the
    push/PR boundary, and gated on that agent's monthly budget before the
    attempt starts.
    """

    branch: str
    title: str
    body: str
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(slots=True)
class WorkerEdit:
    """What a credential-free worker produced in the prepared workspace."""

    title: str
    body: str
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # OpenHands captures a final authenticated delta before its live container
    # disappears. Other worker backends leave this unset.
    workspace_artifact: "WorkspaceArtifactRef | None" = None


@runtime_checkable
class CodingWorker(Protocol):
    """Edits a *prepared* workspace; holds no credentials, touches no remote.

    Implementations mutate files under ``workspace`` (already cloned and on the
    job branch) and return a :class:`WorkerEdit` describing the change. They
    never clone, commit, or push — the orchestrator owns every credential-
    bearing git operation. Call ``on_step`` (if given) after appending each
    completed step to ``state.completed_steps`` so progress is checkpointed.
    """

    async def run(
        self,
        workspace: Path,
        state: WorkerState,
        on_step: StepCallback | None = None,
    ) -> WorkerEdit | WorkerPaused: ...


@runtime_checkable
class AttemptRunner(Protocol):
    """What the durable paths depend on: one attempt → pushed branch + outcome.

    :class:`GitWorkspaceOrchestrator` is the real implementation; tests use
    in-memory fakes. Implementations may return a durable pause and accept a
    later structured decision; legacy workers still replay from clean.
    """

    async def run_attempt(
        self, state: WorkerState, on_step: StepCallback | None = None
    ) -> WorkerOutcome | WorkerPaused: ...

    async def resume_attempt(
        self,
        state: WorkerState,
        decision: ResumeDecision,
        on_step: StepCallback | None = None,
    ) -> WorkerOutcome | WorkerPaused: ...


def _branch_for(job_id: str) -> str:
    return f"openloop/job-{job_id}"


def _failed(
    job_id: str, state: WorkerState, status: str, exc: Exception
) -> ToolResult:
    """A failed outcome for a worker/PR step — never raised out of execute()."""
    return ToolResult(
        ok=False,
        summary=f"coding worker job {job_id} {status}: {exc}",
        data={
            "job_id": job_id,
            "status": status,
            "branch": state.branch,
            "completed_steps": state.completed_steps,
            "error": str(exc),
        },
    )


def _opened_result(cp: WorkerCheckpoint) -> ToolResult:
    """Reconstruct the success result from a checkpoint of an already-opened PR."""
    return ToolResult(
        ok=True,
        summary=(
            f"draft PR #{cp.pr_number} already open in {cp.repo} (job {cp.job_id})"
        ),
        data={
            "job_id": cp.job_id,
            "status": "opened",
            "branch": cp.branch,
            "pr_number": cp.pr_number,
            "pr_url": cp.pr_url,
            "completed_steps": cp.completed_steps,
            "resumed": True,
        },
    )


def _parked_result(state: WorkerState) -> ToolResult:
    resume = state.openhands_resume
    assert resume is not None and resume.status == "parked"
    return ToolResult(
        ok=True,
        summary=(
            f"coding worker job {state.job_id} is waiting for confirmation: "
            f"{resume.pending_action_summary}"
        ),
        data={
            "job_id": state.job_id,
            "status": "parked",
            "branch": state.branch,
            "decision_id": resume.decision_id,
            "pending_action_summary": resume.pending_action_summary,
            "pending_action_fingerprint": resume.pending_action_fingerprint,
            "completed_steps": state.completed_steps,
        },
    )


def _pr_body(body: str, job_id: str) -> str:
    """Stamp the job id into the PR body so the identity survives in GitHub."""
    body = (body or "").rstrip()
    footer = f"---\n🤖 Opened by the OpenLoop coding worker · job `{job_id}`"
    return f"{body}\n\n{footer}" if body else footer


class CodingWorkerConnector:
    """Maps ``coding_worker.pr:write`` onto an attempt runner + :class:`GitHubClient`."""

    name = CODING_WORKER_TOOL_NAME
    # When the gateway has a WorkflowEngine, this action runs as a durable
    # workflow (approval = wait node). Without one, execute() below is the Phase B
    # fallback path (checkpoint-based resume). Kept in sync with WORKFLOW_NAME in
    # openloop.workflows.coding_worker.
    workflow = "coding_worker"

    def __init__(
        self,
        orchestrator: AttemptRunner,
        github: GitHubClient,
        checkpoints: "CheckpointStore | None" = None,
    ) -> None:
        # The orchestrator owns provision → worker → commit → push (and the git
        # credential); this connector never sees a worker that could push.
        self.orchestrator = orchestrator
        self.github = github
        # Optional: when set, jobs are checkpointed per step and resume on crash.
        self.checkpoints = checkpoints

    def supported_permissions(self) -> set[str]:
        return {CODING_WORKER_PR_WRITE}

    def prepare_args(
        self,
        permission: str,
        args: dict,
        agent=None,
        *,
        warm_key: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Finalize args before they cross the approval boundary.

        Called by the gateway prior to creating the approval request, so the
        values are persisted in the approval args and reused verbatim at
        execute time (and as the workflow's initial state):

        - mints ``job_id`` so one identity threads the whole system;
        - stamps the *invoking* agent's name (Phase 5) so the spend ledger
          attributes the attempt to whoever asked. Stamped unconditionally —
          a model-supplied ``agent`` arg must never redirect attribution.
        - stamps the requesting thread's ``warm_key`` (Phase B) so a warm-
          workspace pool can reuse this thread's checkout across turns. Only the
          gateway supplies it (from the invoking turn); a model-supplied value is
          ignored.
        - stamps the originating ``session_id`` (step 5) so worker spend traces
          to the surface session it was invoked from. Gateway-supplied only,
          like ``warm_key``; a model-supplied value is ignored.
        """
        if permission != CODING_WORKER_PR_WRITE:
            return args
        # Normalization (strip) now lives in CodingWorkerPrArgs' validators —
        # the gateway parses raw args through it before calling here, so this
        # method only stamps identity.
        if not args.get("job_id"):
            args = {**args, "job_id": uuid.uuid4().hex[:12]}
        if agent is not None:
            args = {**args, "agent": agent.metadata.name}
        if warm_key:
            args = {**args, "warm_key": warm_key}
        if session_id:
            args = {**args, "session_id": session_id}
        return args

    def describe(self, permission: str) -> ActionSpec:
        # Generated from the args model the gateway parses with — declaration
        # and enforcement cannot drift.
        return ActionSpec(
            "Run the coding worker on an instruction and open a draft pull "
            "request with its changes. This starts the worker and opens a draft "
            "PR for review; it does not merge.",
            CodingWorkerPrArgs.model_json_schema(),
            model=CodingWorkerPrArgs,
            version=CODING_WORKER_ARGS_VERSION,
        )

    async def execute(self, permission: str, args: dict) -> ToolResult:
        if permission != CODING_WORKER_PR_WRITE:
            return ToolResult(ok=False, summary=f"unsupported permission {permission}")

        job_id = args.get("job_id") or uuid.uuid4().hex[:12]
        base = args.get("base", "main")

        # Resume from a checkpoint if one exists (re-invocation after a crash, or
        # an approval re-resolved). Otherwise start fresh from the request args.
        cp = await self.checkpoints.get(job_id) if self.checkpoints else None
        if cp is not None and cp.status == "opened":
            # Idempotent: the draft PR already exists — never open a second one.
            return _opened_result(cp)

        if cp is not None:
            state = WorkerState.from_dict(cp.state_json)
        else:
            state = WorkerState(
                job_id=job_id,
                repo=args["repo"],
                instruction=args["instruction"],
                base=base,
                branch=_branch_for(job_id),
                agent=args.get("agent"),
                requester_id=args.get("approved_by"),
                approval_id=args.get("approval_id"),
                session_id=args.get("session_id"),
                warm_key=args.get("warm_key"),
            )

        resume = state.openhands_resume
        if resume is not None and resume.status == "parked":
            return _parked_result(state)

        # The two side effects run after resolve() already marked the approval
        # approved, so neither may raise out of execute() — that would surface as
        # a generic error with no failed ToolResult. Record the failure instead.
        cost = (0.0, 0, 0)
        if "push" not in state.completed_steps:
            # The workspace is ephemeral, so local steps (clone…commit) can't
            # resume from a crash — only the push survives. Run a fresh attempt.
            state.completed_steps = []
            await self._save(state, "running")
            try:
                if resume is not None and resume.status in {
                    "finalizing",
                    "terminal",
                }:
                    if resume.status == "finalizing":
                        reconcile = getattr(
                            self.orchestrator, "reconcile_finalizing", None
                        )
                        if reconcile is None:
                            raise OpenHandsResumeError(
                                "finalizing OpenHands recovery is unavailable"
                            )
                        await reconcile(state, on_step=self._checkpointer())
                    deliver = getattr(self.orchestrator, "deliver_terminal", None)
                    if deliver is None:
                        raise OpenHandsResumeError(
                            "terminal OpenHands recovery is unavailable"
                        )
                    outcome = await deliver(state, on_step=self._checkpointer())
                elif resume is not None:
                    raise OpenHandsResumeError(
                        f"active OpenHands {resume.status} segment cannot be replayed"
                    )
                else:
                    outcome = await self.orchestrator.run_attempt(
                        state, on_step=self._checkpointer()
                    )
            except Exception as exc:  # noqa: BLE001
                await self._save(state, "failed", error=str(exc))
                return _failed(job_id, state, "failed", exc)
            if isinstance(outcome, WorkerPaused):
                await self._save(state, "parked")
                return _parked_result(state)
            state.title, state.body = outcome.title, outcome.body
            cost = (outcome.cost_usd, outcome.prompt_tokens, outcome.completion_tokens)
            await self._save(state, "pushed")
        else:
            # Branch already pushed in an earlier run; just (re)open the PR.
            outcome = WorkerOutcome(
                branch=state.branch,
                title=state.title or "Automated change",
                body=state.body or "",
            )

        try:
            pull = await self._open_pr(state, outcome)
        except Exception as exc:  # noqa: BLE001
            await self._save(state, "open_pr_failed", error=str(exc))
            return _failed(job_id, state, "open_pr_failed", exc)

        await self._save(
            state, "opened", pr_number=pull.get("number"), pr_url=pull.get("html_url")
        )
        cleanup = getattr(self.orchestrator, "cleanup_attempt", None)
        if cleanup is not None and state.openhands_resume is not None:
            await cleanup(state, on_step=self._checkpointer())
            await self._save(
                state,
                "opened",
                pr_number=pull.get("number"),
                pr_url=pull.get("html_url"),
            )
        return ToolResult(
            ok=True,
            summary=(
                f"opened draft PR #{pull.get('number')} in {state.repo} "
                f"(job {job_id})"
            ),
            data={
                "job_id": job_id,
                "status": "opened",
                "branch": outcome.branch,
                "pr_number": pull.get("number"),
                "pr_url": pull.get("html_url"),
                "completed_steps": state.completed_steps,
                # Observable spend only — not yet enforced (see WorkerOutcome).
                "cost_usd": cost[0],
                "prompt_tokens": cost[1],
                "completion_tokens": cost[2],
                "idempotency_keys": {
                    "push": state.push_key(),
                    "open_pr": state.open_pr_key(),
                },
            },
        )

    async def resolve_openhands(
        self, job_id: str, decision: ResumeDecision
    ) -> ToolResult:
        """Drive one typed accept/reject decision in checkpoint-only mode."""
        if self.checkpoints is None:
            return ToolResult(ok=False, summary="coding-worker checkpoints unavailable")
        cp = await self.checkpoints.get(job_id)
        if cp is None:
            return ToolResult(ok=False, summary=f"unknown coding worker job {job_id}")
        if cp.status == "opened":
            return _opened_result(cp)
        try:
            state = WorkerState.from_dict(cp.state_json)
            resume = state.openhands_resume
            if resume is None or resume.status != "parked":
                raise OpenHandsResumeError("coding worker job is not awaiting a decision")
            outcome = await self.orchestrator.resume_attempt(
                state,
                decision,
                on_step=self._checkpointer(),
            )
            if isinstance(outcome, WorkerPaused):
                await self._save(state, "parked")
                return _parked_result(state)
            state.title, state.body = outcome.title, outcome.body
            await self._save(state, "pushed")
            pull = await self._open_pr(state, outcome)
            await self._save(
                state,
                "opened",
                pr_number=pull.get("number"),
                pr_url=pull.get("html_url"),
            )
            cleanup = getattr(self.orchestrator, "cleanup_attempt", None)
            if cleanup is not None:
                await cleanup(state, on_step=self._checkpointer())
                await self._save(
                    state,
                    "opened",
                    pr_number=pull.get("number"),
                    pr_url=pull.get("html_url"),
                )
            return ToolResult(
                ok=True,
                summary=(
                    f"opened draft PR #{pull.get('number')} in {state.repo} "
                    f"(job {job_id})"
                ),
                data={
                    "job_id": job_id,
                    "status": "opened",
                    "branch": outcome.branch,
                    "pr_number": pull.get("number"),
                    "pr_url": pull.get("html_url"),
                    "completed_steps": state.completed_steps,
                    "cost_usd": outcome.cost_usd,
                    "prompt_tokens": outcome.prompt_tokens,
                    "completion_tokens": outcome.completion_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001
            state = locals().get("state")
            if isinstance(state, WorkerState):
                await self._save(state, "failed", error=str(exc))
                return _failed(job_id, state, "failed", exc)
            return ToolResult(ok=False, summary=f"coding worker job {job_id} failed: {exc}")

    async def _open_pr(self, state: WorkerState, outcome: WorkerOutcome) -> dict:
        """Open the draft PR, reusing an existing one for this head if present.

        The base comes from ``state.base`` (the checkpoint), not the request args:
        a resume that passes only ``job_id`` must still target the job's original
        base, never silently fall back to ``main``.
        """
        existing = await self.github.find_pull(state.repo, head=outcome.branch)
        if existing is not None:
            return existing
        return await self.github.create_pull(
            repo=state.repo,
            head=outcome.branch,
            base=state.base,
            title=outcome.title,
            body=_pr_body(outcome.body, state.job_id),
            draft=True,
        )

    # Positive legacy recovery policy. A new lifecycle must never become
    # executable merely because it is not in a stale terminal tuple.
    _LEGACY_RECOVERABLE = frozenset({"running", "pushed", "open_pr_failed"})
    _LEGACY_TERMINAL = frozenset({"opened", "failed"})
    _VERSIONED_STATE_KEYS = frozenset(
        {
            "schema_version",
            "minimum_reader_version",
            "openhands_resume",
            "openhands_resume_state",
        }
    )

    async def resume_incomplete(self) -> list[str]:
        """Re-drive jobs left non-terminal by a crash. Call once at startup.

        The approval path (:meth:`ToolGateway.resolve`) marks the approval
        ``approved`` *before* :meth:`execute` runs and will not re-invoke it, so a
        crash mid-execute would otherwise strand the job — the resume logic in
        ``execute`` is unreachable through the normal path. This reconciler drives
        resume directly off the checkpoints instead: ``execute`` is idempotent
        (checkpoints + force-push + ``find_pull``), so finishing or restarting each
        job is safe.

        Across replicas, the app lifespan runs this under a ``startup-recovery``
        :class:`~openloop.coordination.DistributedLock` so only the leader resumes
        jobs; ``execute`` itself stays idempotent if two ever overlap. Phase C
        folds this into the workflow engine, where approval is an event and
        ``resolve`` is a thin adapter.
        """
        if self.checkpoints is None:
            return []
        resumed: list[str] = []
        for cp in await self.checkpoints.recent(limit=1000):
            if cp.status in self._LEGACY_TERMINAL:
                continue
            if isinstance(cp.state_json, dict) and cp.state_json.get(
                "openhands_resume"
            ) is not None:
                state: WorkerState | None = None
                try:
                    state = WorkerState.from_dict(cp.state_json)
                    resume = state.openhands_resume
                    assert resume is not None
                    if resume.status in {"parked", "cleaned"}:
                        continue
                    if resume.status == "parking":
                        reconcile = getattr(
                            self.orchestrator, "reconcile_parking", None
                        )
                        if reconcile is None:
                            raise OpenHandsResumeError(
                                "OpenHands parking reconciler is unavailable"
                            )
                        await reconcile(state, on_step=self._checkpointer())
                        await self._save(state, "parked")
                        resumed.append(cp.job_id)
                        continue
                    if resume.status in {"finalizing", "terminal"}:
                        await self.execute(
                            "pr:write",
                            {
                                "job_id": cp.job_id,
                                "repo": cp.repo,
                                "instruction": cp.instruction,
                                "base": cp.base,
                            },
                        )
                        resumed.append(cp.job_id)
                        continue
                    raise OpenHandsResumeError(
                        f"active OpenHands {resume.status} segment was "
                        "interrupted and cannot be replayed safely"
                    )
                except Exception as exc:  # noqa: BLE001 — quarantine typed state
                    logger.error(
                        "typed reconciler failing closed coding-worker "
                        "checkpoint %s: %s",
                        cp.job_id,
                        exc,
                    )
                    if state is not None:
                        await self._save(state, "failed", error=str(exc))
                    continue
            if not isinstance(cp.state_json, dict) or (
                self._VERSIONED_STATE_KEYS.intersection(cp.state_json)
            ):
                logger.error(
                    "quarantining coding-worker checkpoint %s: versioned or "
                    "malformed state requires a typed reconciler (status=%s)",
                    cp.job_id,
                    cp.status,
                )
                continue
            if cp.status not in self._LEGACY_RECOVERABLE:
                logger.error(
                    "quarantining coding-worker checkpoint %s: unsupported "
                    "legacy lifecycle status=%s",
                    cp.job_id,
                    cp.status,
                )
                continue
            logger.info("resuming coding-worker job %s (was %s)", cp.job_id, cp.status)
            await self.execute(
                "pr:write",
                {
                    "job_id": cp.job_id,
                    "repo": cp.repo,
                    "instruction": cp.instruction,
                    "base": cp.base,
                },
            )
            resumed.append(cp.job_id)
        return resumed

    def _checkpointer(self) -> StepCallback | None:
        """A per-step callback that persists progress, or None when no store."""
        if self.checkpoints is None:
            return None

        async def on_step(state: WorkerState) -> None:
            status = (
                state.openhands_resume.status
                if state.openhands_resume is not None
                else "running"
            )
            await self._save(state, status)

        return on_step

    async def _save(
        self,
        state: WorkerState,
        status: str,
        *,
        pr_number: int | None = None,
        pr_url: str | None = None,
        error: str | None = None,
    ) -> None:
        if self.checkpoints is None:
            return
        await self.checkpoints.upsert(
            WorkerCheckpoint(
                job_id=state.job_id,
                repo=state.repo,
                instruction=state.instruction,
                base=state.base,
                branch=state.branch,
                status=status,
                completed_steps=list(state.completed_steps),
                state_json=state.to_dict(),
                title=state.title,
                body=state.body,
                pr_number=pr_number,
                pr_url=pr_url,
                error=error,
            )
        )


@runtime_checkable
class _Completer(Protocol):
    async def complete(self, model: str, messages: list[dict], **kwargs): ...


class GitWorkspaceOrchestrator:
    """The single credential-bearing boundary around a worker attempt.

    Owns provision (clone + branch) → worker edit → commit → push for **both**
    durable paths — the connector's checkpoint fallback and the workflow in
    :mod:`openloop.workflows.coding_worker` — so credential-bearing git lives in
    exactly one place (a Phase 2 exit criterion).

    SECURITY: the credential is resolved through the :class:`CredentialResolver`
    seam at attempt time, kept attempt-local, and handed to git as an ephemeral
    ``http.extraHeader`` on each command — never in a remote URL — so nothing
    credential-shaped is ever written into the workspace (``.git/config`` keeps
    the plain URL). The worker only sees the prepared workspace: it is
    credential-free by construction, not by discipline.

    Legacy workers use one opaque replay unit over a throwaway workspace.
    OpenHands cold resume instead makes pause/final artifacts the replay-safe
    boundaries: an active segment is never guessed or re-executed after a crash.

    SPEND (hardening Phases 4+5): when a
    :class:`~openloop.usage.ledger.WorkerSpendLedger` is wired, the invoking
    agent's monthly budget gates the attempt *before any work* (no credential
    resolve, no clone), and every attempt's model spend is recorded to the
    usage store and checked against that agent's per-task budget — after the
    worker's edit, before commit/push — so an over-budget attempt fails
    closed (no push, no PR) on **both** durable paths at once. Attribution
    follows ``state.agent`` (the invoking agent threaded through the approval
    args), not a boot-time owner.
    """

    def __init__(
        self,
        worker: CodingWorker,
        credentials: CredentialResolver,
        *,
        scope: CredentialScope | None = None,
        remote_base: str = "https://github.com",
        workspace_root: Path | None = None,
        ledger: "WorkerSpendLedger | None" = None,
        warm_pool: "WarmWorkspacePool | None" = None,
    ) -> None:
        self.worker = worker
        self._credentials = credentials
        self._scope = scope or CredentialScope(integration="github")
        # Overridable for GitHub Enterprise — and for local file:// test remotes.
        self._remote_base = remote_base.rstrip("/")
        # Where workspaces are created (default: the system tempdir). A
        # containerized deploy using the docker sandbox must point this at a
        # path bind-mounted from the host at the SAME location — sibling
        # sandbox containers resolve `-v` paths on the host, not in here.
        self._workspace_root = workspace_root
        # Optional Phase 4 spend ledger: record + fail-closed per-task cap.
        self._ledger = ledger
        # Optional Phase B warm-workspace pool: when a WorkerState carries a
        # warm_key, reuse the thread's kept checkout (fetch + reset) instead of
        # cloning cold. The pool owns only the directory lifecycle — every git
        # command still runs here, so the one credential boundary is unchanged.
        self._warm_pool = warm_pool

    async def run_attempt(
        self, state: WorkerState, on_step: StepCallback | None = None
    ) -> WorkerOutcome | WorkerPaused:
        async def step(name: str) -> None:
            state.completed_steps.append(name)
            if on_step is not None:
                await on_step(state)

        # Phase 5 monthly gate: a spent monthly budget refuses the attempt
        # outright, before a credential is resolved or a workspace exists.
        # Raises out of the attempt → terminal fail on both durable paths.
        if self._ledger is not None:
            await self._ledger.check_monthly(
                state.agent,
                job_id=state.job_id,
                approval_id=state.approval_id,
                approver=state.requester_id,
                session_id=state.session_id,
            )
            # Stamp this agent's per-task cap so an agentic worker can stop
            # itself near the ceiling instead of running to completion and
            # failing the post-run settle (the money is already spent by then).
            state.budget_usd = self._ledger.per_task_usd_for(state.agent)

        # Resolved fresh per attempt and kept local — never stored, never in a
        # URL. The auth header value still surfaces in a failed command line,
        # so git failures redact both the token and its basic-auth encoding.
        token = await self._credentials.resolve(self._scope)

        # Provision the workspace. With a warm pool and a thread warm_key, borrow
        # the thread's checkout (reused if live, else freshly provisioned by the
        # pool); otherwise the unchanged ephemeral clone-and-discard temp dir.
        lease = None
        if self._warm_pool is not None and state.warm_key:
            lease = await self._warm_pool.acquire(state.warm_key, state.repo)
            workspace = lease.path
        else:
            if self._workspace_root is not None:
                self._workspace_root.mkdir(parents=True, exist_ok=True)
            workspace = Path(
                tempfile.mkdtemp(
                    prefix=f"openloop-{state.job_id}-",
                    dir=self._workspace_root,
                )
            )
        try:
            if lease is not None and lease.warm:
                # Reuse the kept checkout: reset to the freshly fetched base and
                # branch off it, reusing the object store instead of re-cloning.
                # The job-exclusive branch is (re)created with -B so a re-warm on
                # resume is idempotent, and `clean -fdx` drops any prior leftovers
                # so the worker sees a pristine tree — same as a cold clone.
                await self._git(
                    *_auth_config(token),
                    "fetch", "--depth", "1", "origin", state.base,
                    cwd=workspace, redact=_auth_secrets(token),
                )
                await self._git("reset", "--hard", "FETCH_HEAD", cwd=workspace)
                await self._git("clean", "-fdx", cwd=workspace)
                await self._git("checkout", "-B", state.branch, cwd=workspace)
                await step("clone")
                await step("branch")
            else:
                await self._git(
                    *_auth_config(token),
                    "clone", "--depth", "1", "--branch", state.base,
                    f"{self._remote_base}/{state.repo}.git", str(workspace),
                    redact=_auth_secrets(token),
                )
                await step("clone")

                await self._git("checkout", "-b", state.branch, cwd=workspace)
                await step("branch")

            # The worker edits the prepared workspace. No credential in scope:
            # not in its arguments, not anywhere under the workspace.
            logger.info(
                "coding-worker attempt job=%s repo=%s branch=%s backend=%s "
                "(warm=%s)",
                state.job_id, state.repo, state.branch,
                type(self.worker).__name__,
                bool(lease and lease.warm),
            )
            try:
                edit = await self.worker.run(workspace, state, on_step)
            except WorkerRunAborted as aborted:
                # The worker stopped itself at the spend/time ceiling. Record the
                # spend that already happened, then fail the attempt closed — no
                # push, no PR, and the workspace teardown discards the partial edit.
                if self._ledger is not None:
                    await self._ledger.settle(
                        agent=state.agent,
                        job_id=state.job_id,
                        approval_id=state.approval_id,
                        approver=state.requester_id,
                        session_id=state.session_id,
                        cost_usd=aborted.cost_usd,
                        prompt_tokens=aborted.prompt_tokens,
                        completion_tokens=aborted.completion_tokens,
                    )
                # settle() raises on an over-cap spend; a deadline abort under the
                # cap won't, so re-raise to fail closed regardless.
                raise
            if isinstance(edit, WorkerPaused):
                await self._park(state, edit, on_step)
                # A paused checkout is never reusable warm state. The encrypted
                # cumulative delta is now the source of truth.
                if lease is not None:
                    await lease.discard()
                return edit
            # Persist title/body so a post-push crash can still open the PR.
            state.title, state.body = edit.title, edit.body

            # Phase 4 gate: record the attempt's spend and fail closed on the
            # invoking agent's per-task cap BEFORE the push/PR boundary.
            # Raises out of the attempt (both durable paths mark the job
            # failed — terminal, so no resume loop re-spends), and the
            # workspace teardown in `finally` discards the over-budget edit.
            if state.openhands_resume is not None:
                await self._record_terminal(state, edit, on_step)
            elif self._ledger is not None:
                await self._ledger.settle(
                    agent=state.agent,
                    job_id=state.job_id,
                    approval_id=state.approval_id,
                    approver=state.requester_id,
                    session_id=state.session_id,
                    cost_usd=edit.cost_usd,
                    prompt_tokens=edit.prompt_tokens,
                    completion_tokens=edit.completion_tokens,
                )

            if (workspace / "OPENLOOP_PR.md").exists():
                raise OpenHandsResumeError(
                    "reserved OPENLOOP_PR.md survived the terminal capture"
                )
            await self._git("add", "-A", cwd=workspace)
            await self._git(
                "-c", "user.email=worker@openloop.team",
                "-c", "user.name=OpenLoop coding worker",
                "commit", "-m", edit.title, cwd=workspace,
            )
            await step("commit")

            # Re-resolve right before the push: a long edit step can outlive the
            # token minted at clone time (App installation tokens expire). The
            # resolver's cache makes this free while the original is still fresh.
            push_token = await self._credentials.resolve(self._scope)
            # Force-push to the job-exclusive branch so the push is idempotent.
            # The "push" checkpoint is only written by the step() below, so a crash
            # in the window between this push succeeding and that write leaves the
            # checkpoint saying "not pushed". Resume then runs a fresh attempt and
            # pushes again — without --force that second push is rejected as a
            # non-fast-forward (the branch already exists). The branch is owned
            # solely by this job_id, so overwriting it is safe; the trade-off is
            # that a resumed run may carry a freshly regenerated diff.
            await self._git(
                *_auth_config(push_token),
                "push", "--force", "origin", state.branch, cwd=workspace,
                redact=_auth_secrets(push_token),
            )
            await step("push")

            outcome = WorkerOutcome(
                branch=state.branch,
                title=edit.title,
                body=edit.body,
                cost_usd=edit.cost_usd,
                prompt_tokens=edit.prompt_tokens,
                completion_tokens=edit.completion_tokens,
            )
            logger.info(
                "coding-worker attempt done job=%s branch=%s backend=%s pushed "
                "($%.4f, %d+%d tok)",
                state.job_id, state.branch, type(self.worker).__name__,
                edit.cost_usd, edit.prompt_tokens, edit.completion_tokens,
            )
            # Keep the checkout warm for this thread's next turn (a no-op for the
            # ephemeral path). The push already succeeded, so the warm tree is a
            # clean, pushed state — safe to reuse.
            if lease is not None:
                await lease.keep()
            return outcome
        except BaseException:
            # Any failure may leave the tree dirty/corrupt; drop it so the next
            # turn cold-reconstructs rather than reusing a bad checkout.
            if lease is not None:
                await lease.discard()
            raise
        finally:
            if lease is not None:
                await lease.release()
            else:
                shutil.rmtree(workspace, ignore_errors=True)

    async def resume_attempt(
        self,
        state: WorkerState,
        decision: ResumeDecision,
        on_step: StepCallback | None = None,
    ) -> WorkerOutcome | WorkerPaused:
        """Continue one parked OpenHands job from its authenticated artifact.

        This path deliberately never acquires the mutable warm pool. It fetches
        the recorded base object into a fresh checkout and applies only the
        fully verified cumulative delta before attaching the conversation.
        """
        resume = state.openhands_resume
        if resume is None or resume.status != "parked":
            raise OpenHandsResumeError("only a parked OpenHands job can resume")
        if decision.decision_id != resume.decision_id:
            raise OpenHandsResumeError("stale OpenHands resume decision")
        if decision.actor_id != resume.slack_requester_id:
            raise OpenHandsResumeError("OpenHands resume decision is unauthorized")

        worker_image = getattr(self.worker, "server_image", None)
        artifact_store = getattr(self.worker, "artifact_store", None)
        if worker_image != resume.image_digest:
            raise OpenHandsResumeError("OpenHands resume image digest mismatch")
        if artifact_store is None or (
            artifact_store.keys.master_key_id != resume.master_key_id
        ):
            raise OpenHandsResumeError("OpenHands resume master-key mismatch")
        artifact_ref = resume.workspace_artifact
        if artifact_ref is None:
            raise OpenHandsResumeError("parked OpenHands job has no workspace artifact")

        resume.transition_to(
            "resuming",
            segment_id=uuid.uuid4().hex,
            resolved_event_id=decision.event_id,
            resolved_decision=decision,
        )
        state.completed_steps = []
        if on_step is not None:
            await on_step(state)

        if self._ledger is not None:
            await self._ledger.check_monthly(
                state.agent,
                job_id=state.job_id,
                approval_id=state.approval_id,
                approver=state.requester_id,
                session_id=state.session_id,
            )
            state.budget_usd = self._ledger.per_task_usd_for(state.agent)
        token = await self._credentials.resolve(self._scope)
        if self._workspace_root is not None:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"openloop-{state.job_id}-resume-",
                dir=self._workspace_root,
            )
        )

        async def step(name: str) -> None:
            state.completed_steps.append(name)
            if on_step is not None:
                await on_step(state)

        try:
            await self._git("init", cwd=workspace)
            await self._git(
                "remote",
                "add",
                "origin",
                f"{self._remote_base}/{state.repo}.git",
                cwd=workspace,
            )
            await self._git(
                *_auth_config(token),
                "fetch",
                "--depth",
                "1",
                "origin",
                artifact_ref.base_commit,
                cwd=workspace,
                redact=_auth_secrets(token),
            )
            await step("clone")
            await self._git("checkout", "-B", state.branch, "FETCH_HEAD", cwd=workspace)
            await step("branch")
            await self._restore_artifact(workspace, artifact_ref)

            try:
                edit = await self.worker.run(workspace, state, on_step)
            except WorkerRunAborted as aborted:
                await self._settle_cumulative(
                    state,
                    aborted.cost_usd,
                    aborted.prompt_tokens,
                    aborted.completion_tokens,
                )
                raise
            if isinstance(edit, WorkerPaused):
                await self._park(state, edit, on_step)
                return edit

            state.title, state.body = edit.title, edit.body
            await self._record_terminal(state, edit, on_step)
            if (workspace / "OPENLOOP_PR.md").exists():
                raise OpenHandsResumeError(
                    "reserved OPENLOOP_PR.md survived the terminal capture"
                )
            await self._git("add", "-A", cwd=workspace)
            await self._git(
                "-c",
                "user.email=worker@openloop.team",
                "-c",
                "user.name=OpenLoop coding worker",
                "commit",
                "-m",
                edit.title,
                cwd=workspace,
            )
            await step("commit")
            push_token = await self._credentials.resolve(self._scope)
            await self._git(
                *_auth_config(push_token),
                "push",
                "--force",
                "origin",
                state.branch,
                cwd=workspace,
                redact=_auth_secrets(push_token),
            )
            await step("push")
            return WorkerOutcome(
                branch=state.branch,
                title=edit.title,
                body=edit.body,
                cost_usd=edit.cost_usd,
                prompt_tokens=edit.prompt_tokens,
                completion_tokens=edit.completion_tokens,
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    async def _restore_artifact(
        self, workspace: Path, artifact_ref: "WorkspaceArtifactRef"
    ):
        store = getattr(self.worker, "artifact_store", None)
        if store is None:
            raise OpenHandsResumeError("OpenHands artifact store is unavailable")
        identity = artifact_ref.artifact.identity
        with store.open_verified(artifact_ref.artifact, identity) as verified:
            if (
                verified.manifest.format != "git-delta"
                or verified.manifest.base_commit != artifact_ref.base_commit
            ):
                raise OpenHandsResumeError("OpenHands artifact manifest mismatch")
            try:
                patch = verified.stream.read().decode("utf-8")
            except UnicodeDecodeError as exc:
                raise OpenHandsResumeError("OpenHands git delta is not valid") from exc
        if patch:
            await self._git("apply", "--binary", "-", cwd=workspace, stdin=patch)
        return verified.manifest

    async def deliver_terminal(
        self,
        state: WorkerState,
        on_step: StepCallback | None = None,
    ) -> WorkerOutcome:
        """Reconstruct and push a terminal artifact without another model call."""
        resume = state.openhands_resume
        if (
            resume is None
            or resume.status != "terminal"
            or resume.workspace_artifact is None
            or resume.workspace_artifact.artifact.identity.kind
            not in {"final", "checkpoint"}
        ):
            raise OpenHandsResumeError("no recoverable terminal OpenHands artifact")
        artifact_ref = resume.workspace_artifact
        await self._settle_cumulative(
            state,
            resume.cumulative_cost,
            resume.cumulative_prompt_tokens,
            resume.cumulative_completion_tokens,
        )
        resume.last_settled_cumulative_cost = resume.cumulative_cost
        resume.last_settled_cumulative_prompt_tokens = (
            resume.cumulative_prompt_tokens
        )
        resume.last_settled_cumulative_completion_tokens = (
            resume.cumulative_completion_tokens
        )
        if on_step is not None:
            await on_step(state)
        token = await self._credentials.resolve(self._scope)
        if self._workspace_root is not None:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"openloop-{state.job_id}-terminal-",
                dir=self._workspace_root,
            )
        )

        async def step(name: str) -> None:
            if name not in state.completed_steps:
                state.completed_steps.append(name)
            if on_step is not None:
                await on_step(state)

        try:
            await self._git("init", cwd=workspace)
            await self._git(
                "remote",
                "add",
                "origin",
                f"{self._remote_base}/{state.repo}.git",
                cwd=workspace,
            )
            await self._git(
                *_auth_config(token),
                "fetch",
                "--depth",
                "1",
                "origin",
                artifact_ref.base_commit,
                cwd=workspace,
                redact=_auth_secrets(token),
            )
            await step("clone")
            await self._git("checkout", "-B", state.branch, "FETCH_HEAD", cwd=workspace)
            await step("branch")
            manifest = await self._restore_artifact(workspace, artifact_ref)
            if manifest.pr_title is None:
                raise OpenHandsResumeError("final artifact has no PR metadata")
            state.title = manifest.pr_title
            state.body = manifest.pr_body or ""
            if (workspace / "OPENLOOP_PR.md").exists():
                raise OpenHandsResumeError(
                    "reserved OPENLOOP_PR.md exists in final artifact"
                )
            await self._git("add", "-A", cwd=workspace)
            await self._git(
                "-c",
                "user.email=worker@openloop.team",
                "-c",
                "user.name=OpenLoop coding worker",
                "commit",
                "-m",
                state.title,
                cwd=workspace,
            )
            await step("commit")
            push_token = await self._credentials.resolve(self._scope)
            await self._git(
                *_auth_config(push_token),
                "push",
                "--force",
                "origin",
                state.branch,
                cwd=workspace,
                redact=_auth_secrets(push_token),
            )
            await step("push")
            return WorkerOutcome(
                branch=state.branch,
                title=state.title,
                body=state.body,
                cost_usd=resume.cumulative_cost,
                prompt_tokens=resume.cumulative_prompt_tokens,
                completion_tokens=resume.cumulative_completion_tokens,
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    async def cleanup_attempt(
        self,
        state: WorkerState,
        on_step: StepCallback | None = None,
    ) -> None:
        """Delete terminal OpenHands state after the draft PR is durable."""
        resume = state.openhands_resume
        if resume is None or resume.status == "cleaned":
            return
        if resume.status != "terminal":
            raise OpenHandsResumeError("cannot clean a non-terminal OpenHands job")
        store = getattr(self.worker, "artifact_store", None)
        if store is None:
            raise OpenHandsResumeError("OpenHands artifact store is unavailable")
        artifact_job_id = resume.broker_job_id or state.job_id
        paths = store.layout.for_job(artifact_job_id)
        shutil.rmtree(paths.root, ignore_errors=True)
        resume.transition_to(
            "cleaned",
            workspace_artifact=None,
            pending_action_summary=None,
            pending_action_fingerprint=None,
        )
        if on_step is not None:
            await on_step(state)

    async def _park(
        self,
        state: WorkerState,
        paused: WorkerPaused,
        on_step: StepCallback | None,
    ) -> None:
        resume = state.openhands_resume
        if resume is None or resume.status not in {
            "running",
            "resuming",
            "parking",
        }:
            raise OpenHandsResumeError("cannot park inactive OpenHands segment")
        if (
            paused.conversation_id != resume.conversation_id
            or paused.segment_id != resume.segment_id
        ):
            raise OpenHandsResumeError("OpenHands paused result identity mismatch")
        if resume.status != "parking":
            resume.transition_to(
                "parking",
                decision_id=paused.decision_id,
                pending_action_summary=paused.pending_action_summary,
                pending_action_fingerprint=paused.pending_action_fingerprint,
                workspace_artifact=paused.workspace_artifact,
                cumulative_cost=paused.cumulative_cost,
                cumulative_prompt_tokens=paused.cumulative_prompt_tokens,
                cumulative_completion_tokens=paused.cumulative_completion_tokens,
                resolved_event_id=None,
                resolved_decision=None,
            )
            if on_step is not None:
                await on_step(state)
        elif any(
            (
                resume.decision_id != paused.decision_id,
                resume.workspace_artifact != paused.workspace_artifact,
                resume.cumulative_cost != paused.cumulative_cost,
                resume.cumulative_prompt_tokens != paused.cumulative_prompt_tokens,
                resume.cumulative_completion_tokens
                != paused.cumulative_completion_tokens,
            )
        ):
            raise OpenHandsResumeError("broker parking checkpoint mismatch")
        await self._settle_cumulative(
            state,
            paused.cumulative_cost,
            paused.cumulative_prompt_tokens,
            paused.cumulative_completion_tokens,
        )
        resume.transition_to(
            "parked",
            last_settled_cumulative_cost=paused.cumulative_cost,
            last_settled_cumulative_prompt_tokens=paused.cumulative_prompt_tokens,
            last_settled_cumulative_completion_tokens=(
                paused.cumulative_completion_tokens
            ),
        )
        if on_step is not None:
            await on_step(state)

    async def reconcile_parking(
        self,
        state: WorkerState,
        on_step: StepCallback | None = None,
    ) -> WorkerPaused:
        """Finish a crash-interrupted parking transition without model work."""
        resume = state.openhands_resume
        if (
            resume is None
            or resume.status != "parking"
            or resume.workspace_artifact is None
            or not resume.decision_id
            or not resume.pending_action_summary
            or not resume.pending_action_fingerprint
        ):
            raise OpenHandsResumeError("OpenHands parking state is incomplete")
        store = getattr(self.worker, "artifact_store", None)
        if store is None:
            raise OpenHandsResumeError("OpenHands artifact store is unavailable")
        identity = resume.workspace_artifact.artifact.identity
        with store.open_verified(resume.workspace_artifact.artifact, identity) as verified:
            if verified.manifest.base_commit != resume.resolved_base_commit:
                raise OpenHandsResumeError("OpenHands parking artifact base mismatch")
        if resume.broker_job_id is not None:
            adapter = getattr(self.worker, "_docker_adapter", None)
            is_parked = getattr(adapter, "is_parked", None)
            if not callable(is_parked) or resume.broker_generation is None:
                raise OpenHandsResumeError("broker parking recovery is unavailable")
            parked = await asyncio.to_thread(
                is_parked,
                state.job_id,
                resume.broker_job_id,
                resume.broker_generation,
            )
            if not parked:
                recover = getattr(adapter, "recover_checkpoint", None)
                if not callable(recover):
                    raise OpenHandsResumeError(
                        "broker checkpoint recovery is unavailable"
                    )
                await asyncio.to_thread(
                    recover,
                    state.job_id,
                    resume.broker_job_id,
                    resume.broker_generation,
                    resume.segment_id,
                    resume.workspace_artifact.artifact,
                    terminal=False,
                )
                parked = await asyncio.to_thread(
                    is_parked,
                    state.job_id,
                    resume.broker_job_id,
                    resume.broker_generation,
                )
                if not parked:
                    raise OpenHandsResumeError(
                        "broker job has not reached parked state"
                    )
        await self._settle_cumulative(
            state,
            resume.cumulative_cost,
            resume.cumulative_prompt_tokens,
            resume.cumulative_completion_tokens,
        )
        resume.transition_to(
            "parked",
            last_settled_cumulative_cost=resume.cumulative_cost,
            last_settled_cumulative_prompt_tokens=resume.cumulative_prompt_tokens,
            last_settled_cumulative_completion_tokens=(
                resume.cumulative_completion_tokens
            ),
        )
        if on_step is not None:
            await on_step(state)
        return WorkerPaused(
            conversation_id=resume.conversation_id,
            segment_id=resume.segment_id,
            decision_id=resume.decision_id,
            pending_action_summary=resume.pending_action_summary,
            pending_action_fingerprint=resume.pending_action_fingerprint,
            workspace_artifact=resume.workspace_artifact,
            cumulative_cost=resume.cumulative_cost,
            cumulative_prompt_tokens=resume.cumulative_prompt_tokens,
            cumulative_completion_tokens=resume.cumulative_completion_tokens,
        )

    async def reconcile_finalizing(
        self,
        state: WorkerState,
        on_step: StepCallback | None = None,
    ) -> None:
        """Finish a crash-interrupted broker finalization without model work."""
        resume = state.openhands_resume
        if (
            resume is None
            or resume.status != "finalizing"
            or resume.workspace_artifact is None
            or resume.broker_job_id is None
            or resume.broker_generation is None
        ):
            raise OpenHandsResumeError("OpenHands finalizing state is incomplete")
        store = getattr(self.worker, "artifact_store", None)
        if store is None:
            raise OpenHandsResumeError("OpenHands artifact store is unavailable")
        identity = resume.workspace_artifact.artifact.identity
        with store.open_verified(resume.workspace_artifact.artifact, identity) as verified:
            if (
                verified.manifest.base_commit != resume.resolved_base_commit
                or verified.manifest.pr_title is None
            ):
                raise OpenHandsResumeError(
                    "OpenHands finalizing artifact manifest mismatch"
                )
        adapter = getattr(self.worker, "_docker_adapter", None)
        recover = getattr(adapter, "recover_checkpoint", None)
        if not callable(recover):
            raise OpenHandsResumeError("broker checkpoint recovery is unavailable")
        await asyncio.to_thread(
            recover,
            state.job_id,
            resume.broker_job_id,
            resume.broker_generation,
            resume.segment_id,
            resume.workspace_artifact.artifact,
            terminal=True,
        )
        resume.transition_to("terminal")
        if on_step is not None:
            await on_step(state)

    async def _record_terminal(
        self,
        state: WorkerState,
        edit: WorkerEdit,
        on_step: StepCallback | None,
    ) -> None:
        resume = state.openhands_resume
        if resume is None or edit.workspace_artifact is None:
            raise OpenHandsResumeError("terminal OpenHands edit has no final artifact")
        if edit.workspace_artifact.artifact.identity.kind not in {
            "final",
            "checkpoint",
        }:
            raise OpenHandsResumeError("terminal OpenHands artifact has wrong kind")
        await self._settle_cumulative(
            state,
            edit.cost_usd,
            edit.prompt_tokens,
            edit.completion_tokens,
        )
        if resume.status == "terminal":
            if (
                resume.workspace_artifact != edit.workspace_artifact
                or resume.cumulative_cost != edit.cost_usd
                or resume.cumulative_prompt_tokens != edit.prompt_tokens
                or resume.cumulative_completion_tokens != edit.completion_tokens
            ):
                raise OpenHandsResumeError("broker terminal checkpoint mismatch")
            resume.last_settled_cumulative_cost = edit.cost_usd
            resume.last_settled_cumulative_prompt_tokens = edit.prompt_tokens
            resume.last_settled_cumulative_completion_tokens = edit.completion_tokens
        else:
            resume.transition_to(
                "terminal",
                workspace_artifact=edit.workspace_artifact,
                cumulative_cost=edit.cost_usd,
                cumulative_prompt_tokens=edit.prompt_tokens,
                cumulative_completion_tokens=edit.completion_tokens,
                last_settled_cumulative_cost=edit.cost_usd,
                last_settled_cumulative_prompt_tokens=edit.prompt_tokens,
                last_settled_cumulative_completion_tokens=edit.completion_tokens,
            )
        if on_step is not None:
            await on_step(state)

    async def _settle_cumulative(
        self,
        state: WorkerState,
        cost: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        resume = state.openhands_resume
        if resume is None:
            return
        deltas = (
            cost - resume.last_settled_cumulative_cost,
            prompt_tokens - resume.last_settled_cumulative_prompt_tokens,
            completion_tokens - resume.last_settled_cumulative_completion_tokens,
        )
        if min(deltas) < 0:
            raise OpenHandsResumeError("OpenHands cumulative metrics moved backwards")
        if self._ledger is not None:
            await self._ledger.settle(
                agent=state.agent,
                job_id=state.job_id,
                approval_id=state.approval_id,
                approver=state.requester_id,
                session_id=state.session_id,
                broker_job_id=resume.broker_job_id,
                broker_generation=resume.broker_generation,
                idempotency_key=(
                    f"{state.job_id}:{resume.conversation_id}:{resume.segment_id}"
                ),
                record_cost_usd=deltas[0],
                record_prompt_tokens=deltas[1],
                record_completion_tokens=deltas[2],
                cap_cost_usd=cost,
            )

    async def _git(
        self,
        *args: str,
        cwd: Path | None = None,
        stdin: str | None = None,
        redact: "str | tuple[str, ...] | None" = None,
    ) -> str:
        return await self._run("git", *args, cwd=cwd, stdin=stdin, redact=redact)

    async def _run(
        self,
        *cmd: str,
        cwd: Path | None = None,
        stdin: str | None = None,
        redact: "str | tuple[str, ...] | None" = None,
    ) -> str:
        return await _run_process(*cmd, cwd=cwd, stdin=stdin, redact=redact)


class BuiltinCodingWorker:
    """OpenLoop's own light worker (``CODING_WORKER_BACKEND=builtin``).

    Its one strategy today is *diff*: ask the model for a unified diff in a
    single completion and apply it. Future strategies live inside this class
    (behind a ``CODING_WORKER_BUILTIN_STRATEGY`` setting when a second one
    exists), not as sibling classes.

    Credential-free (hardening Phase 2): it receives a *prepared* workspace
    from :class:`GitWorkspaceOrchestrator` and only generates + ``git apply``s
    a diff — it holds no token, never sees a remote, and cannot push. Anything
    that doesn't apply cleanly fails the job rather than being force-written.

    Isolation (hardening Phase 3): the model-influenced execution — applying
    the generated diff — runs through a :class:`~openloop.sandbox.Sandbox`.
    The model *call* stays in this (controller) process, so the LLM key is
    never in the sandbox's reach; with :class:`~openloop.sandbox.DockerSandbox`
    the apply runs in a throwaway container with no network and no env.
    """

    def __init__(
        self,
        model: str,
        gateway: _Completer | None = None,
        *,
        sandbox: "Sandbox | None" = None,
        max_context_bytes: int = 60_000,
    ) -> None:
        from openloop.sandbox import HostSandbox

        self.model = model
        self._gateway = gateway
        self.sandbox = sandbox or HostSandbox()
        self.max_context_bytes = max_context_bytes

    def _completer(self) -> _Completer:
        if self._gateway is None:
            from openloop.models.gateway import ModelGateway

            self._gateway = ModelGateway()
        return self._gateway

    async def run(
        self,
        workspace: Path,
        state: WorkerState,
        on_step: StepCallback | None = None,
    ) -> WorkerEdit:
        diff, title, body, resp = await self._generate(state, workspace)
        # The one place model-generated content executes — through the sandbox.
        await self.sandbox.exec(
            workspace, "git", "apply", "--whitespace=nowarn", stdin=diff
        )
        state.completed_steps.append("edit")
        if on_step is not None:
            await on_step(state)
        return WorkerEdit(
            title=title,
            body=body,
            cost_usd=resp.cost_usd,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )

    async def _generate(self, state: WorkerState, workspace: Path):
        """Ask the model for a unified diff + PR title/body for the instruction.

        Returns ``(diff, title, body, response)`` — the response carries token
        counts and cost so the caller can surface the worker's model spend.
        """
        context = self._repo_context(workspace)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding worker. Given a repository snapshot and an "
                    "instruction, produce changes as a single unified diff that "
                    "applies cleanly with `git apply` from the repo root. Respond "
                    "with exactly three sections, each on its own line and in this "
                    "order:\nTITLE: <one-line PR title>\nBODY: <short PR description>\n"
                    "DIFF:\n<the unified diff>\nDo not wrap the diff in markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instruction:\n{state.instruction}\n\n"
                    f"Repository {state.repo} (base {state.base}):\n{context}"
                ),
            },
        ]
        resp = await self._completer().complete(self.model, messages)
        diff, title, body = _parse_generation(resp.text)
        return diff, title, body, resp

    def _repo_context(self, workspace: Path) -> str:
        """A best-effort, size-capped snapshot of tracked text files."""
        parts: list[str] = []
        budget = self.max_context_bytes
        for path in sorted(workspace.rglob("*")):
            if ".git" in path.parts or not path.is_file():
                continue
            try:
                text = path.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.relative_to(workspace)
            block = f"\n=== {rel} ===\n{text}"
            if len(block) > budget:
                break
            parts.append(block)
            budget -= len(block)
        return "".join(parts)


async def _run_process(
    *cmd: str,
    cwd: Path | None = None,
    stdin: str | None = None,
    redact: "str | tuple[str, ...] | None" = None,
) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(
        stdin.encode() if stdin is not None else None
    )
    if proc.returncode != 0:
        # Redact secrets from BOTH the command and git's stderr — the failing
        # command line carries the auth header, and this text is returned in
        # the failed ToolResult.
        cmd_str = _redact(" ".join(cmd), redact)
        stderr = _redact(err.decode().strip(), redact)
        raise RuntimeError(f"`{cmd_str}` failed ({proc.returncode}): {stderr}")
    return out.decode()


def _basic_auth(token: str) -> str:
    return base64.b64encode(f"x-access-token:{token}".encode()).decode()


def _auth_config(token: str) -> tuple[str, str]:
    """Git ``-c`` flags carrying auth for ONE command, persisted nowhere.

    ``http.extraHeader`` applies only to the invocation it is passed to — unlike
    a token-in-URL clone it never lands in ``.git/config``, so the workspace
    handed to the worker contains no credential.
    """
    return ("-c", f"http.extraHeader=AUTHORIZATION: basic {_basic_auth(token)}")


def _auth_secrets(token: str) -> tuple[str, str]:
    """Every form of the credential that could surface in git output."""
    return (token, _basic_auth(token))


def _redact(text: str, secrets: "str | tuple[str, ...] | None") -> str:
    if isinstance(secrets, str):
        secrets = (secrets,)
    for secret in secrets or ():
        if secret:
            text = text.replace(secret, "***")
    return text


def _parse_generation(text: str) -> tuple[str, str, str]:
    """Split the model output into (diff, title, body)."""
    title, body, diff = "Automated change", "", ""
    if "DIFF:" in text:
        head, _, diff = text.partition("DIFF:")
        diff = diff.lstrip("\n")
    else:
        head = text
    for line in head.splitlines():
        if line.startswith("TITLE:"):
            title = line[len("TITLE:"):].strip() or title
        elif line.startswith("BODY:"):
            body = line[len("BODY:"):].strip()
    if not diff.strip():
        raise RuntimeError("model returned no diff")
    if not diff.endswith("\n"):
        diff += "\n"
    return diff, title, body
