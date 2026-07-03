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
worker sandbox is credential-free by construction. The whole attempt is one
**opaque replay unit** — interrupted before the push boundary, a resume runs a
fresh attempt (re-provision, re-run; a regenerated diff is acceptable); after
it, the branch/PR are reconciled idempotently (force-push + ``find_pull``).
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

from openloop.checkpoints.store import CheckpointStore, WorkerCheckpoint
from openloop.credentials import CredentialResolver, CredentialScope
from openloop.tools.base import ActionSpec, ToolResult
from openloop.tools.github import GitHubClient

if TYPE_CHECKING:
    from openloop.sandbox import Sandbox
    from openloop.usage.ledger import WorkerSpendLedger

# Persist-after-each-step callback invoked after each completed step so a crash
# leaves an accurate mid-phase record (completed_steps + state_json), not just a
# status. The orchestrator owns the git-side steps; the worker reports its own.
StepCallback = Callable[["WorkerState"], Awaitable[None]]

logger = logging.getLogger(__name__)

_REPO = {"type": "string", "description": "owner/repo, e.g. acme/ingestion"}

# Named steps one attempt walks through. The orchestrator owns the git-side
# steps (clone, branch, commit, push); the worker reports its own ("edit").
STEPS = ("clone", "branch", "edit", "commit", "push")


@dataclass(slots=True)
class WorkerState:
    """Worker progress for one job — serialized into a checkpoint's state_json.

    ``title`` / ``body`` are filled once the worker generates the change so they
    survive in the checkpoint; on resume after a push they let the PR be opened
    without re-running the worker.
    """

    job_id: str
    repo: str
    instruction: str
    base: str
    branch: str
    completed_steps: list[str] = field(default_factory=list)
    title: str | None = None
    body: str | None = None

    def push_key(self) -> str:
        """Idempotency key for the branch push — never a single global key."""
        return f"{self.job_id}:push:{self.branch}"

    def open_pr_key(self) -> str:
        """Idempotency key for opening the PR."""
        return f"{self.job_id}:open_pr:{self.repo}:{self.branch}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerState":
        fields = {
            "job_id", "repo", "instruction", "base", "branch",
            "completed_steps", "title", "body",
        }
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass(slots=True)
class WorkerOutcome:
    """What one worker attempt produced: a pushed branch ready for a draft PR.

    Carries the model spend so it is observable in the tool result. With a
    :class:`~openloop.usage.ledger.WorkerSpendLedger` wired into the
    orchestrator (Phase 4), the spend is also recorded to the usage store and
    fail-closed capped per task before the push/PR boundary; monthly budget
    enforcement and full ``/usage`` unification remain Phase 5.
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
    ) -> WorkerEdit: ...


@runtime_checkable
class AttemptRunner(Protocol):
    """What the durable paths depend on: one attempt → pushed branch + outcome.

    :class:`GitWorkspaceOrchestrator` is the real implementation; tests use
    in-memory fakes. The attempt is an opaque replay unit — safe to re-run from
    clean (force-push to the job-exclusive branch keeps the retry idempotent).
    """

    async def run_attempt(
        self, state: WorkerState, on_step: StepCallback | None = None
    ) -> WorkerOutcome: ...


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


def _pr_body(body: str, job_id: str) -> str:
    """Stamp the job id into the PR body so the identity survives in GitHub."""
    body = (body or "").rstrip()
    footer = f"---\n🤖 Opened by the OpenLoop coding worker · job `{job_id}`"
    return f"{body}\n\n{footer}" if body else footer


class CodingWorkerConnector:
    """Maps ``coding_worker.pr:write`` onto an attempt runner + :class:`GitHubClient`."""

    name = "coding_worker"
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
        return {"pr:write"}

    def prepare_args(self, permission: str, args: dict) -> dict:
        """Mint ``job_id`` before approval so it threads the whole system.

        Called by the gateway prior to creating the approval request, so the id
        is persisted in the approval args and reused verbatim at execute time.
        """
        if permission == "pr:write" and not args.get("job_id"):
            args = {**args, "job_id": uuid.uuid4().hex[:12]}
        return args

    def describe(self, permission: str) -> ActionSpec:
        return ActionSpec(
            "Run the coding worker on an instruction and open a draft pull "
            "request with its changes. This starts the worker and opens a draft "
            "PR for review; it does not merge.",
            {
                "type": "object",
                "properties": {
                    "repo": _REPO,
                    "instruction": {
                        "type": "string",
                        "description": "what change the worker should make",
                    },
                    "base": {
                        "type": "string",
                        "description": "branch to open the PR against (default main)",
                    },
                },
                "required": ["repo", "instruction"],
            },
        )

    async def execute(self, permission: str, args: dict) -> ToolResult:
        if permission != "pr:write":
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
            )

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
                outcome = await self.orchestrator.run_attempt(
                    state, on_step=self._checkpointer()
                )
            except Exception as exc:  # noqa: BLE001
                await self._save(state, "failed", error=str(exc))
                return _failed(job_id, state, "failed", exc)
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

    # Statuses that are done (no resume) vs. interrupted (resume on startup).
    _TERMINAL = ("opened", "failed")

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
            if cp.status in self._TERMINAL:
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
            await self._save(state, "running")

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

    The attempt is one opaque replay unit: the workspace is a throwaway temp
    dir removed after each run, so a resumed job re-provisions from clean, and
    force-push to the job-exclusive branch keeps the retry idempotent.

    SPEND (hardening Phase 4): when a
    :class:`~openloop.usage.ledger.WorkerSpendLedger` is wired, every
    attempt's model spend is recorded to the usage store and checked against
    the per-task budget *here* — after the worker's edit, before commit/push —
    so an over-budget attempt fails closed (no push, no PR) on **both**
    durable paths at once. Full budget unification is Phase 5.
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

    async def run_attempt(
        self, state: WorkerState, on_step: StepCallback | None = None
    ) -> WorkerOutcome:
        async def step(name: str) -> None:
            state.completed_steps.append(name)
            if on_step is not None:
                await on_step(state)

        # Resolved fresh per attempt and kept local — never stored, never in a
        # URL. The auth header value still surfaces in a failed command line,
        # so git failures redact both the token and its basic-auth encoding.
        token = await self._credentials.resolve(self._scope)
        if self._workspace_root is not None:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"openloop-{state.job_id}-",
                dir=self._workspace_root,
            )
        )
        try:
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
            edit = await self.worker.run(workspace, state, on_step)
            # Persist title/body so a post-push crash can still open the PR.
            state.title, state.body = edit.title, edit.body

            # Phase 4 gate: record the attempt's spend and fail closed on the
            # per-task cap BEFORE the push/PR boundary. Raises out of the
            # attempt (both durable paths mark the job failed — terminal, so
            # no resume loop re-spends), and the workspace teardown in
            # `finally` discards the over-budget edit.
            if self._ledger is not None:
                await self._ledger.settle(
                    job_id=state.job_id,
                    cost_usd=edit.cost_usd,
                    prompt_tokens=edit.prompt_tokens,
                    completion_tokens=edit.completion_tokens,
                )

            await self._git("add", "-A", cwd=workspace)
            await self._git(
                "-c", "user.email=worker@openloop.ai",
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

    async def _git(
        self,
        *args: str,
        cwd: Path | None = None,
        redact: "str | tuple[str, ...] | None" = None,
    ) -> str:
        return await self._run("git", *args, cwd=cwd, redact=redact)

    async def _run(
        self,
        *cmd: str,
        cwd: Path | None = None,
        stdin: str | None = None,
        redact: "str | tuple[str, ...] | None" = None,
    ) -> str:
        return await _run_process(*cmd, cwd=cwd, stdin=stdin, redact=redact)


class GitCodingWorker:
    """Default light worker: ask the model for a unified diff and apply it.

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
