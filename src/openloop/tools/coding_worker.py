"""Native coding-worker connector — opens *draft* PRs from an instruction.

Exposes a single write action, ``coding_worker.pr:write``. On execution it runs a
pluggable :class:`CodingWorker` (clone → model-edit → commit → push) and then
opens a **draft** pull request from the pushed branch.

Phase A runs the whole pipeline inside :meth:`execute`, **after** approval. Two
distinct gates, never conflated:

1. the human **approval** lets the worker **start**;
2. the **draft PR itself** is the review gate before **merge**.

There is no "approve the generated diff before opening the PR" step here — the
approval summary says *run worker + open draft PR* and must never imply diff
review.

``job_id`` is the stable thread through the whole system. It is minted in
:meth:`prepare_args` **before** the approval is created, so it is carried in the
approval args, the worker state, the branch name + idempotency keys, and the
final PR metadata — giving one identity even before checkpointing exists
(Phase B). Named steps and :class:`WorkerState` live in code now but are **not**
checkpointed yet; Phase A persists the *outcome only* (in the ToolResult data).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from openloop.tools.base import ActionSpec, ToolResult
from openloop.tools.github import GitHubClient

logger = logging.getLogger(__name__)

_REPO = {"type": "string", "description": "owner/repo, e.g. acme/ingestion"}

# Named steps the worker walks through. In code now; checkpointed in Phase B.
STEPS = ("clone", "branch", "edit", "commit", "push")


@dataclass(slots=True)
class WorkerState:
    """In-process worker progress for one job.

    Phase A does **not** persist this — it lives only for the duration of one
    ``execute`` call. Phase B promotes it into a checkpoint row (``state_json`` +
    ``completed_step_names``) so a mid-flight crash can resume.
    """

    job_id: str
    repo: str
    instruction: str
    base: str
    branch: str
    completed_steps: list[str] = field(default_factory=list)

    def push_key(self) -> str:
        """Idempotency key for the branch push — never a single global key."""
        return f"{self.job_id}:push:{self.branch}"

    def open_pr_key(self) -> str:
        """Idempotency key for opening the PR."""
        return f"{self.job_id}:open_pr:{self.repo}:{self.branch}"


@dataclass(slots=True)
class WorkerOutcome:
    """What the worker produced: a pushed branch ready for a draft PR.

    Carries the model spend so it is at least *observable* in the tool result.
    NOTE: this runs inside ``ToolGateway.resolve()``, which is outside
    ``Runtime.handle``'s usage accounting — so this spend is not yet recorded in
    ``/usage`` nor checked against per-task/monthly budgets. Enforcing that means
    threading a UsageStore + agent scope through the approval-resolution path,
    which lands with Phase B/C (where approval becomes an event on the workflow).
    """

    branch: str
    title: str
    body: str
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class CodingWorker(Protocol):
    """Clones a repo, applies model-generated edits, commits and pushes.

    Implementations must push ``state.branch`` and return a :class:`WorkerOutcome`
    describing the PR to open. They own the ``state.push_key()`` idempotency key.
    """

    async def run(self, state: WorkerState) -> WorkerOutcome: ...


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


def _pr_body(body: str, job_id: str) -> str:
    """Stamp the job id into the PR body so the identity survives in GitHub."""
    body = (body or "").rstrip()
    footer = f"---\n🤖 Opened by the OpenLoop coding worker · job `{job_id}`"
    return f"{body}\n\n{footer}" if body else footer


class CodingWorkerConnector:
    """Maps ``coding_worker.pr:write`` onto a worker + a :class:`GitHubClient`."""

    name = "coding_worker"

    def __init__(self, worker: CodingWorker, github: GitHubClient) -> None:
        self.worker = worker
        self.github = github

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
        repo = args["repo"]
        instruction = args["instruction"]
        base = args.get("base", "main")
        state = WorkerState(
            job_id=job_id,
            repo=repo,
            instruction=instruction,
            base=base,
            branch=_branch_for(job_id),
        )

        # Both side effects (worker push, PR open) run after resolve() has already
        # marked the approval approved, so neither may raise out of execute() —
        # that would surface as a generic error with no failed ToolResult. Record
        # the failure instead. open_pr_failed leaves the branch pushed (no resume
        # until Phase B), which the result makes explicit.
        try:
            outcome = await self.worker.run(state)
        except Exception as exc:  # noqa: BLE001 — record the outcome, don't crash the loop
            return _failed(job_id, state, "failed", exc)

        try:
            pull = await self.github.create_pull(
                repo=repo,
                head=outcome.branch,
                base=base,
                title=outcome.title,
                body=_pr_body(outcome.body, job_id),
                draft=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _failed(job_id, state, "open_pr_failed", exc)

        return ToolResult(
            ok=True,
            summary=(
                f"opened draft PR #{pull.get('number')} in {repo} (job {job_id})"
            ),
            data={
                "job_id": job_id,
                "status": "opened",
                "branch": outcome.branch,
                "pr_number": pull.get("number"),
                "pr_url": pull.get("html_url"),
                "completed_steps": state.completed_steps,
                # Observable spend only — not yet enforced (see WorkerOutcome).
                "cost_usd": outcome.cost_usd,
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
                "idempotency_keys": {
                    "push": state.push_key(),
                    "open_pr": state.open_pr_key(),
                },
            },
        )


@runtime_checkable
class _Completer(Protocol):
    async def complete(self, model: str, messages: list[dict], **kwargs): ...


class GitCodingWorker:
    """Real worker: clone → model-edit → commit → push, in a temp sandbox.

    SECURITY: this runs model-generated edits, so it needs a least-privilege
    ``contents:write`` token and an isolated checkout. The clone happens in a
    throwaway temp dir that is removed after each run. Edits are applied as a
    unified diff via ``git apply`` — anything that doesn't apply cleanly fails
    the job rather than being force-written.

    Phase A only: no checkpointing. A crash mid-run loses the sandbox and leaves
    the approval stuck approved-but-incomplete — Phase B adds resume.
    """

    def __init__(
        self,
        token: str,
        model: str,
        gateway: _Completer | None = None,
        *,
        max_context_bytes: int = 60_000,
    ) -> None:
        self.token = token
        self.model = model
        self._gateway = gateway
        self.max_context_bytes = max_context_bytes

    def _completer(self) -> _Completer:
        if self._gateway is None:
            from openloop.models.gateway import ModelGateway

            self._gateway = ModelGateway()
        return self._gateway

    async def run(self, state: WorkerState) -> WorkerOutcome:
        sandbox = Path(tempfile.mkdtemp(prefix=f"openloop-{state.job_id}-"))
        try:
            url = (
                f"https://x-access-token:{self.token}@github.com/{state.repo}.git"
            )
            await self._git(
                "clone", "--depth", "1", "--branch", state.base, url, str(sandbox)
            )
            state.completed_steps.append("clone")

            await self._git("checkout", "-b", state.branch, cwd=sandbox)
            state.completed_steps.append("branch")

            diff, title, body, resp = await self._generate(state, sandbox)
            await self._git_input("apply", "--whitespace=nowarn", stdin=diff, cwd=sandbox)
            state.completed_steps.append("edit")

            await self._git("add", "-A", cwd=sandbox)
            await self._git(
                "-c", "user.email=worker@openloop.ai",
                "-c", "user.name=OpenLoop coding worker",
                "commit", "-m", title, cwd=sandbox,
            )
            state.completed_steps.append("commit")

            # Idempotency: re-pushing the same branch (state.push_key()) is a
            # no-op on GitHub, so a replay before checkpointing is still safe.
            await self._git("push", "origin", state.branch, cwd=sandbox)
            state.completed_steps.append("push")

            return WorkerOutcome(
                branch=state.branch,
                title=title,
                body=body,
                cost_usd=resp.cost_usd,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
            )
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    async def _generate(self, state: WorkerState, sandbox: Path):
        """Ask the model for a unified diff + PR title/body for the instruction.

        Returns ``(diff, title, body, response)`` — the response carries token
        counts and cost so the caller can surface the worker's model spend.
        """
        context = self._repo_context(sandbox)
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

    def _repo_context(self, sandbox: Path) -> str:
        """A best-effort, size-capped snapshot of tracked text files."""
        parts: list[str] = []
        budget = self.max_context_bytes
        for path in sorted(sandbox.rglob("*")):
            if ".git" in path.parts or not path.is_file():
                continue
            try:
                text = path.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.relative_to(sandbox)
            block = f"\n=== {rel} ===\n{text}"
            if len(block) > budget:
                break
            parts.append(block)
            budget -= len(block)
        return "".join(parts)

    async def _git(self, *args: str, cwd: Path | None = None) -> str:
        return await self._run("git", *args, cwd=cwd)

    async def _git_input(self, *args: str, stdin: str, cwd: Path | None = None) -> str:
        return await self._run("git", *args, cwd=cwd, stdin=stdin)

    async def _run(
        self, *cmd: str, cwd: Path | None = None, stdin: str | None = None
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
            # Redact the token from BOTH the command and git's stderr — git prints
            # the full https://x-access-token:<token>@github.com/... URL on many
            # failures, and this text is returned in the failed ToolResult.
            cmd_str = self._redact(" ".join(cmd))
            stderr = self._redact(err.decode().strip())
            raise RuntimeError(f"`{cmd_str}` failed ({proc.returncode}): {stderr}")
        return out.decode()

    def _redact(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text


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
