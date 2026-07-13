"""Claude Code (``claude -p``) as an optional coding-worker backend.

.. warning::

   **EXPERIMENTAL / UNSUPPORTED — personal, self-hosted use only.** This backend
   drives the ``claude`` CLI in headless (``-p``) mode, which authenticates with
   whatever ``claude`` is logged into — including a personal Claude Pro/Max
   **subscription**. Anthropic designs the subscription for individual use
   through its first-party apps, not for powering a shared, multi-user agent
   runtime. Pooling one personal subscription across a team surface is outside
   that intent and may violate the Consumer Terms; it also puts the account at
   risk. Availability rides on a private CLI/auth contract that can change
   between ``claude`` releases. Keep it off by default, use it solo, and fall
   back to ``builtin``/``openhands`` (metered API keys) or a local model for
   anything shared or production. The backend seam makes that switch one env var.

:class:`ClaudeCodeCodingWorker` implements the same credential-free
:class:`~openloop.tools.coding_worker.CodingWorker` protocol as the other
backends: it receives a *prepared* workspace (already cloned, on the job branch)
from :class:`~openloop.tools.coding_worker.GitWorkspaceOrchestrator` and only
edits files. It never clones, commits, or pushes — the orchestrator owns every
credential-bearing git operation, so the agent runs with no git credential in
scope *by construction*. A rogue ``git push`` from the agent simply fails: the
workspace's ``.git/config`` carries a plain URL and no auth.

**Spend control — the "C" stance.** Under a subscription the dollar signal is
unreliable (``total_cost_usd`` is an API-*equivalent estimate*, sometimes ``0``),
so the load-bearing fail-closed bound is **resource-based, not dollar-based**:
every run is capped by ``--max-turns`` *and* a hard wall-clock deadline (the
subprocess is killed on expiry). The
:class:`~openloop.usage.ledger.WorkerSpendLedger` still rides along — it records
the estimated cost/tokens for the audit trail and enforces the invoking agent's
``per_task_usd`` *when the estimate is non-zero* — but the guarantee that an
attempt cannot run unbounded no longer depends on that estimate. Because turns +
deadline are the real backstop, this backend does **not** require a per-task
dollar cap to register (unlike ``openhands``); :meth:`probe` instead refuses to
run without a positive turn cap and deadline.

Only ``CODING_WORKER_SANDBOX=host`` is supported today: a containerized
``claude`` (workspace mounted, subscription credentials forwarded) is not yet
implemented, and the wiring fails **closed** rather than silently running on the
host when ``docker`` is requested.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from openloop.tools.coding_worker import StepCallback, WorkerEdit, WorkerRunAborted

if TYPE_CHECKING:
    from openloop.tools.coding_worker import WorkerState

logger = logging.getLogger(__name__)

# Same filesystem handoff the OpenHands backend uses: the agent's last act is to
# write this file (first line = PR title, rest = body). The worker reads and
# DELETES it before returning, so it never lands in the commit. Filesystem is
# the one interface stable across CLI versions — we never parse the event stream.
PR_FILE = "OPENLOOP_PR.md"

# A runner takes (cmd, cwd, timeout_seconds) and returns
# (returncode, stdout, stderr, timed_out). Injectable so tests never spawn the
# real `claude` binary; the default enforces the deadline by killing the process.
Runner = Callable[
    [list[str], Path, "float | None"], "Awaitable[tuple[int, str, str, bool]]"
]


class ClaudeCodeUnavailable(RuntimeError):
    """The Claude Code backend cannot run on this host (fail-closed at boot)."""


class ClaudeCodeCodingWorker:
    """Headless-``claude`` worker: run ``claude -p`` over the prepared workspace.

    Credential-free (Phase 2 contract): holds no git token and receives none —
    only the prepared workspace path. The one secret it relies on is Claude
    Code's own login (the subscription/API credentials ``claude`` manages), which
    lives outside this process's git reach entirely.
    """

    def __init__(
        self,
        model: str,
        *,
        claude_bin: str = "claude",
        max_turns: int = 100,
        deadline_seconds: float | None = 600.0,
        permission_mode: str = "acceptEdits",
        extra_args: tuple[str, ...] = (),
        runner: Runner | None = None,
    ) -> None:
        self.model = model
        self.claude_bin = claude_bin
        # --max-turns: the agent-turn ceiling. One half of the load-bearing bound.
        self.max_turns = max_turns
        # Hard wall-clock kill for one run (None/0 → unbounded, which probe()
        # refuses: under a subscription this is the real fail-closed backstop).
        self.deadline_seconds = deadline_seconds
        # Headless permission handling. "acceptEdits" auto-accepts file edits (a
        # conservative default); "bypassPermissions" grants full autonomy (tests,
        # shell) at higher risk — recommended only inside a sandbox.
        self.permission_mode = permission_mode
        self.extra_args = extra_args
        self._runner = runner or self._default_runner

    def probe(self) -> None:
        """Check cheap prerequisites at boot; raise to fail the backend closed.

        Two invariants, both fail-closed:

        - the ``claude`` binary must be resolvable on ``PATH`` (or an absolute
          ``claude_bin``), else the backend cannot run at all;
        - the run must be *bounded* — a positive ``--max-turns`` **and** a
          positive deadline. This is the C-stance guarantee: because the dollar
          cap is non-load-bearing under a subscription, an unbounded run would
          have no fail-closed ceiling at all.
        """
        if shutil.which(self.claude_bin) is None and not Path(self.claude_bin).exists():
            raise ClaudeCodeUnavailable(
                f"the Claude Code CLI ({self.claude_bin!r}) was not found on PATH "
                "— install it and log in to use CODING_WORKER_BACKEND=claude"
            )
        if self.max_turns < 1:
            raise ClaudeCodeUnavailable(
                f"CODING_WORKER_MAX_ITERATIONS must be >= 1 (got {self.max_turns}) "
                "— the claude backend needs a bounded turn cap to fail closed"
            )
        if not self.deadline_seconds or self.deadline_seconds <= 0:
            raise ClaudeCodeUnavailable(
                "CODING_WORKER_DEADLINE_SECONDS must be > 0 for the claude backend "
                "— the wall-clock deadline is its load-bearing fail-closed bound "
                "(the dollar cap is unreliable under a subscription)"
            )

    async def run(
        self,
        workspace: Path,
        state: "WorkerState",
        on_step: StepCallback | None = None,
    ) -> WorkerEdit:
        cmd = self._command(self._prompt(state))
        started = time.monotonic()
        rc, out, err, timed_out = await self._runner(
            cmd, workspace, self.deadline_seconds
        )
        if timed_out:
            # Deadline hit: the bound did its job. Fail the attempt closed — the
            # orchestrator records whatever spend is known (none, on a hard kill)
            # and the workspace teardown discards the partial edit. No push, no PR.
            elapsed = time.monotonic() - started
            raise WorkerRunAborted(
                f"claude run exceeded the {self.deadline_seconds:.0f}s attempt "
                f"deadline (killed after {elapsed:.0f}s)"
            )
        if rc != 0:
            raise RuntimeError(
                f"`{self.claude_bin} -p` exited {rc}: {err.strip()[:500]}"
            )

        cost, prompt_tokens, completion_tokens = self._parse_result(out)
        title, body = self._read_pr_file(workspace, state)
        state.completed_steps.append("edit")
        if on_step is not None:
            await on_step(state)
        return WorkerEdit(
            title=title,
            body=body,
            cost_usd=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _command(self, prompt: str) -> list[str]:
        """The headless invocation. Flags kept minimal and version-stable.

        ``--output-format json`` gives the structured result we map onto
        :class:`WorkerEdit` (text + ``total_cost_usd`` + ``usage``); ``--max-turns``
        is half the fail-closed bound (the deadline is the other half, enforced by
        the runner). The model id is stripped of any ``provider/`` prefix — the
        CLI names Claude models directly (e.g. ``claude-sonnet-4-6``).
        """
        cmd = [
            self.claude_bin,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--max-turns",
            str(self.max_turns),
        ]
        model = self.model.split("/", 1)[-1]
        if model:
            cmd += ["--model", model]
        if self.permission_mode:
            cmd += ["--permission-mode", self.permission_mode]
        cmd += list(self.extra_args)
        return cmd

    def _prompt(self, state: "WorkerState") -> str:
        """The task handed to the agent: the instruction plus hard boundaries.

        The rules restate what the architecture already enforces (no credential
        exists in the workspace to push with) so the agent doesn't waste turns
        trying, and they establish the PR-file handoff.
        """
        return (
            f"You are working in a prepared git checkout of {state.repo} "
            f"(branch {state.branch}, based on {state.base}). "
            "Make the following change:\n\n"
            f"{state.instruction}\n\n"
            "Rules:\n"
            "- Work only inside the current directory.\n"
            "- Do NOT run `git commit`, `git push`, `git checkout`, or touch "
            "any git remote or credential — the platform commits and pushes for "
            "you after you finish.\n"
            "- Run the project's tests if they are quick to run.\n"
            f"- When the change is complete, write a file named `{PR_FILE}` in "
            "the repository root: first line = a one-line pull-request title, the "
            "rest = a short PR description. This file is removed before committing."
        )

    def _parse_result(self, stdout: str) -> tuple[float, int, int]:
        """Map ``claude --output-format json`` onto (cost, prompt, completion).

        Cost is Claude Code's ``total_cost_usd`` — a real charge on an API login,
        an API-*equivalent estimate* (often ``0``) on a subscription. Either way
        it feeds the ledger for the audit trail and the per-task cap; the run's
        fail-closed ceiling is turns + deadline, not this number. A result the CLI
        flags as an error, or output we can't parse, fails the attempt rather than
        shipping a broken or unaccountable change.
        """
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"could not parse `{self.claude_bin} -p` JSON output: {exc}"
            ) from exc
        if isinstance(data, dict) and data.get("is_error"):
            subtype = data.get("subtype") or data.get("result") or "unknown"
            raise RuntimeError(f"claude reported an error result ({subtype})")

        cost = float(data.get("total_cost_usd") or 0.0)
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        return cost, prompt_tokens, completion_tokens

    def _read_pr_file(self, workspace: Path, state: "WorkerState") -> tuple[str, str]:
        """Consume the PR-metadata handoff file (never committed).

        A run that didn't write it still produced an edit worth reviewing, so
        fall back to an instruction-derived title instead of failing.
        """
        pr_file = workspace / PR_FILE
        try:
            text = pr_file.read_text("utf-8")
        except (FileNotFoundError, OSError):
            logger.warning(
                "claude run for job %s wrote no %s — using fallback title",
                state.job_id, PR_FILE,
            )
            title = state.instruction.strip().splitlines()[0][:72]
            return title or "Automated change", ""
        pr_file.unlink(missing_ok=True)
        lines = text.strip().splitlines() or ["Automated change"]
        title = lines[0].lstrip("# ").strip() or "Automated change"
        body = "\n".join(lines[1:]).strip()
        return title, body

    async def _default_runner(
        self, cmd: list[str], cwd: Path, timeout: float | None
    ) -> tuple[int, str, str, bool]:
        """Spawn ``claude`` over the workspace, killing it on the deadline.

        The hard kill is what makes the deadline a real (not soft) bound — unlike
        the OpenHands host mode, this backend owns the subprocess, so a frozen run
        cannot outlive its ceiling.
        """
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return (-1, "", "", True)
        return (
            proc.returncode or 0,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
            False,
        )
