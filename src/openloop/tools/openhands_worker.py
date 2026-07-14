"""OpenHands as an optional heavy worker backend (hardening Phase 4).

:class:`OpenHandsCodingWorker` implements the same credential-free
:class:`~openloop.tools.coding_worker.CodingWorker` protocol as the light
diff-apply worker: it receives a *prepared* workspace (already cloned, on the
job branch) from :class:`~openloop.tools.coding_worker.GitWorkspaceOrchestrator`
and only edits files. It never clones, commits, or pushes — the orchestrator
owns every credential-bearing git operation, so the agent runs with no git
credential in scope *by construction*, and the whole run stays the one opaque
replay unit (a resume re-provisions and re-runs; force-push keeps it
idempotent).

Two execution modes, following ``CODING_WORKER_SANDBOX``:

- ``host`` — the OpenHands agent loop *and* its tool actions run in this
  (controller) process over the workspace directory. No isolation; same trust
  level as the light worker's :class:`~openloop.sandbox.HostSandbox`.
- ``docker`` — the **mounted-workspace pattern** the roadmap calls for: the
  agent server runs in its own container with the workspace bind-mounted at
  ``/workspace`` (``DockerWorkspace``), no host environment forwarded. The
  container edits files through the mount; the host keeps filesystem access
  and pushes. Unlike the light worker's ``--network none`` sandbox, this
  container needs egress to the model provider — point
  ``CODING_WORKER_OPENHANDS_NETWORK`` at an egress-proxy network to allowlist
  it.

Spend control: the OpenHands SDK v1 dropped the old ``max_budget_per_task``
in-run cap, so the in-run knob passed here is ``max_iteration_per_run``; the
authoritative budget cap is the Phase 4
:class:`~openloop.usage.ledger.WorkerSpendLedger` in the orchestrator, which
records this worker's metrics and fails the attempt closed — before any
push/PR — when the per-task budget is exceeded. Wiring therefore refuses to
register this backend without a per-task budget.

The SDK import is lazy (``openhands`` extra); :meth:`probe` checks SDK/tool
imports and, in docker mode, daemon reachability so common missing prerequisites
disable the coding worker loudly before approval. The real
``DockerWorkspace`` is still constructed per attempt because starting the
agent-server image is comparatively expensive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from openloop.tools.coding_worker import StepCallback, WorkerEdit, WorkerRunAborted
from openloop.tools.openhands_docker import (
    DEFAULT_OPENHANDS_SERVER_IMAGE,
    HardenedDockerWorkspace,
    HardenedDockerWorkspaceError,
)

if TYPE_CHECKING:
    from openloop.tools.coding_worker import WorkerState
    from openloop.tools.openhands_artifacts import WorkspaceArtifactStore

logger = logging.getLogger(__name__)


class _RunAborted(Exception):
    """Signal raised from the SDK callback to stop ``conversation.run()`` early.

    Internal to the worker: ``_drive`` catches it, reads the spend accrued so
    far, and re-raises as :class:`WorkerRunAborted` for the orchestrator.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

# The handoff protocol for PR metadata: the agent's last required act is to
# write this file (first line = PR title, rest = body). The worker reads and
# DELETES it before returning, so it never lands in the commit. This keeps the
# integration on the filesystem — the one interface that is stable across SDK
# versions — instead of parsing the agent's event stream.
PR_FILE = "OPENLOOP_PR.md"

# Checkpoint heartbeats while the agent works: at most one on_step bridge per
# this many seconds, so a chatty agent doesn't turn every LLM event into a
# checkpoint write.
_HEARTBEAT_SECONDS = 5.0

_DEFAULT_SERVER_IMAGE = DEFAULT_OPENHANDS_SERVER_IMAGE

# A conversation factory takes (workspace, callbacks, job_id) and returns a
# Conversation-shaped object (send_message / run / conversation_stats /
# close) plus a cleanup callable that tears down whatever runtime the factory
# started (the docker agent-server container; a no-op for local mode).
# Cleanup is separate from close() on purpose: in the SDK,
# RemoteConversation.close() deliberately does NOT stop the workspace — the
# DockerWorkspace owns its container (started at construction) and only its
# own cleanup() reaps it. Injectable so tests never import the heavy SDK.
ConversationFactory = Callable[
    [Path, list, str], "tuple[object, Callable[[], None]]"
]


class OpenHandsUnavailable(RuntimeError):
    """The OpenHands backend cannot run on this host (fail-closed at boot)."""


class OpenHandsCodingWorker:
    """Heavy agentic worker: drive an OpenHands conversation over the workspace.

    Credential-free (Phase 2 contract): holds no token and receives none —
    only the prepared workspace path. The one secret it does hold is the LLM
    provider key for its own model calls; in ``docker`` mode that key reaches
    the agent-server container (the agent loop runs there), but never a git
    credential — the workspace's ``.git/config`` carries a plain URL and the
    host-side orchestrator does all remote operations.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_iterations: int = 100,
        deadline_seconds: float | None = None,
        docker: bool = False,
        server_image: str = _DEFAULT_SERVER_IMAGE,
        network: str | None = None,
        docker_adapter: HardenedDockerWorkspace | None = None,
        artifact_store: "WorkspaceArtifactStore | None" = None,
        conversation_factory: ConversationFactory | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.max_iterations = max_iterations
        # Soft wall-clock ceiling for one run, checked between agent events
        # (None/0 disables). Cannot interrupt a frozen single call — docker mode
        # hard-kills for that; here it bounds a slow-but-responsive run.
        self.deadline_seconds = deadline_seconds
        self.docker = docker
        self.server_image = server_image
        # None = docker's default bridge (the agent server needs egress to the
        # model provider); set to an egress-proxy network for an allowlist.
        self.network = network
        self._docker_adapter = docker_adapter
        self.artifact_store = artifact_store
        self._factory = conversation_factory or self._build_conversation

    def probe(self) -> None:
        """Check cheap backend prerequisites at boot.

        Raises :class:`OpenHandsUnavailable` when the ``openhands`` extra is
        missing (or, in docker mode, when the workspace package or the docker
        daemon is unusable), so wiring can fail closed for common setup
        mistakes. Docker image pull/start and provider auth are still validated
        by the real per-attempt conversation.
        """
        try:
            import openhands.sdk  # noqa: F401
            import openhands.tools.file_editor  # noqa: F401
            import openhands.tools.terminal  # noqa: F401
        except ImportError as exc:
            raise OpenHandsUnavailable(
                "the OpenHands SDK is not installed — install the `openhands` "
                f"extra to use CODING_WORKER_BACKEND=openhands ({exc})"
            ) from exc
        if self.docker:
            try:
                import openhands.workspace  # noqa: F401
            except ImportError as exc:
                raise OpenHandsUnavailable(
                    "openhands-workspace is not installed — required for the "
                    f"docker-mounted OpenHands runtime ({exc})"
                ) from exc
            if self._docker_adapter is None:
                raise OpenHandsUnavailable(
                    "the hardened OpenHands Docker adapter is not configured"
                )
            try:
                self._docker_adapter.probe()
            except HardenedDockerWorkspaceError as exc:
                raise OpenHandsUnavailable(str(exc)) from exc
            import subprocess

            try:
                subprocess.run(
                    ["docker", "version", "--format", "{{.Server.Version}}"],
                    check=True, capture_output=True, timeout=10,
                )
            except Exception as exc:
                raise OpenHandsUnavailable(
                    f"docker is not usable ({exc}); the OpenHands docker "
                    "runtime cannot start"
                ) from exc

    async def run(
        self,
        workspace: Path,
        state: "WorkerState",
        on_step: StepCallback | None = None,
    ) -> WorkerEdit:
        loop = asyncio.get_running_loop()
        heartbeat = self._heartbeat(loop, state, on_step)
        cost, prompt_tokens, completion_tokens = await asyncio.to_thread(
            self._drive,
            workspace,
            self._prompt(state),
            heartbeat,
            state.budget_usd,
            state.job_id,
        )
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

    def _drive(
        self,
        workspace: Path,
        prompt: str,
        heartbeat,
        cost_limit: float | None,
        job_id: str,
    ) -> tuple[float, int, int]:
        """Run the conversation to completion (worker thread; SDK is sync).

        A guard runs on every agent event: it heartbeats, and stops the run early
        (raising :class:`_RunAborted`) once accrued spend crosses ``cost_limit``
        or wall-time crosses the deadline — the only in-thread hook that can
        interrupt the sync ``run()``. It depends on the SDK invoking callbacks and
        propagating their exceptions; if it doesn't, the run finishes and the
        post-run ledger settle still fails an over-budget attempt closed.

        Final metrics are read without defensive fallbacks on purpose: if the
        SDK's stats API drifts, this raises and the attempt fails — silently
        reporting $0 would waltz straight past the fail-closed spend cap.
        """
        # Per-call holder (never instance state — one worker may drive concurrent
        # attempts in separate threads); the guard reads spend off the live
        # conversation once the factory has produced one.
        conv_box: list = []
        started = time.monotonic()
        deadline = self.deadline_seconds

        def guard(_event) -> None:
            heartbeat()
            conversation = conv_box[0] if conv_box else None
            if conversation is not None and cost_limit is not None:
                spent = self._accumulated_cost(conversation)
                if spent is not None and spent > cost_limit:
                    raise _RunAborted(
                        f"in-run spend ${spent:.4f} reached the "
                        f"${cost_limit:.2f} per-task cap"
                    )
            if deadline and time.monotonic() - started > deadline:
                raise _RunAborted(
                    f"exceeded the {deadline:.0f}s attempt deadline"
                )

        conversation, cleanup = self._factory(workspace, [guard], job_id)
        conv_box.append(conversation)
        try:
            conversation.send_message(prompt)
            try:
                conversation.run()
            except _RunAborted as abort:
                cost, pt, ct = self._safe_metrics(conversation)
                raise WorkerRunAborted(
                    abort.reason, cost_usd=cost, prompt_tokens=pt,
                    completion_tokens=ct,
                ) from abort
            metrics = conversation.conversation_stats.get_combined_metrics()
            usage = metrics.accumulated_token_usage
            return (
                float(metrics.accumulated_cost or 0.0),
                int(usage.prompt_tokens or 0) if usage else 0,
                int(usage.completion_tokens or 0) if usage else 0,
            )
        finally:
            # Close the conversation BEFORE cleaning up the runtime: the
            # workspace owns the HTTP client close() still uses, and cleanup
            # is what actually reaps the docker agent-server container —
            # skipping it leaks a running container per attempt.
            close = getattr(conversation, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001 — teardown must not mask the run
                    logger.warning("openhands conversation close failed", exc_info=True)
            try:
                cleanup()
            except Exception:  # noqa: BLE001 — teardown must not mask the run
                logger.warning("openhands runtime cleanup failed", exc_info=True)

    def _accumulated_cost(self, conversation) -> float | None:
        """Best-effort spend-so-far for the in-run guard (never crashes the run).

        Defensive on purpose — unlike the authoritative final read — so a
        transient or absent stat mid-run doesn't turn every event into a failure;
        the post-run ledger settle stays the fail-closed backstop.
        """
        try:
            metrics = conversation.conversation_stats.get_combined_metrics()
            cost = metrics.accumulated_cost
            return float(cost) if cost is not None else None
        except Exception:  # noqa: BLE001 — a stat blip must not kill the run
            return None

    def _safe_metrics(self, conversation) -> tuple[float, int, int]:
        """Best-effort (cost, prompt, completion) for the abort path.

        Records whatever spend can be read before failing an aborted attempt
        closed; a read failure here degrades to zeros rather than masking the
        abort with an unrelated exception.
        """
        try:
            metrics = conversation.conversation_stats.get_combined_metrics()
            usage = metrics.accumulated_token_usage
            return (
                float(metrics.accumulated_cost or 0.0),
                int(usage.prompt_tokens or 0) if usage else 0,
                int(usage.completion_tokens or 0) if usage else 0,
            )
        except Exception:  # noqa: BLE001 — best-effort audit on an aborted run
            return (0.0, 0, 0)

    def _heartbeat(self, loop, state: "WorkerState", on_step: StepCallback | None):
        """A thread-safe, throttled bridge from SDK events to on_step.

        Streams progress into the checkpoint (same state, no new step — a
        liveness heartbeat, not a step transition) at most once per
        ``_HEARTBEAT_SECONDS``. No-op without a callback.
        """
        if on_step is None:
            return lambda: None
        last = 0.0

        def emit() -> None:
            nonlocal last
            now = time.monotonic()
            if now - last < _HEARTBEAT_SECONDS:
                return
            last = now
            asyncio.run_coroutine_threadsafe(on_step(state), loop)

        return emit

    def _prompt(self, state: "WorkerState") -> str:
        """The task handed to the agent: the instruction plus hard boundaries.

        The rules restate what the architecture already enforces (no
        credential exists in the workspace to push with) so the agent doesn't
        waste iterations trying, and they establish the PR-file handoff.
        """
        return (
            f"You are working in a prepared git checkout of {state.repo} "
            f"(branch {state.branch}, based on {state.base}). "
            "Make the following change:\n\n"
            f"{state.instruction}\n\n"
            "Rules:\n"
            "- Work only inside the current directory.\n"
            "- Do NOT run `git commit`, `git push`, `git checkout`, or touch "
            "any git remote or credential — the platform commits and pushes "
            "for you after you finish.\n"
            "- Run the project's tests if they are quick to run.\n"
            f"- When the change is complete, write a file named `{PR_FILE}` "
            "in the repository root: first line = a one-line pull-request "
            "title, the rest = a short PR description. This file is removed "
            "before committing."
        )

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
                "openhands run for job %s wrote no %s — using fallback title",
                state.job_id, PR_FILE,
            )
            title = state.instruction.strip().splitlines()[0][:72]
            return title or "Automated change", ""
        pr_file.unlink(missing_ok=True)
        lines = text.strip().splitlines() or ["Automated change"]
        title = lines[0].lstrip("# ").strip() or "Automated change"
        body = "\n".join(lines[1:]).strip()
        return title, body

    def _build_conversation(self, workspace: Path, callbacks: list, job_id: str):
        """Construct the real SDK conversation (verified against SDK v1).

        Local mode passes the workspace path (LocalConversation); docker mode
        wraps it in a DockerWorkspace so the agent server runs containerized
        with the workspace bind-mounted at /workspace — the container holds
        no git credential and no host environment.
        """
        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.terminal import TerminalTool

        llm = LLM(model=self.model, api_key=self._api_key)
        agent = Agent(
            llm=llm,
            tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
        )
        cleanup: Callable[[], None] = lambda: None  # noqa: E731
        if self.docker:
            if self._docker_adapter is None:
                raise OpenHandsUnavailable(
                    "the hardened OpenHands Docker adapter is not configured"
                )
            # Constructing the adapter target starts the authenticated,
            # loopback-only agent-server over this job's isolated state mount.
            target: object = self._docker_adapter.create(workspace, job_id)
            cleanup = target.cleanup
        else:
            target = str(workspace)
        try:
            conversation = Conversation(
                agent=agent,
                workspace=target,
                callbacks=callbacks,
                max_iteration_per_run=self.max_iterations,
                visualizer=None,
            )
        except BaseException:
            # The container is already running; don't leak it when the
            # conversation itself fails to construct.
            cleanup()
            raise
        return conversation, cleanup
