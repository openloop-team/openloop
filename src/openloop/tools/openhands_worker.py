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

from openloop.tools.coding_worker import StepCallback, WorkerEdit

if TYPE_CHECKING:
    from openloop.tools.coding_worker import WorkerState

logger = logging.getLogger(__name__)

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

_DEFAULT_SERVER_IMAGE = "ghcr.io/openhands/agent-server:latest-python"

# A conversation factory takes (workspace, callbacks) and returns a
# Conversation-shaped object (send_message / run / conversation_stats /
# close) plus a cleanup callable that tears down whatever runtime the factory
# started (the docker agent-server container; a no-op for local mode).
# Cleanup is separate from close() on purpose: in the SDK,
# RemoteConversation.close() deliberately does NOT stop the workspace — the
# DockerWorkspace owns its container (started at construction) and only its
# own cleanup() reaps it. Injectable so tests never import the heavy SDK.
ConversationFactory = Callable[
    [Path, list], "tuple[object, Callable[[], None]]"
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
        docker: bool = False,
        server_image: str = _DEFAULT_SERVER_IMAGE,
        network: str | None = None,
        conversation_factory: ConversationFactory | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.max_iterations = max_iterations
        self.docker = docker
        self.server_image = server_image
        # None = docker's default bridge (the agent server needs egress to the
        # model provider); set to an egress-proxy network for an allowlist.
        self.network = network
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
            self._drive, workspace, self._prompt(state), heartbeat
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

    def _drive(self, workspace: Path, prompt: str, heartbeat) -> tuple[float, int, int]:
        """Run the conversation to completion (worker thread; SDK is sync).

        Metrics are read without defensive fallbacks on purpose: if the SDK's
        stats API drifts, this raises and the attempt fails — silently
        reporting $0 would waltz straight past the fail-closed spend cap.
        """
        conversation, cleanup = self._factory(
            workspace, [lambda event: heartbeat()]
        )
        try:
            conversation.send_message(prompt)
            conversation.run()
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

    def _build_conversation(self, workspace: Path, callbacks: list):
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
            from openhands.workspace import DockerWorkspace

            # Constructing the workspace STARTS the agent-server container;
            # only its own cleanup() stops it (Conversation.close() will not).
            target: object = DockerWorkspace(
                server_image=self.server_image,
                working_dir="/workspace",
                volumes=[f"{workspace}:/workspace:rw"],
                # Forward nothing from the controller environment (the field
                # defaults to forwarding DEBUG; even that is not needed).
                forward_env=[],
                network=self.network,
            )
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
