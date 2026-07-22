"""OpenHands as an optional heavy worker backend (hardening Phase 4).

:class:`OpenHandsCodingWorker` implements the same credential-free
:class:`~openloop.tools.coding_worker.CodingWorker` protocol as the light
diff-apply worker: it receives a *prepared* workspace (already cloned, on the
job branch) from :class:`~openloop.tools.coding_worker.GitWorkspaceOrchestrator`
and only edits files. It never clones, commits, or pushes — the orchestrator
owns every credential-bearing git operation, so the agent runs with no git
credential in scope *by construction*. With cold resume enabled, each
confirmation-bounded segment captures an authenticated cumulative Git delta,
removes its container, and later attaches the persisted conversation in a fresh
container without resending the task.

Two execution modes, following ``CODING_WORKER_SANDBOX``:

- ``host`` — the OpenHands agent loop *and* its tool actions run in this
  (controller) process over the workspace directory. No isolation; same trust
  level as the light worker's :class:`~openloop.sandbox.HostSandbox`.
- ``docker`` — an external broker owns the container and exposes only an
  authenticated workspace protocol to this process. The app never launches or
  probes containers directly.

Spend control: the OpenHands SDK v1 dropped the old ``max_budget_per_task``
in-run cap, so the in-run knob passed here is ``max_iteration_per_run``; the
authoritative budget cap is the Phase 4
:class:`~openloop.usage.ledger.WorkerSpendLedger` in the orchestrator, which
records this worker's metrics and fails the attempt closed — before any
push/PR — when the per-task budget is exceeded. Wiring therefore refuses to
register this backend without a per-task budget.

The SDK import is lazy (``openhands`` extra); :meth:`probe` checks SDK/tool and
broker-adapter compatibility so common missing prerequisites disable the coding
worker loudly before approval.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from openloop.openhands.runtime_profile import (
    DEFAULT_OPENHANDS_SERVER_IMAGE,
    OpenHandsRuntimeProfileError,
)
from openloop.openhands.workspace_protocol import OpenHandsWorkspace
from openloop.tools.coding_worker import StepCallback, WorkerEdit, WorkerRunAborted
from openloop.tools.openhands_broker_workspace import BrokerWorkspaceError
from openloop.tools.openhands_resume import (
    OPENHANDS_REJECTION_REASON,
    OpenHandsResumeError,
    OpenHandsResumeState,
    WorkerPaused,
    WorkspaceArtifactRef,
)
from openloop.tools.openhands_relay import (
    OpenHandsRelayError,
    probe_relay_compatibility,
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


@dataclass(slots=True)
class _ColdRuntime:
    conversation: object
    workspace: object
    cleanup: Callable[[], None]


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
        docker_adapter: OpenHandsWorkspace | None = None,
        artifact_store: "WorkspaceArtifactStore | None" = None,
        cold_resume_enabled: bool = False,
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
        self.cold_resume_enabled = cold_resume_enabled
        self._factory = conversation_factory or self._build_conversation

    def probe(self) -> None:
        """Check cheap backend prerequisites at boot.

        Raises :class:`OpenHandsUnavailable` when the ``openhands`` extra is
        missing (or, in broker mode, when the workspace adapter is unusable),
        so wiring can fail closed for common setup mistakes. Runtime image and
        provider auth are validated by the broker-owned attempt.
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
            if self._docker_adapter is None:
                raise OpenHandsUnavailable(
                    "the OpenHands Docker adapter is not configured"
                )
            try:
                self._docker_adapter.probe()
            except (OpenHandsRuntimeProfileError, BrokerWorkspaceError) as exc:
                raise OpenHandsUnavailable(str(exc)) from exc
            try:
                probe_relay_compatibility()
            except OpenHandsRelayError as exc:
                raise OpenHandsUnavailable(
                    f"native OpenHands relay compatibility check failed: {exc}"
                ) from exc

    async def run(
        self,
        workspace: Path,
        state: "WorkerState",
        on_step: StepCallback | None = None,
    ) -> WorkerEdit | WorkerPaused:
        if self.cold_resume_enabled:
            return await self._run_cold(workspace, state, on_step)

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

    async def _run_cold(
        self,
        workspace: Path,
        state: "WorkerState",
        on_step: StepCallback | None,
    ) -> WorkerEdit | WorkerPaused:
        """Drive one durable OpenHands segment in a disposable container."""
        if not self.docker or self._docker_adapter is None or self.artifact_store is None:
            raise OpenHandsResumeError(
                "OpenHands cold resume requires the hardened Docker adapter "
                "and encrypted artifact store"
            )

        resume = state.openhands_resume
        prepare_broker_job = getattr(self._docker_adapter, "prepare_job", None)
        if resume is None:
            resolved_base = await asyncio.to_thread(self._git_head, workspace)
            broker_identity = None
            if callable(prepare_broker_job):
                broker_identity = await asyncio.to_thread(
                    prepare_broker_job,
                    state.job_id,
                    current_generation=0,
                )
            resume = OpenHandsResumeState(
                status="running",
                conversation_id=(
                    str(broker_identity.conversation_id)
                    if broker_identity is not None
                    else str(uuid.uuid4())
                ),
                segment_id=uuid.uuid4().hex,
                base_ref=state.base,
                resolved_base_commit=resolved_base,
                image_digest=self.server_image,
                master_key_id=self.artifact_store.keys.master_key_id,
                broker_job_id=(
                    str(broker_identity.broker_job_id)
                    if broker_identity is not None
                    else None
                ),
                slack_requester_id=state.requester_id,
            )
            state.openhands_resume = resume
            # The IDs and immutable base are durable before container creation,
            # making the artifact key deterministic across a crash.
            if on_step is not None:
                await on_step(state)
        elif resume.status not in {"running", "resuming"}:
            raise OpenHandsResumeError(
                f"cannot execute OpenHands segment in {resume.status!r} state"
            )
        elif callable(prepare_broker_job):
            await asyncio.to_thread(
                prepare_broker_job,
                state.job_id,
                broker_job_id=resume.broker_job_id,
                current_generation=resume.broker_generation or 0,
            )

        loop = asyncio.get_running_loop()
        heartbeat = self._heartbeat(loop, state, on_step)
        checkpoint = self._checkpoint_bridge(loop, state, on_step)
        result = await asyncio.to_thread(
            self._drive_cold,
            workspace,
            state,
            heartbeat,
            state.budget_usd,
            checkpoint,
        )
        if isinstance(result, WorkerEdit):
            state.completed_steps.append("edit")
            if on_step is not None:
                await on_step(state)
        return result

    @staticmethod
    def _git_head(workspace: Path) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            raise OpenHandsResumeError("prepared checkout has an invalid base commit")
        return commit

    def _drive_cold(
        self,
        workspace: Path,
        state: "WorkerState",
        heartbeat,
        cost_limit: float | None,
        checkpoint,
    ) -> WorkerEdit | WorkerPaused:
        """Run and capture one fresh/resumed segment before tearing it down."""
        resume = state.openhands_resume
        assert resume is not None
        conv_box: list = []
        started = time.monotonic()

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
            if (
                self.deadline_seconds
                and time.monotonic() - started > self.deadline_seconds
            ):
                raise _RunAborted(
                    f"exceeded the {self.deadline_seconds:.0f}s attempt deadline"
                )

        runtime = self._open_cold_runtime(workspace, state, [guard])
        conversation = runtime.conversation
        conv_box.append(conversation)
        try:
            broker_identity = getattr(
                self._docker_adapter, "generation_identity", None
            )
            if callable(broker_identity):
                identity = broker_identity(state.job_id)
                if (
                    str(identity.conversation_id) != resume.conversation_id
                    or (
                        resume.broker_job_id is not None
                        and str(identity.broker_job_id) != resume.broker_job_id
                    )
                    or identity.generation is None
                ):
                    raise OpenHandsResumeError(
                        "broker generation identity does not match worker state"
                    )
                resume.broker_job_id = str(identity.broker_job_id)
                resume.broker_generation = identity.generation
                checkpoint()
            if resume.status == "running":
                conversation.send_message(self._prompt(state))
                # Remote agent initialization is lazy and may replace policy
                # state, so confirmation is installed only after send_message.
                from openhands.sdk.security.confirmation_policy import AlwaysConfirm

                conversation.set_confirmation_policy(AlwaysConfirm())
            else:
                decision = resume.resolved_decision
                if decision is None:
                    raise OpenHandsResumeError(
                        "resuming segment has no structured decision"
                    )
                if decision.kind == "reject":
                    conversation.reject_pending_actions(OPENHANDS_REJECTION_REASON)

            try:
                conversation.run()
            except _RunAborted as abort:
                cost, pt, ct = self._safe_metrics(conversation)
                raise WorkerRunAborted(
                    abort.reason,
                    cost_usd=cost,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                ) from abort

            cost, prompt_tokens, completion_tokens = self._metrics(conversation)
            status = self._execution_status(conversation)
            if status == "waiting":
                summary, fingerprint = self._pending_action(conversation)
                decision_id = uuid.uuid4().hex
                artifact = self._capture_artifact(
                    runtime.workspace,
                    state,
                    kind="paused",
                    barrier_id=resume.segment_id,
                )
                paused = WorkerPaused(
                    conversation_id=resume.conversation_id,
                    segment_id=resume.segment_id,
                    decision_id=decision_id,
                    pending_action_summary=summary,
                    pending_action_fingerprint=fingerprint,
                    workspace_artifact=artifact,
                    cumulative_cost=cost,
                    cumulative_prompt_tokens=prompt_tokens,
                    cumulative_completion_tokens=completion_tokens,
                )
                if resume.broker_job_id is not None:
                    resume.transition_to(
                        "parking",
                        decision_id=decision_id,
                        pending_action_summary=summary,
                        pending_action_fingerprint=fingerprint,
                        workspace_artifact=artifact,
                        cumulative_cost=cost,
                        cumulative_prompt_tokens=prompt_tokens,
                        cumulative_completion_tokens=completion_tokens,
                        resolved_event_id=None,
                        resolved_decision=None,
                    )
                    checkpoint()
                    self._complete_broker_checkpoint(
                        state, resume.segment_id, artifact, terminal=False
                    )
                return paused
            if status != "finished":
                raise OpenHandsResumeError(
                    "OpenHands segment returned an unsupported execution status"
                )

            if resume.broker_job_id is not None:
                title, body, artifact = self._capture_broker_terminal(
                    runtime.workspace, workspace, state, resume.segment_id
                )
                resume.transition_to(
                    "finalizing",
                    workspace_artifact=artifact,
                    cumulative_cost=cost,
                    cumulative_prompt_tokens=prompt_tokens,
                    cumulative_completion_tokens=completion_tokens,
                )
                checkpoint()
                self._complete_broker_checkpoint(
                    state, resume.segment_id, artifact, terminal=True
                )
                resume.transition_to("terminal")
                checkpoint()
            else:
                title, body = self._read_pr_file(workspace, state)
                artifact = self._capture_artifact(
                    runtime.workspace,
                    state,
                    kind="final",
                    pr_title=title,
                    pr_body=body,
                )
            return WorkerEdit(
                title=title,
                body=body,
                cost_usd=cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                workspace_artifact=artifact,
            )
        finally:
            close = getattr(conversation, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.warning("openhands conversation close failed", exc_info=True)
            try:
                runtime.cleanup()
            except Exception:  # noqa: BLE001
                logger.warning("openhands runtime cleanup failed", exc_info=True)

    def _open_cold_runtime(
        self, workspace: Path, state: "WorkerState", callbacks: list
    ) -> _ColdRuntime:
        """Create a disposable server and create/attach the durable conversation."""
        from openhands.sdk import Conversation

        assert self._docker_adapter is not None
        resume = state.openhands_resume
        assert resume is not None
        agent = self._build_agent()
        target = self._docker_adapter.create(workspace, state.job_id)
        try:
            conversation_id = uuid.UUID(resume.conversation_id)
            if resume.status == "running":
                conversation_kwargs = dict(
                    agent=agent,
                    workspace=target,
                    conversation_id=conversation_id,
                    callbacks=callbacks,
                    max_iteration_per_run=self.max_iterations,
                    visualizer=None,
                    delete_on_close=False,
                )
                websocket_factory = getattr(
                    self._docker_adapter, "websocket_factory", None
                )
                if callable(websocket_factory):
                    conversation_kwargs["websocket_client_factory"] = (
                        websocket_factory(state.job_id)
                    )
                    # The public Conversation factory does not accept the
                    # pinned fork's injected WebSocket transport parameter.
                    # Broker workspaces are remote by construction, so select
                    # the concrete remote implementation directly.
                    from openhands.sdk.conversation.impl.remote_conversation import (
                        RemoteConversation,
                    )

                    conversation = RemoteConversation(**conversation_kwargs)
                else:
                    conversation = Conversation(**conversation_kwargs)
            else:
                conversation = self._docker_adapter.attach_conversation(
                    target,
                    agent=agent,
                    conversation_id=conversation_id,
                    callbacks=callbacks,
                    max_iterations=self.max_iterations,
                )
        except BaseException:
            close_workspace = getattr(
                self._docker_adapter, "close_workspace", None
            )
            if callable(close_workspace):
                close_workspace(target)
            else:
                target.cleanup()
            raise
        close_workspace = getattr(self._docker_adapter, "close_workspace", None)
        cleanup = (
            (lambda: close_workspace(target))
            if callable(close_workspace)
            else target.cleanup
        )
        return _ColdRuntime(conversation, target, cleanup)

    def _build_agent(self):
        from openhands.sdk import LLM, Agent, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.terminal import TerminalTool

        return Agent(
            llm=LLM(model=self.model, api_key=self._api_key),
            tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
        )

    @staticmethod
    def _metrics(conversation) -> tuple[float, int, int]:
        metrics = conversation.conversation_stats.get_combined_metrics()
        usage = metrics.accumulated_token_usage
        return (
            float(metrics.accumulated_cost or 0.0),
            int(usage.prompt_tokens or 0) if usage else 0,
            int(usage.completion_tokens or 0) if usage else 0,
        )

    @staticmethod
    def _execution_status(conversation) -> str:
        status = conversation.state.execution_status
        rendered = f"{getattr(status, 'name', '')} {getattr(status, 'value', '')} {status}"
        if "WAITING_FOR_CONFIRMATION" in rendered.upper():
            return "waiting"
        if "FINISHED" in rendered.upper():
            return "finished"
        return "unknown"

    @staticmethod
    def _pending_action(conversation) -> tuple[str, str]:
        events = list(getattr(conversation.state, "events", ()))
        action = next(
            (event for event in reversed(events) if "Action" in type(event).__name__),
            None,
        )
        if action is None:
            raise OpenHandsResumeError(
                "OpenHands confirmation wait has no pending action event"
            )
        tool = getattr(action, "tool_name", None) or getattr(action, "name", None)
        tool = str(tool) if tool is not None else type(action).__name__
        safe_tool = re.sub(r"[^A-Za-z0-9_. -]", "", tool).strip()[:100]
        summary = f"OpenHands requests confirmation for {safe_tool or 'a tool action'}"
        fingerprint = hashlib.sha256(repr(action).encode("utf-8")).hexdigest()
        return summary, fingerprint

    def _capture_artifact(
        self,
        runtime_workspace: object,
        state: "WorkerState",
        *,
        kind: str,
        pr_title: str | None = None,
        pr_body: str | None = None,
        barrier_id: str | None = None,
    ) -> WorkspaceArtifactRef:
        from openloop.tools.openhands_artifacts import (
            WorkspaceArtifactIdentity,
            WorkspaceArtifactManifest,
        )

        assert self._docker_adapter is not None
        assert self.artifact_store is not None
        resume = state.openhands_resume
        assert resume is not None
        checkpoint_identity = getattr(
            self._docker_adapter, "checkpoint_identity", None
        )
        identity = (
            checkpoint_identity(state.job_id, barrier_id)
            if callable(checkpoint_identity) and barrier_id is not None
            else WorkspaceArtifactIdentity(
                state.job_id,
                resume.conversation_id,
                resume.segment_id,
                kind,
            )
        )
        with tempfile.TemporaryFile(mode="w+b") as plaintext:
            archived = self._docker_adapter.stream_git_delta(
                runtime_workspace,
                plaintext,
                base_ref=resume.resolved_base_commit,
            )
            if archived.base_commit != resume.resolved_base_commit:
                raise OpenHandsResumeError("OpenHands artifact base commit mismatch")
            plaintext.seek(0)
            descriptor = self.artifact_store.put_atomic(
                identity,
                plaintext,
                WorkspaceArtifactManifest(
                    format="git-delta",
                    base_commit=archived.base_commit,
                    pr_title=pr_title,
                    pr_body=pr_body,
                ),
            )
        return WorkspaceArtifactRef(
            artifact=descriptor,
            format="git-delta",
            base_commit=archived.base_commit,
        )

    def _complete_broker_checkpoint(
        self,
        state: "WorkerState",
        barrier_id: str,
        artifact: WorkspaceArtifactRef,
        *,
        terminal: bool,
    ) -> None:
        assert self._docker_adapter is not None
        publish = getattr(self._docker_adapter, "publish_checkpoint", None)
        if not callable(publish):
            raise OpenHandsResumeError("broker checkpoint publisher is unavailable")
        self._docker_adapter.quiesce(state.job_id, barrier_id)
        receipt = publish(state.job_id, barrier_id, artifact.artifact)
        if terminal:
            self._docker_adapter.finalize(state.job_id, receipt)
        else:
            self._docker_adapter.park(state.job_id, receipt)

    def _capture_broker_terminal(
        self,
        runtime_workspace: object,
        workspace: Path,
        state: "WorkerState",
        barrier_id: str,
    ) -> tuple[str, str, WorkspaceArtifactRef]:
        """Reconstruct broker scratch locally, sanitize PR metadata, checkpoint."""
        from openloop.tools.openhands_artifacts import WorkspaceArtifactManifest

        assert self._docker_adapter is not None
        assert self.artifact_store is not None
        resume = state.openhands_resume
        assert resume is not None
        with tempfile.TemporaryFile(mode="w+b") as remote_delta:
            archived = self._docker_adapter.stream_git_delta(
                runtime_workspace,
                remote_delta,
                base_ref=resume.resolved_base_commit,
            )
            if archived.base_commit != resume.resolved_base_commit:
                raise OpenHandsResumeError("OpenHands artifact base commit mismatch")
            remote_delta.seek(0)
            patch = remote_delta.read()
        self._run_git_bytes(workspace, ("reset", "--hard", "HEAD"))
        self._run_git_bytes(workspace, ("clean", "-fdx"))
        if patch:
            self._run_git_bytes(workspace, ("apply", "--binary", "-"), patch)
        title, body = self._read_pr_file(workspace, state)
        self._run_git_bytes(workspace, ("add", "-N", "--", "."))
        sanitized = self._run_git_bytes(
            workspace,
            ("diff", "--binary", "--full-index", "--no-ext-diff", "HEAD", "--", "."),
            capture=True,
        )
        identity = self._docker_adapter.checkpoint_identity(
            state.job_id, barrier_id
        )
        # ``put_atomic`` consumes a stream; publish the sanitized bytes through a
        # bounded temporary file instead of ever placing plaintext in the store.
        with tempfile.TemporaryFile(mode="w+b") as plaintext:
            plaintext.write(sanitized)
            plaintext.seek(0)
            descriptor = self.artifact_store.put_atomic(
                identity,
                plaintext,
                WorkspaceArtifactManifest(
                    format="git-delta",
                    base_commit=archived.base_commit,
                    pr_title=title,
                    pr_body=body,
                ),
            )
        return title, body, WorkspaceArtifactRef(
            artifact=descriptor,
            format="git-delta",
            base_commit=archived.base_commit,
        )

    @staticmethod
    def _run_git_bytes(
        workspace: Path,
        args: tuple[str, ...],
        stdin: bytes | None = None,
        *,
        capture: bool = False,
    ) -> bytes:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace,
            input=stdin,
            check=True,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return result.stdout if capture else b""

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

    @staticmethod
    def _checkpoint_bridge(
        loop, state: "WorkerState", on_step: StepCallback | None
    ):
        """Synchronously persist an intent before the worker thread has effects."""
        if on_step is None:
            return lambda: None

        def persist() -> None:
            future = asyncio.run_coroutine_threadsafe(on_step(state), loop)
            future.result(timeout=30.0)

        return persist

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
