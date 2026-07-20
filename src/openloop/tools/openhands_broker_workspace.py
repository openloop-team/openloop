"""Broker-backed OpenHands workspace adapter (architecture step 4, phase 3).

Drop-in for the ``docker_adapter`` surface :class:`OpenHandsCodingWorker`
consumes from
:class:`~openloop.tools.openhands_docker.HardenedDockerWorkspace`, but instead
of launching a container itself it drives the reviewed broker RPC intents
(``create_job`` → ``start_segment`` → …) and serves the agent over the
per-generation relay UDS.

The worker calls these methods **synchronously** from its ``asyncio.to_thread``
pool, so the adapter bridges each async broker-client call back onto the app
event loop with ``run_coroutine_threadsafe`` — never call the sync methods from
the loop thread itself (that would deadlock).

This module implements the **forward path** (phase 3a): ``probe``, ``create``
(a fresh running segment), ``stream_git_delta``, and ``attach_conversation``.
The lifecycle-transition intents (``quiesce_segment`` / ``release_segment`` /
``finalize_job`` for park/finish, with a ``receipt_issuer``-signed checkpoint
receipt) are phase 3b and need worker-orchestration hookpoints; ``receipt_issuer``
is threaded through now so that wiring is local when it lands.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from openloop.broker.models import (
    ReleaseTarget,
    SignedCheckpointReceipt,
    TerminalOutcome,
)
from openloop.broker_rpc.capability import JobCapability
from openloop.broker_rpc.client import BrokerRpcClient
from openloop.tools.openhands_docker import ArchiveStreamResult
from openloop.tools.openhands_relay_client import (
    RelayClientEndpoint,
    RelayMode,
    create_relay_workspace,
)

log = logging.getLogger("openloop")

_BRIDGE_TIMEOUT_SECONDS = 30.0

WorkspaceFactory = Callable[[RelayClientEndpoint], object]


class BrokerWorkspaceError(RuntimeError):
    """The broker-backed workspace could not satisfy a worker request."""


@dataclass(slots=True)
class _JobState:
    """Per app coding-job broker identity and generation cursor.

    ``current_generation`` is the *expected* generation for the next
    ``start_segment`` (0 for a fresh job); the broker produces
    ``current_generation + 1`` as the running generation. It advances only when a
    segment is parked (``release_segment``) — phase 3b — so within the forward
    path one job maps to one running generation.
    """

    broker_job_id: UUID
    capability: JobCapability
    current_generation: int = 0
    running_generation: int | None = None
    conversation_id: UUID | None = None


def _create_idempotency_key(job_id: str) -> str:
    # Stable per job so a retried create replays the same broker job.
    return f"broker-create:{job_id}"


def _start_idempotency_key(job_id: str, expected_generation: int) -> str:
    # Stable per (job, generation) so a retried start replays the same segment.
    return f"broker-start:{job_id}:{expected_generation}"


def _quiesce_idempotency_key(job_id: str, generation: int) -> str:
    return f"broker-quiesce:{job_id}:{generation}"


def _release_idempotency_key(job_id: str, generation: int) -> str:
    return f"broker-release:{job_id}:{generation}"


def _finalize_idempotency_key(job_id: str, generation: int) -> str:
    return f"broker-finalize:{job_id}:{generation}"


class BrokerWorkspaceAdapter:
    """Serve an OpenHands agent over broker-owned generations, not a container."""

    # The broker owns the Docker socket; this process never runs containers or
    # dials a local daemon. The worker's probe reads this to skip local-Docker
    # checks (``docker version`` / ``import openhands.workspace``).
    runs_containers_locally = False

    def __init__(
        self,
        *,
        client: BrokerRpcClient,
        loop: asyncio.AbstractEventLoop,
        receipt_issuer: object | None = None,
        workspace_factory: WorkspaceFactory = create_relay_workspace,
        bridge_timeout_seconds: float = _BRIDGE_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._loop = loop
        self._receipt_issuer = receipt_issuer  # phase 3b (checkpoint signing)
        self._workspace_factory = workspace_factory
        self._bridge_timeout = bridge_timeout_seconds
        self._jobs: dict[str, _JobState] = {}

    def _bridge(self, coro):
        """Run an async broker-client call from the worker thread on the app loop."""
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(timeout=self._bridge_timeout)
        except BrokerWorkspaceError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize to the adapter error
            raise BrokerWorkspaceError(str(exc)) from exc

    def probe(self) -> None:
        """Fail loudly before approval if the relay workspace SDK is missing.

        Skipped when a custom ``workspace_factory`` is injected (tests) — the
        broker client itself needs no probe (it is co-process and reachable).
        """
        if self._workspace_factory is create_relay_workspace:
            try:
                from openhands.sdk.workspace import RemoteWorkspace  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                raise BrokerWorkspaceError(
                    "OpenHands relay workspace SDK is unavailable"
                ) from exc

    def create(self, workspace: Path, job_id: str) -> object:
        """Open a fresh running segment and return its relay-backed workspace."""
        state = self._jobs.get(job_id)
        if state is None:
            created = self._bridge(
                self._client.create_job(_create_idempotency_key(job_id))
            )
            state = _JobState(
                broker_job_id=created.ticket.job_id,
                capability=created.capability,
            )
            self._jobs[job_id] = state

        started = self._bridge(
            self._client.start_segment(
                state.broker_job_id,
                state.current_generation,
                _start_idempotency_key(job_id, state.current_generation),
                state.capability,
            )
        )
        access = started.access
        state.running_generation = access.generation
        state.conversation_id = access.conversation_id
        endpoint = RelayClientEndpoint(
            socket_path=access.socket_path,
            conversation_id=access.conversation_id,
            relay_capability=access.relay_capability,
            session_api_key=access.session_api_key,
            mode=RelayMode.RUNNING,
        )
        return self._workspace_factory(endpoint)

    def stream_git_delta(
        self, workspace: object, sink: BinaryIO, *, base_ref: str
    ) -> ArchiveStreamResult:
        """Stream the authenticated cumulative Git delta from the relay workspace."""
        stream = getattr(workspace, "stream_git_delta", None)
        if not callable(stream):
            raise BrokerWorkspaceError(
                "relay workspace does not support git-delta streaming"
            )
        base_commit, written = stream(sink, base_ref=base_ref)
        return ArchiveStreamResult(
            base_commit=base_commit, base_ref=base_ref, bytes_written=written
        )

    def attach_conversation(
        self,
        workspace: object,
        *,
        agent: object,
        conversation_id: UUID,
        callbacks: list | None = None,
        max_iterations: int = 500,
    ) -> object:
        """Attach to an already-loaded persisted conversation (resume path).

        Verifies the conversation exists before calling the SDK constructor —
        pinned ``RemoteConversation`` would otherwise silently create a new
        conversation with the caller's ID after a 404 during a stale lease.
        """
        client = getattr(workspace, "client", None)
        api_key = getattr(workspace, "api_key", None)
        if client is None or not api_key:
            raise BrokerWorkspaceError(
                "authenticated relay workspace client is unavailable"
            )
        response = client.get(f"/api/conversations/{conversation_id}")
        if response.status_code == 404:
            raise BrokerWorkspaceError(
                "persisted OpenHands conversation is not available for attach; "
                "its ownership lease may still be active"
            )
        try:
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise BrokerWorkspaceError(
                "failed to verify persisted OpenHands conversation"
            ) from exc

        from openhands.sdk.conversation.impl.remote_conversation import (
            RemoteConversation,
        )

        return RemoteConversation(
            agent=agent,
            workspace=workspace,
            conversation_id=conversation_id,
            callbacks=callbacks,
            max_iteration_per_run=max_iterations,
            visualizer=None,
            delete_on_close=False,
        )

    # --- lifecycle transitions (phase 3b) --------------------------------
    # These drive the running segment to a durable checkpoint and beyond. The
    # signed receipt is produced by the checkpoint store from the captured
    # artifact (phase 3b-iii); these methods only carry it into the broker.

    def _running(self, job_id: str) -> _JobState:
        state = self._jobs.get(job_id)
        if state is None or state.running_generation is None:
            raise BrokerWorkspaceError(
                f"no running broker segment for job {job_id!r}"
            )
        return state

    def quiesce(self, job_id: str, barrier_id: str) -> None:
        """Quiesce the running segment at a confirmation barrier (checkpoint)."""
        state = self._running(job_id)
        self._bridge(
            self._client.quiesce_segment(
                state.broker_job_id,
                state.running_generation,
                _quiesce_idempotency_key(job_id, state.running_generation),
                barrier_id,
                state.capability,
            )
        )

    def park(self, job_id: str, receipt: SignedCheckpointReceipt) -> None:
        """Release the quiesced segment to PARKED and advance the generation.

        The next :meth:`create` for this job resumes into the following
        generation (cold resume from durable conversation state).
        """
        state = self._running(job_id)
        generation = state.running_generation
        self._bridge(
            self._client.release_segment(
                state.broker_job_id,
                generation,
                _release_idempotency_key(job_id, generation),
                receipt,
                ReleaseTarget.PARKED,
                state.capability,
            )
        )
        state.current_generation = generation
        state.running_generation = None

    def finalize(
        self,
        job_id: str,
        receipt: SignedCheckpointReceipt,
        *,
        outcome: TerminalOutcome = TerminalOutcome.SUCCESS,
    ) -> None:
        """Release the quiesced segment to FINALIZING and finalize the job."""
        state = self._running(job_id)
        generation = state.running_generation
        self._bridge(
            self._client.release_segment(
                state.broker_job_id,
                generation,
                _release_idempotency_key(job_id, generation),
                receipt,
                ReleaseTarget.FINALIZING,
                state.capability,
                terminal_outcome=outcome,
            )
        )
        self._bridge(
            self._client.finalize_job(
                state.broker_job_id,
                generation,
                _finalize_idempotency_key(job_id, generation),
                outcome,
                state.capability,
            )
        )
        state.running_generation = None
