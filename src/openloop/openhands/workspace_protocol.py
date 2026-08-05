"""Dependency-light workspace contract consumed by the OpenHands worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ArchiveStreamResult:
    base_commit: str
    base_ref: str
    bytes_written: int


@runtime_checkable
class OpenHandsWorkspace(Protocol):
    """Required broker operations; optional workspace hooks remain additive.

    Every type this module deliberately does not import — receipts, outcomes, the
    opaque workspace, agent, and conversation handles — is ``Any``, not
    ``object``, in both parameter and return position.

    In parameters the two are not interchangeable: parameter types are
    contravariant, so ``object`` would demand implementations accept *every*
    type, which the real adapter (taking SignedCheckpointReceipt) does not and
    should not. In returns ``object`` is satisfiable but useless — it hands the
    caller a value with no attributes, and the worker's whole job is to call
    methods on the conversation ``create`` returns.

    ``Any`` in both directions says "this module does not name the type", which
    is what dependency-light means here.
    """

    def probe(self) -> None: ...

    def create(self, workspace: Path, job_id: str) -> Any: ...

    def attach_conversation(
        self,
        workspace: Any,
        *,
        agent: Any,
        conversation_id: UUID,
        callbacks: list | None = None,
        max_iterations: int = 500,
    ) -> Any: ...

    def stream_git_delta(
        self, workspace: Any, sink: BinaryIO, *, base_ref: str
    ) -> ArchiveStreamResult: ...

    def quiesce(self, job_id: str, barrier_id: str) -> None: ...

    def park(self, job_id: str, receipt: Any) -> None: ...

    def finalize(self, job_id: str, receipt: Any, *, outcome: Any = ...) -> None: ...

    def checkpoint_identity(self, job_id: str, barrier_id: str) -> Any: ...
