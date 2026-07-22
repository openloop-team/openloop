"""Dependency-light workspace contract consumed by the OpenHands worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ArchiveStreamResult:
    base_commit: str
    base_ref: str
    bytes_written: int


@runtime_checkable
class OpenHandsWorkspace(Protocol):
    """Required broker operations; optional workspace hooks remain additive."""

    def probe(self) -> None: ...

    def create(self, workspace: Path, job_id: str) -> object: ...

    def attach_conversation(
        self,
        workspace: object,
        *,
        agent: object,
        conversation_id: UUID,
        callbacks: list | None = None,
        max_iterations: int = 500,
    ) -> object: ...

    def stream_git_delta(
        self, workspace: object, sink: BinaryIO, *, base_ref: str
    ) -> ArchiveStreamResult: ...

    def quiesce(self, job_id: str, barrier_id: str) -> None: ...

    def park(self, job_id: str, receipt: object) -> None: ...

    def finalize(self, job_id: str, receipt: object, *, outcome: object = ...) -> None: ...

    def checkpoint_identity(self, job_id: str, barrier_id: str) -> object: ...
