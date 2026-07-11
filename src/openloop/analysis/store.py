"""Stores for provisioned analysis input and sealed-run report artifacts.

Raw input bytes are staged by a trusted caller before an approval is created;
the analysis tool sees only an ``input_ref``.  The orchestrator looks up the
matching job-scoped manifest only after its monthly budget gate passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_filename(name: str) -> None:
    """Reject paths that could escape the materialized inputs directory."""
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name in ("", ".", "..")
    ):
        raise ValueError(f"analysis input filename must be a bare filename: {name!r}")


@dataclass(slots=True, frozen=True)
class InputFile:
    """One controller-provisioned input file.

    Filenames are deliberately restricted to one component in Phase 1. The
    generated program receives the files only through the read-only
    ``/workspace/inputs`` mount.
    """

    name: str
    content: bytes

    def __post_init__(self) -> None:
        _validate_filename(self.name)


@dataclass(slots=True, frozen=True)
class InputManifest:
    """The staged input set for one analysis job and opaque input reference."""

    job_id: str
    input_ref: str
    files: tuple[InputFile, ...]
    created_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("analysis input manifest needs a job_id")
        if not self.input_ref:
            raise ValueError("analysis input manifest needs an input_ref")
        names = [file.name for file in self.files]
        if len(names) != len(set(names)):
            raise ValueError("analysis input manifest has duplicate filenames")

    def materialize(self, destination: Path) -> None:
        """Write the trusted manifest into a newly-created inputs directory."""
        destination.mkdir(parents=True, exist_ok=False)
        for file in self.files:
            # The name is revalidated at the sink so persistence corruption can
            # never turn into a controller path traversal.
            _validate_filename(file.name)
            (destination / file.name).write_bytes(file.content)


@runtime_checkable
class InputStore(Protocol):
    async def stage(self, manifest: InputManifest) -> None: ...

    async def get(self, job_id: str, input_ref: str) -> InputManifest | None: ...


@dataclass(slots=True, frozen=True)
class AnalysisArtifact:
    """The report body retained after a successful, settled sealed run."""

    job_id: str
    artifact_ref: str
    body: bytes
    created_at: datetime = field(default_factory=_now)


@runtime_checkable
class ArtifactStore(Protocol):
    async def put(self, job_id: str, body: bytes) -> str: ...

    async def get(self, artifact_ref: str) -> AnalysisArtifact | None: ...


@dataclass(slots=True)
class AnalysisAttempt:
    """Durable accounting state for one model-authoring attempt.

    ``started`` exists before the model call. ``charged`` means OpenLoop has
    observed a successful completion and durably retained its usage; ``settled``
    means the same charge reached the idempotent usage ledger. A later workflow
    phase can reconcile stalled states without guessing that they were free.
    """

    attempt_id: str
    job_id: str
    status: str = "started"  # started | charged | settled | unknown
    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    charged_at: datetime | None = None
    settled_at: datetime | None = None
    updated_at: datetime = field(default_factory=_now)


@runtime_checkable
class AnalysisAttemptStore(Protocol):
    async def begin(self, attempt_id: str, job_id: str) -> tuple[AnalysisAttempt, bool]: ...

    async def get(self, attempt_id: str) -> AnalysisAttempt | None: ...

    async def charge(
        self,
        attempt_id: str,
        *,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> AnalysisAttempt: ...

    async def settle(self, attempt_id: str) -> AnalysisAttempt: ...

    async def mark_unknown(self, attempt_id: str, error: str) -> AnalysisAttempt: ...


class InMemoryInputStore:
    """Process-local staged inputs for development and tests."""

    def __init__(self) -> None:
        self._by_job: dict[str, InputManifest] = {}

    async def stage(self, manifest: InputManifest) -> None:
        self._by_job[manifest.job_id] = manifest

    async def get(self, job_id: str, input_ref: str) -> InputManifest | None:
        manifest = self._by_job.get(job_id)
        if manifest is None or manifest.input_ref != input_ref:
            return None
        return manifest


class InMemoryArtifactStore:
    """Process-local report artifacts for development and tests."""

    def __init__(self) -> None:
        self._by_ref: dict[str, AnalysisArtifact] = {}

    @staticmethod
    def ref_for(job_id: str) -> str:
        return f"analysis://{job_id}/report.md"

    async def put(self, job_id: str, body: bytes) -> str:
        ref = self.ref_for(job_id)
        existing = self._by_ref.get(ref)
        self._by_ref[ref] = AnalysisArtifact(
            job_id=job_id,
            artifact_ref=ref,
            body=bytes(body),
            created_at=existing.created_at if existing is not None else _now(),
        )
        return ref

    async def get(self, artifact_ref: str) -> AnalysisArtifact | None:
        return self._by_ref.get(artifact_ref)


class InMemoryAnalysisAttemptStore:
    """Process-local attempt accounting for development and tests."""

    def __init__(self) -> None:
        self._by_id: dict[str, AnalysisAttempt] = {}

    async def begin(self, attempt_id: str, job_id: str) -> tuple[AnalysisAttempt, bool]:
        existing = self._by_id.get(attempt_id)
        if existing is not None:
            return existing, False
        attempt = AnalysisAttempt(attempt_id=attempt_id, job_id=job_id)
        self._by_id[attempt_id] = attempt
        return attempt, True

    async def get(self, attempt_id: str) -> AnalysisAttempt | None:
        return self._by_id.get(attempt_id)

    async def charge(
        self,
        attempt_id: str,
        *,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> AnalysisAttempt:
        attempt = self._require(attempt_id)
        if attempt.status in ("charged", "settled"):
            self._assert_same_charge(
                attempt, cost_usd, prompt_tokens, completion_tokens
            )
            return attempt
        if attempt.status != "started":
            raise RuntimeError(
                f"analysis attempt {attempt_id} is {attempt.status}; cannot charge"
            )
        attempt.status = "charged"
        attempt.cost_usd = cost_usd
        attempt.prompt_tokens = prompt_tokens
        attempt.completion_tokens = completion_tokens
        attempt.charged_at = _now()
        attempt.updated_at = _now()
        return attempt

    async def settle(self, attempt_id: str) -> AnalysisAttempt:
        attempt = self._require(attempt_id)
        if attempt.status == "settled":
            return attempt
        if attempt.status != "charged":
            raise RuntimeError(
                f"analysis attempt {attempt_id} is {attempt.status}; cannot settle"
            )
        attempt.status = "settled"
        attempt.settled_at = _now()
        attempt.updated_at = _now()
        return attempt

    async def mark_unknown(self, attempt_id: str, error: str) -> AnalysisAttempt:
        attempt = self._require(attempt_id)
        if attempt.status == "settled":
            return attempt
        attempt.status = "unknown"
        attempt.error = error
        attempt.updated_at = _now()
        return attempt

    def _require(self, attempt_id: str) -> AnalysisAttempt:
        attempt = self._by_id.get(attempt_id)
        if attempt is None:
            raise KeyError(f"unknown analysis attempt {attempt_id}")
        return attempt

    @staticmethod
    def _assert_same_charge(
        attempt: AnalysisAttempt,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        if (
            attempt.cost_usd != cost_usd
            or attempt.prompt_tokens != prompt_tokens
            or attempt.completion_tokens != completion_tokens
        ):
            raise RuntimeError(
                f"analysis attempt {attempt.attempt_id} already has different charge data"
            )
