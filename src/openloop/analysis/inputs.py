"""Typed argument models for the analysis action (Phase 4 + typed tool args).

The JSON schema the model sees is **generated** from :class:`AnalysisReportArgs`
(``describe()`` returns ``model_json_schema()``), the gateway **parses** raw
args into it before anything durable exists, and consumers **re-parse**
persisted records through :class:`ExecutableAnalysisRequest` — so declaration
and enforcement cannot drift, and an invalid-args durable record is
unrepresentable rather than checked for (docs/typed-tool-args.md §3).

``inputs`` is the Phase 4 manifest contract: a discriminated union over the
provisioning sources, replacing the scalar ``input_ref``. Identity fields
(job_id, attempt_id, agent, the request scope) deliberately do NOT appear here
— they are gateway-stamped after the parse and never model-suppliable.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Bumped on any breaking change to the args contract below. Durable records
# (approvals, parked workflows) are stamped with it at creation; the
# orchestrator refuses to execute a record stamped with any other version —
# including the NULL/absent pre-version sentinel, so a record written before
# versioning existed (the scalar-input_ref era) can never be misread as
# current.
ANALYSIS_ARGS_VERSION = 1

# An unbounded list would let one approved request buy arbitrarily many
# individually-capped fetches; the merged byte cap is the other half.
MAX_INPUTS = 8


def _stripped(value):
    return value.strip() if isinstance(value, str) else value


class StagedInput(BaseModel):
    """An operator-staged input, addressed by its capability token.

    Possession of the ref is the authorization (the same posture as
    ``artifact_ref``): the staging CLI generates it with high entropy and the
    store lookup is job-agnostic.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["staged"]
    input_ref: str = Field(
        min_length=1,
        description="Opaque reference to an operator-staged input set.",
    )

    _strip = field_validator("input_ref", mode="before")(_stripped)


class UploadInput(BaseModel):
    """A file shared in this conversation thread (see the shared-file list)."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["upload"]
    upload_ref: str = Field(
        min_length=1,
        description=(
            "Reference of a file shared in this conversation thread, from "
            "the shared-file list in your context."
        ),
    )

    _strip = field_validator("upload_ref", mode="before")(_stripped)


class GithubInput(BaseModel):
    """A GitHub repository archive, provisioned as one tarball input file."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["github"]
    repo: str = Field(
        min_length=1, description="owner/repo, e.g. acme/ingestion"
    )
    ref: str | None = Field(
        default=None,
        description=(
            "Branch, tag, or commit to archive (default: the repository's "
            "default branch)."
        ),
    )

    _strip_repo = field_validator("repo", mode="before")(_stripped)

    @field_validator("ref", mode="before")
    @classmethod
    def _blank_ref_means_default(cls, value):
        if isinstance(value, str):
            return value.strip() or None
        return value


AnalysisInput = Annotated[
    Union[StagedInput, UploadInput, GithubInput],
    Field(discriminator="source"),
]


class AnalysisReportArgs(BaseModel):
    """The model-facing args contract for ``analysis.report:write``."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(
        min_length=1, description="The analysis question to answer."
    )
    inputs: list[AnalysisInput] = Field(
        min_length=1,
        max_length=MAX_INPUTS,
        description=(
            "Data sources to provision into the sealed run (at most "
            f"{MAX_INPUTS})."
        ),
    )

    _strip = field_validator("instruction", mode="before")(_stripped)


class ExecutableAnalysisRequest(AnalysisReportArgs):
    """The spend-boundary re-parse of a durable record (typed-tool-args §3.5).

    Parsed by the orchestrator AFTER attempt reconciliation (settling
    already-observed spend must work on garbage args) and BEFORE the ledger,
    any provisioning fetch, or a model call — this parse IS the execution
    precondition check. ``extra="ignore"`` because a record also carries
    gateway-stamped identity and display-metadata keys.
    """

    model_config = ConfigDict(extra="ignore")
