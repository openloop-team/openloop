"""Versioned, secret-free domain facts for OpenHands cold resume."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from openloop.tools.openhands_artifacts import WorkspaceArtifact
from openloop.tools.openhands_state import validate_state_identifier


OPENHANDS_RESUME_SCHEMA_VERSION = 1
OPENHANDS_RESUME_READER_VERSION = 1
OPENHANDS_REJECTION_REASON = "User rejected the pending action in Slack"

OpenHandsResumeStatus = Literal[
    "running", "parking", "parked", "resuming", "terminal", "cleaned"
]
DecisionKind = Literal["accept", "reject"]

_STATUSES = frozenset(
    {"running", "parking", "parked", "resuming", "terminal", "cleaned"}
)
_TRANSITIONS = {
    "running": frozenset({"parking", "terminal"}),
    "parking": frozenset({"parked", "terminal"}),
    "parked": frozenset({"resuming", "terminal"}),
    "resuming": frozenset({"parking", "terminal"}),
    "terminal": frozenset({"cleaned"}),
    "cleaned": frozenset(),
}
_DIGEST_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class OpenHandsResumeError(ValueError):
    """Persisted OpenHands lifecycle state is malformed or incompatible."""


def _text(value: str, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\0" in value:
        raise OpenHandsResumeError(f"invalid OpenHands resume {field}")
    return value


def _git_object(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise OpenHandsResumeError(f"invalid OpenHands resume {field}")
    return value


def _counter(value: int | float, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise OpenHandsResumeError(f"invalid OpenHands resume {field}")
    return value


@dataclass(frozen=True, slots=True)
class WorkspaceArtifactRef:
    """A verified descriptor plus reconstruction metadata."""

    artifact: WorkspaceArtifact
    format: str
    base_commit: str

    def __post_init__(self) -> None:
        if self.format != "git-delta":
            raise OpenHandsResumeError("unsupported OpenHands artifact format")
        _git_object(self.base_commit, field="artifact base commit")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.artifact.to_dict(),
            "format": self.format,
            "base_commit": self.base_commit,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "WorkspaceArtifactRef":
        if not isinstance(raw, dict):
            raise OpenHandsResumeError("invalid OpenHands workspace artifact")
        return cls(
            artifact=WorkspaceArtifact.from_dict(raw),
            format=raw.get("format"),
            base_commit=raw.get("base_commit"),
        )


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    kind: DecisionKind
    decision_id: str
    event_id: str
    actor_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"accept", "reject"}:
            raise OpenHandsResumeError("unsupported OpenHands resume decision")
        validate_state_identifier(self.decision_id, field="decision_id")
        validate_state_identifier(self.event_id, field="event_id")
        validate_state_identifier(self.actor_id, field="actor_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "decision_id": self.decision_id,
            "event_id": self.event_id,
            "actor_id": self.actor_id,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ResumeDecision":
        if not isinstance(raw, dict):
            raise OpenHandsResumeError("invalid OpenHands resume decision")
        return cls(
            kind=raw.get("kind"),
            decision_id=raw.get("decision_id"),
            event_id=raw.get("event_id"),
            actor_id=raw.get("actor_id"),
        )


@dataclass(frozen=True, slots=True)
class WorkerPaused:
    """One OpenHands segment reached a durable confirmation boundary."""

    conversation_id: str
    segment_id: str
    decision_id: str
    pending_action_summary: str
    pending_action_fingerprint: str
    workspace_artifact: WorkspaceArtifactRef
    cumulative_cost: float = 0.0
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0

    def __post_init__(self) -> None:
        validate_state_identifier(self.conversation_id, field="conversation_id")
        validate_state_identifier(self.segment_id, field="segment_id")
        validate_state_identifier(self.decision_id, field="decision_id")
        _text(self.pending_action_summary, field="pending action summary", maximum=2000)
        if not _SHA256.fullmatch(self.pending_action_fingerprint):
            raise OpenHandsResumeError("invalid pending action fingerprint")
        _counter(self.cumulative_cost, field="cumulative cost")
        _counter(self.cumulative_prompt_tokens, field="cumulative prompt tokens")
        _counter(
            self.cumulative_completion_tokens,
            field="cumulative completion tokens",
        )


@dataclass(slots=True)
class OpenHandsResumeState:
    """The complete durable join for one resumable OpenHands job."""

    status: OpenHandsResumeStatus
    conversation_id: str
    segment_id: str
    base_ref: str
    resolved_base_commit: str
    image_digest: str
    master_key_id: str
    schema_version: int = OPENHANDS_RESUME_SCHEMA_VERSION
    minimum_reader_version: int = OPENHANDS_RESUME_READER_VERSION
    decision_id: str | None = None
    slack_requester_id: str | None = None
    pending_action_summary: str | None = None
    pending_action_fingerprint: str | None = None
    workspace_artifact: WorkspaceArtifactRef | None = None
    cumulative_cost: float = 0.0
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    last_settled_cumulative_cost: float = 0.0
    last_settled_cumulative_prompt_tokens: int = 0
    last_settled_cumulative_completion_tokens: int = 0
    resolved_event_id: str | None = None
    resolved_decision: ResumeDecision | None = None

    def __post_init__(self) -> None:
        if self.schema_version != OPENHANDS_RESUME_SCHEMA_VERSION:
            raise OpenHandsResumeError(
                f"unsupported OpenHands resume schema version {self.schema_version}"
            )
        if self.minimum_reader_version > OPENHANDS_RESUME_READER_VERSION:
            raise OpenHandsResumeError(
                "OpenHands resume state requires a newer reader"
            )
        if self.minimum_reader_version < 1:
            raise OpenHandsResumeError("invalid OpenHands minimum reader version")
        if self.status not in _STATUSES:
            raise OpenHandsResumeError(
                f"unsupported OpenHands resume status {self.status!r}"
            )
        validate_state_identifier(self.conversation_id, field="conversation_id")
        validate_state_identifier(self.segment_id, field="segment_id")
        _text(self.base_ref, field="base ref", maximum=512)
        if self.base_ref.startswith("-"):
            raise OpenHandsResumeError("invalid OpenHands resume base ref")
        _git_object(self.resolved_base_commit, field="resolved base commit")
        if not _DIGEST_IMAGE.fullmatch(self.image_digest):
            raise OpenHandsResumeError("invalid OpenHands resume image digest")
        validate_state_identifier(self.master_key_id, field="master_key_id")

        if self.decision_id is not None:
            validate_state_identifier(self.decision_id, field="decision_id")
        if self.slack_requester_id is not None:
            validate_state_identifier(
                self.slack_requester_id, field="slack_requester_id"
            )
        if self.pending_action_summary is not None:
            _text(
                self.pending_action_summary,
                field="pending action summary",
                maximum=2000,
            )
        if self.pending_action_fingerprint is not None and not _SHA256.fullmatch(
            self.pending_action_fingerprint
        ):
            raise OpenHandsResumeError("invalid pending action fingerprint")
        if self.resolved_event_id is not None:
            validate_state_identifier(self.resolved_event_id, field="resolved_event_id")

        for field in (
            "cumulative_cost",
            "cumulative_prompt_tokens",
            "cumulative_completion_tokens",
            "last_settled_cumulative_cost",
            "last_settled_cumulative_prompt_tokens",
            "last_settled_cumulative_completion_tokens",
        ):
            _counter(getattr(self, field), field=field.replace("_", " "))
        if self.last_settled_cumulative_cost > self.cumulative_cost:
            raise OpenHandsResumeError("settled cost exceeds cumulative cost")
        if self.last_settled_cumulative_prompt_tokens > self.cumulative_prompt_tokens:
            raise OpenHandsResumeError("settled prompt tokens exceed cumulative tokens")
        if (
            self.last_settled_cumulative_completion_tokens
            > self.cumulative_completion_tokens
        ):
            raise OpenHandsResumeError(
                "settled completion tokens exceed cumulative tokens"
            )

        if self.status in {"parked", "resuming"}:
            if not all(
                (
                    self.decision_id,
                    self.pending_action_summary,
                    self.pending_action_fingerprint,
                    self.workspace_artifact,
                )
            ):
                raise OpenHandsResumeError(
                    f"OpenHands {self.status} state is missing decision data"
                )
            if self.workspace_artifact.artifact.identity.kind != "paused":
                raise OpenHandsResumeError("parked state requires a paused artifact")
        if self.workspace_artifact is not None:
            identity = self.workspace_artifact.artifact.identity
            if identity.conversation_id != self.conversation_id:
                raise OpenHandsResumeError("artifact conversation mismatch")
            # A parked artifact belongs to the segment that produced it. Once
            # a decision is durably accepted, ``resuming`` preallocates the
            # *next* segment ID before external work while retaining that
            # previous artifact as its reconstruction input.
            if self.status != "resuming" and identity.segment_id != self.segment_id:
                raise OpenHandsResumeError("artifact segment mismatch")
            if self.workspace_artifact.base_commit != self.resolved_base_commit:
                raise OpenHandsResumeError("artifact base commit mismatch")
            if self.workspace_artifact.artifact.master_key_id != self.master_key_id:
                raise OpenHandsResumeError("artifact master-key mismatch")
        if self.status == "resuming":
            if self.resolved_event_id is None or self.resolved_decision is None:
                raise OpenHandsResumeError(
                    "resuming state requires a structured resolving decision"
                )
            if self.resolved_decision.event_id != self.resolved_event_id:
                raise OpenHandsResumeError("resolving decision event mismatch")
            if self.resolved_decision.decision_id != self.decision_id:
                raise OpenHandsResumeError("resolving decision ID mismatch")

    def transition_to(self, status: OpenHandsResumeStatus, **changes) -> None:
        if status not in _TRANSITIONS[self.status]:
            raise OpenHandsResumeError(
                f"illegal OpenHands resume transition {self.status!r} -> {status!r}"
            )
        candidate = replace(self, status=status, **changes)
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(candidate, name))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "minimum_reader_version": self.minimum_reader_version,
            "status": self.status,
            "conversation_id": self.conversation_id,
            "segment_id": self.segment_id,
            "base_ref": self.base_ref,
            "resolved_base_commit": self.resolved_base_commit,
            "decision_id": self.decision_id,
            "slack_requester_id": self.slack_requester_id,
            "pending_action_summary": self.pending_action_summary,
            "pending_action_fingerprint": self.pending_action_fingerprint,
            "workspace_artifact": (
                self.workspace_artifact.to_dict()
                if self.workspace_artifact is not None
                else None
            ),
            "cumulative_cost": self.cumulative_cost,
            "cumulative_prompt_tokens": self.cumulative_prompt_tokens,
            "cumulative_completion_tokens": self.cumulative_completion_tokens,
            "last_settled_cumulative_cost": self.last_settled_cumulative_cost,
            "last_settled_cumulative_prompt_tokens": (
                self.last_settled_cumulative_prompt_tokens
            ),
            "last_settled_cumulative_completion_tokens": (
                self.last_settled_cumulative_completion_tokens
            ),
            "image_digest": self.image_digest,
            "master_key_id": self.master_key_id,
            "resolved_event_id": self.resolved_event_id,
            "resolved_decision": (
                self.resolved_decision.to_dict()
                if self.resolved_decision is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "OpenHandsResumeState":
        if not isinstance(raw, dict):
            raise OpenHandsResumeError("invalid OpenHands resume state")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise OpenHandsResumeError(
                f"unknown OpenHands resume fields: {sorted(unknown)!r}"
            )
        artifact = raw.get("workspace_artifact")
        values = dict(raw)
        values["workspace_artifact"] = (
            WorkspaceArtifactRef.from_dict(artifact) if artifact is not None else None
        )
        decision = raw.get("resolved_decision")
        values["resolved_decision"] = (
            ResumeDecision.from_dict(decision) if decision is not None else None
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise OpenHandsResumeError("incomplete OpenHands resume state") from exc
