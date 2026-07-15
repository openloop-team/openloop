"""Typed, versioned OpenHands cold-resume contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from openloop.tools.coding_worker import WorkerState
from openloop.tools.openhands_artifacts import (
    WorkspaceArtifact,
    WorkspaceArtifactIdentity,
)
from openloop.tools.openhands_resume import (
    OPENHANDS_RESUME_READER_VERSION,
    OpenHandsResumeError,
    OpenHandsResumeState,
    ResumeDecision,
    WorkerPaused,
    WorkspaceArtifactRef,
)


IMAGE = (
    "ghcr.io/openhands/agent-server@"
    "sha256:08d3994f9287f8d52b07907ac1575ecfaa48b972697ddae4f1cb5c2f03713fab"
)
BASE = "a" * 40
FINGERPRINT = "b" * 64


def _artifact(kind="paused", segment="segment-1") -> WorkspaceArtifactRef:
    identity = WorkspaceArtifactIdentity(
        "job-1", "conversation-1", segment, kind
    )
    return WorkspaceArtifactRef(
        artifact=WorkspaceArtifact(
            identity=identity,
            key=f"jobs/job-1/artifacts/conversation-1/{segment}.{kind}.artifact",
            ciphertext_sha256="c" * 64,
            ciphertext_bytes=123,
            envelope_version=1,
            master_key_id="key-v1",
        ),
        format="git-delta",
        base_commit=BASE,
    )


def _parked() -> OpenHandsResumeState:
    return OpenHandsResumeState(
        status="parked",
        conversation_id="conversation-1",
        segment_id="segment-1",
        base_ref="refs/heads/main",
        resolved_base_commit=BASE,
        decision_id="decision-1",
        slack_requester_id="U123",
        pending_action_summary="Run the repository tests",
        pending_action_fingerprint=FINGERPRINT,
        workspace_artifact=_artifact(),
        cumulative_cost=0.5,
        cumulative_prompt_tokens=100,
        cumulative_completion_tokens=25,
        last_settled_cumulative_cost=0.5,
        last_settled_cumulative_prompt_tokens=100,
        last_settled_cumulative_completion_tokens=25,
        image_digest=IMAGE,
        master_key_id="key-v1",
    )


def test_resume_state_round_trips_without_secrets():
    state = _parked()
    raw = state.to_dict()
    restored = OpenHandsResumeState.from_dict(raw)

    assert restored.to_dict() == raw
    rendered = repr(raw).lower()
    assert "api_key" not in rendered
    assert "secret_key" not in rendered
    assert "provider_key" not in rendered


def test_worker_state_round_trips_versioned_and_legacy_checkpoints():
    state = WorkerState(
        job_id="job-1",
        repo="acme/repo",
        instruction="do work",
        base="main",
        branch="openloop/job-job-1",
        openhands_resume=_parked(),
    )
    restored = WorkerState.from_dict(state.to_dict())
    assert restored.openhands_resume is not None
    assert restored.openhands_resume.status == "parked"

    legacy = state.to_dict()
    legacy.pop("openhands_resume")
    assert WorkerState.from_dict(legacy).openhands_resume is None


def test_legal_transition_requires_resolving_event_and_preserves_schema():
    state = _parked()
    decision = ResumeDecision(
        kind="accept",
        decision_id="decision-1",
        event_id="Ev123",
        actor_id="U123",
    )
    state.transition_to(
        "resuming",
        segment_id="segment-2",
        resolved_event_id="Ev123",
        resolved_decision=decision,
    )
    assert state.status == "resuming"
    assert state.segment_id == "segment-2"
    assert state.resolved_event_id == "Ev123"
    assert state.resolved_decision == decision
    assert state.schema_version == 1

    with pytest.raises(OpenHandsResumeError, match="illegal"):
        state.transition_to("cleaned")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"schema_version": 999}, "schema"),
        (
            {"minimum_reader_version": OPENHANDS_RESUME_READER_VERSION + 1},
            "newer reader",
        ),
        ({"status": "mystery"}, "status"),
        ({"conversation_id": "../escape"}, "conversation_id"),
        ({"pending_action_fingerprint": "short"}, "fingerprint"),
        ({"image_digest": "latest"}, "image"),
    ],
)
def test_incompatible_or_malformed_state_fails_closed(mutation, match):
    raw = {**_parked().to_dict(), **mutation}
    with pytest.raises((OpenHandsResumeError, ValueError), match=match):
        OpenHandsResumeState.from_dict(raw)


def test_unknown_fields_are_rejected():
    raw = {**_parked().to_dict(), "future_side_effect": True}
    with pytest.raises(OpenHandsResumeError, match="unknown"):
        OpenHandsResumeState.from_dict(raw)


def test_artifact_identity_base_and_key_must_match_state():
    with pytest.raises(OpenHandsResumeError, match="segment"):
        replace(_parked(), segment_id="segment-2")
    with pytest.raises(OpenHandsResumeError, match="master-key"):
        replace(_parked(), master_key_id="key-v2")
    with pytest.raises(OpenHandsResumeError, match="base commit"):
        replace(_parked(), resolved_base_commit="d" * 40)


def test_decision_and_paused_contracts_are_typed():
    decision = ResumeDecision(
        kind="reject",
        decision_id="decision-1",
        event_id="Ev123",
        actor_id="U123",
    )
    assert ResumeDecision.from_dict(decision.to_dict()) == decision

    paused = WorkerPaused(
        conversation_id="conversation-1",
        segment_id="segment-1",
        decision_id="decision-1",
        pending_action_summary="Run the tests",
        pending_action_fingerprint=FINGERPRINT,
        workspace_artifact=_artifact(),
        cumulative_cost=0.5,
        cumulative_prompt_tokens=100,
        cumulative_completion_tokens=25,
    )
    assert paused.workspace_artifact.artifact.identity.kind == "paused"


def test_parked_state_requires_complete_decision_data():
    with pytest.raises(OpenHandsResumeError, match="missing decision"):
        replace(_parked(), decision_id=None)
