"""Outcome-neutral workspace-task contract (roadmap Stage 1)."""

from openloop.tasks.contract import Gate, TaskProfile, WorkspaceTask
from openloop.tasks.outcomes import (
    Diagnosis,
    EvidenceBundle,
    Failed,
    PullRequest,
    TaskOutcome,
    to_deliverable,
)

__all__ = [
    "Gate",
    "TaskProfile",
    "WorkspaceTask",
    "Diagnosis",
    "EvidenceBundle",
    "Failed",
    "PullRequest",
    "TaskOutcome",
    "to_deliverable",
]
