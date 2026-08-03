"""Outcome-neutral workspace-task contract (roadmap Stage 1)."""

from openloop.tasks.binding import (
    BUSY,
    CLOSED,
    OPEN,
    InMemoryThreadTaskStore,
    ThreadTask,
    ThreadTaskStore,
)
from openloop.tasks.continuation import (
    CONTINUATION,
    ContinuationUnavailable,
    continuable,
    continuation_instance_id,
    continuation_state,
    may_continue,
)
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
    "BUSY",
    "CLOSED",
    "OPEN",
    "InMemoryThreadTaskStore",
    "ThreadTask",
    "ThreadTaskStore",
    "CONTINUATION",
    "ContinuationUnavailable",
    "continuable",
    "continuation_instance_id",
    "continuation_state",
    "may_continue",
]
