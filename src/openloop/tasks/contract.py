"""The neutral workspace-task contract: gate, profile, and durable task state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Gate(str, Enum):
    """Where a profile's approval boundary sits. A profile's value is a floor;
    agent config may add gating, never remove it."""

    START = "start"
    EFFECT = "effect"  # reserved for mid-task proposed effects (Gate F); unused in Stage 1
    NONE = "none"


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """A profile declaration: its entry action, typed args, gate, and grants."""

    name: str
    entry_action: str
    args_model: type
    gate: Gate
    capabilities: frozenset[str]


@dataclass(slots=True)
class WorkspaceTask:
    """Durable task identity + profile-neutral shared state.

    Holds nothing that presumes an outcome. Outcome-specific fields (repo,
    branch, PR title/body) live in ``profile_state``. ``task_id`` is the
    workflow instance id — it identifies the task, not the profile."""

    task_id: str
    profile: str
    entry_action: str
    agent: str | None = None
    agent_id: str | None = None
    approval_id: str | None = None
    requester_id: str | None = None
    session_id: str | None = None
    warm_key: str | None = None
    progress: str | None = None
    profile_state: dict = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceTask":
        return cls(**{k: v for k, v in data.items() if k in _WT_FIELDS})


_WT_FIELDS = frozenset(WorkspaceTask.__dataclass_fields__)


WORKSPACE_TASK_PROFILES: dict[str, TaskProfile] = {
    "code:write": TaskProfile(
        name="code", entry_action="code:write", args_model=object,
        gate=Gate.START, capabilities=frozenset({"repo:write"}),
    ),
    "investigate:read": TaskProfile(
        name="investigate", entry_action="investigate:read", args_model=object,
        gate=Gate.NONE, capabilities=frozenset({"repo:read"}),
    ),
}


def profile_for(permission: str) -> "TaskProfile | None":
    """Look up a profile by permission string."""
    return WORKSPACE_TASK_PROFILES.get(permission)
