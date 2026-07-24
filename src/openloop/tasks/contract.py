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
        """Emit the durable nested layout: core fields plus ``profile_state``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceTask":
        """Rehydrate from either the current nested layout or the
        pre-convergence flat coding-worker checkpoint layout (compat shim).

        New layout: keyed by ``task_id``; every top-level key naming a
        WorkspaceTask field (``_WT_FIELDS``) — including ``profile_state``,
        already shaped for this contract — passes straight through.

        Old layout (pre contract-convergence coding-worker checkpoint /
        workflow state): a flat top-level ``{job_id, repo, instruction, base,
        agent, agent_id, approval_id, session_id, warm_key}`` plus a nested
        ``worker_state`` dict (a ``WorkerState.to_dict()``), and no
        ``task_id``. Detected by that absence alongside the presence of
        ``job_id`` and/or ``worker_state``, so an in-flight row written
        before this convergence still loads after it: ``job_id`` becomes
        ``task_id``; the profile defaults to ``"code"`` / ``"code:write"``;
        the identity/attribution fields are lifted to the core; and the
        code-specific bits — the original ``worker_state`` sub-dict
        (carried verbatim; never touched here) plus any flat
        ``repo``/``instruction``/``base`` — move under
        ``profile_state["code"]``. Tolerant of missing keys throughout
        (``.get``) since legacy rows predate several of these fields.
        """
        if "task_id" not in data and ("job_id" in data or "worker_state" in data):
            code_state: dict = {}
            worker_state = data.get("worker_state")
            if worker_state is not None:
                code_state["worker_state"] = worker_state
            for key in ("repo", "instruction", "base"):
                if key in data:
                    code_state[key] = data[key]
            return cls(
                task_id=data.get("job_id"),
                profile="code",
                entry_action="code:write",
                agent=data.get("agent"),
                agent_id=data.get("agent_id"),
                approval_id=data.get("approval_id"),
                requester_id=data.get("requester_id"),
                session_id=data.get("session_id"),
                warm_key=data.get("warm_key"),
                profile_state={"code": code_state},
            )
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
