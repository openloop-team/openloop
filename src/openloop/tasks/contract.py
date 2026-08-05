"""The neutral workspace-task contract: gate, profile, and durable task state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Gate(str, Enum):
    """Where a profile's approval boundary sits. A profile's value is a floor;
    agent config may add gating, never remove it."""

    START = "start"
    EFFECT = (
        "effect"  # reserved for mid-task proposed effects (Gate F); unused in Stage 1
    )
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
    # The invoking agent's per-task spend ceiling (a contract-level field, not
    # profile-specific). Stamped for observability by whichever profile wires a
    # WorkerSpendLedger; enforcement always happens in that ledger's settle(),
    # never by reading this field back.
    budget_usd: float | None = None
    profile_state: dict = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Emit the durable nested layout: core fields plus ``profile_state``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> WorkspaceTask:
        """Rehydrate from either the current nested layout or the
        pre-convergence flat coding-worker checkpoint layout (compat shim).

        New layout: keyed by ``task_id``; every top-level key naming a
        WorkspaceTask field (``_WT_FIELDS``) — including ``profile_state``,
        already shaped for this contract — passes straight through.

        Old workflow layout: flat task fields plus a nested ``worker_state``
        dict. Old checkpoint-only layout: ``WorkerState.to_dict()`` itself at
        the top level. Both lack ``task_id``. The shim detects either form,
        lifts identity/attribution/progress to the task core, and carries the
        original worker blob verbatim under
        ``profile_state["code"]["worker_state"]``. Any flat
        ``repo``/``instruction``/``base`` values also move under the code
        profile. Missing keys remain tolerated because legacy rows predate
        several shared fields.

        This shim must stay until no pre-convergence rows remain in the workflow
        and checkpoint stores. Removing it earlier breaks recovery of in-flight
        work: those rows rehydrate as a task with no identity.
        """
        if "task_id" not in data and ("job_id" in data or "worker_state" in data):
            code_state: dict = {}
            worker_state = data.get("worker_state")
            if worker_state is None and "job_id" in data and "branch" in data:
                # Checkpoint-only rows stored WorkerState.to_dict() directly as
                # state_json (without the workflow's outer ``worker_state``
                # key). Carry that blob verbatim too; the code adapter is the
                # only layer allowed to interpret it.
                worker_state = dict(data)
            if worker_state is not None:
                code_state["worker_state"] = worker_state
            for key in ("repo", "instruction", "base"):
                if key in data:
                    code_state[key] = data[key]
                elif isinstance(worker_state, dict) and key in worker_state:
                    code_state[key] = worker_state[key]

            def shared(key: str):
                value = data.get(key)
                if value is None and isinstance(worker_state, dict):
                    value = worker_state.get(key)
                return value

            completed_steps = data.get("completed_steps")
            if completed_steps is None and isinstance(worker_state, dict):
                completed_steps = worker_state.get("completed_steps")
            return cls(
                task_id=shared("job_id"),
                profile="code",
                entry_action="code:write",
                agent=shared("agent"),
                agent_id=shared("agent_id"),
                approval_id=shared("approval_id"),
                requester_id=shared("requester_id"),
                session_id=shared("session_id"),
                warm_key=shared("warm_key"),
                progress=data.get("progress"),
                budget_usd=shared("budget_usd"),
                profile_state={"code": code_state},
                completed_steps=list(completed_steps or []),
            )
        return cls(**{k: v for k, v in data.items() if k in _WT_FIELDS})


_WT_FIELDS = frozenset(WorkspaceTask.__dataclass_fields__)


WORKSPACE_TASK_PROFILES: dict[str, TaskProfile] = {
    "code:write": TaskProfile(
        name="code",
        entry_action="code:write",
        args_model=object,
        gate=Gate.START,
        capabilities=frozenset({"repo:write"}),
    ),
    "investigate:read": TaskProfile(
        name="investigate",
        entry_action="investigate:read",
        args_model=object,
        gate=Gate.NONE,
        capabilities=frozenset({"repo:read"}),
    ),
}


def profile_for(permission: str) -> TaskProfile | None:
    """Look up a profile by permission string."""
    return WORKSPACE_TASK_PROFILES.get(permission)
