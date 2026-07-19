"""Pending approval requests and where they live.

Decide-once: the approval row is the single atomic arbiter of a decision.
``claim_decision`` flips a pending row exactly once; every effect (workflow
wake, cancel, direct execute) follows the durable decision, and
``effect_at`` / ``decided_unreconciled`` / ``mark_reconciled`` are the
bookkeeping that lets a recovery sweep heal a crash between claim and effect.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class ApprovalRequest:
    """A write action held pending a human decision.

    Carries everything needed to execute the action once approved, so the
    gateway can run it without the original caller's context.
    """

    agent: str
    action: str  # e.g. "github.issues:write"
    tool: str
    permission: str
    args: dict
    approvers: list[str]
    summary: str
    requested_by: str | None = None
    status: str = "pending"  # pending | approved | denied
    decided_by: str | None = None
    # The per-action args-contract version the args were parsed under
    # (ActionSpec.version), stamped at creation. None is the pre-version
    # sentinel: a version-checking consumer must refuse it, so a record written
    # before versioning existed can never be mislabeled current.
    args_schema: int | None = None
    # Tri-state execution marker stamped at creation: True = the gateway
    # committed this request to a durable workflow, False = direct execute,
    # None = legacy row predating the marker (classified by the registry at
    # resolve time). Decided paths route on this marker, never on the
    # resolver's current engine/tool shape.
    workflow_backed: bool | None = None
    # The workflow instance id, stamped iff workflow_backed is True — the only
    # id a denial may cancel (a direct request's model-supplied job_id could
    # collide with an unrelated live workflow).
    workflow_instance_id: str | None = None
    # Durable marker that the decision's effect (wake, cancel, or direct
    # execute) is known to have been performed; NULL keeps the row in the
    # decision reconciler's sweep.
    effect_at: datetime | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@runtime_checkable
class ApprovalStore(Protocol):
    async def create(self, request: ApprovalRequest) -> None: ...

    async def get(self, request_id: str) -> ApprovalRequest | None: ...

    async def pending(self, agent: str | None = None) -> list[ApprovalRequest]: ...

    async def claim_decision(
        self, request_id: str, approver: str, *, approve: bool
    ) -> ApprovalRequest | None:
        """Atomically decide a pending request; ``None`` = already decided or
        missing (the caller lost the race and must treat the stored decision
        as the truth)."""
        ...

    async def decided_unreconciled(
        self,
        limit: int = 200,
        after: tuple[datetime, str] | None = None,
    ) -> list[ApprovalRequest]:
        """Decided rows whose effect is not yet marked, ``(created_at, id)``
        ascending, with ``after`` as a keyset cursor past already-seen rows."""
        ...

    async def mark_reconciled(self, request_id: str) -> None:
        """Idempotently record that the decision's effect was performed (or
        that no effect is possible for this row anywhere)."""
        ...


class InMemoryApprovalStore:
    """Process-local approvals — good for dev and tests.

    Snapshot-isolated like the in-memory workflow store: every operation
    stores and returns detached copies, so callers can never mutate stored
    state by aliasing — a state change happens only through a store op.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, ApprovalRequest] = {}

    async def create(self, request: ApprovalRequest) -> None:
        self._by_id[request.id] = copy.deepcopy(request)

    async def get(self, request_id: str) -> ApprovalRequest | None:
        stored = self._by_id.get(request_id)
        return copy.deepcopy(stored) if stored is not None else None

    async def pending(self, agent: str | None = None) -> list[ApprovalRequest]:
        return [
            copy.deepcopy(r)
            for r in self._by_id.values()
            if r.status == "pending" and (agent is None or r.agent == agent)
        ]

    async def claim_decision(
        self, request_id: str, approver: str, *, approve: bool
    ) -> ApprovalRequest | None:
        # Check-and-mutate with no await between them — the same
        # single-event-loop atomicity the workflow store relies on.
        stored = self._by_id.get(request_id)
        if stored is None or stored.status != "pending":
            return None
        stored.status = "approved" if approve else "denied"
        stored.decided_by = approver
        return copy.deepcopy(stored)

    async def decided_unreconciled(
        self,
        limit: int = 200,
        after: tuple[datetime, str] | None = None,
    ) -> list[ApprovalRequest]:
        decided = sorted(
            (
                r
                for r in self._by_id.values()
                if r.status != "pending" and r.effect_at is None
            ),
            key=lambda r: (r.created_at, r.id),
        )
        if after is not None:
            decided = [r for r in decided if (r.created_at, r.id) > after]
        return [copy.deepcopy(r) for r in decided[:limit]]

    async def mark_reconciled(self, request_id: str) -> None:
        stored = self._by_id.get(request_id)
        if stored is not None and stored.effect_at is None:
            stored.effect_at = datetime.now(timezone.utc)
