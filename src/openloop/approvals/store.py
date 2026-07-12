"""Pending approval requests and where they live."""

from __future__ import annotations

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
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@runtime_checkable
class ApprovalStore(Protocol):
    async def create(self, request: ApprovalRequest) -> None: ...

    async def get(self, request_id: str) -> ApprovalRequest | None: ...

    async def pending(self, agent: str | None = None) -> list[ApprovalRequest]: ...

    async def update(self, request: ApprovalRequest) -> None:
        """Persist a status/decided_by change after a decision."""
        ...


class InMemoryApprovalStore:
    """Process-local approvals — good for dev and tests."""

    def __init__(self) -> None:
        self._by_id: dict[str, ApprovalRequest] = {}

    async def create(self, request: ApprovalRequest) -> None:
        self._by_id[request.id] = request

    async def get(self, request_id: str) -> ApprovalRequest | None:
        return self._by_id.get(request_id)

    async def pending(self, agent: str | None = None) -> list[ApprovalRequest]:
        return [
            r
            for r in self._by_id.values()
            if r.status == "pending" and (agent is None or r.agent == agent)
        ]

    async def update(self, request: ApprovalRequest) -> None:
        self._by_id[request.id] = request
