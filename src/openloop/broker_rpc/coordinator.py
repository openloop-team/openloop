"""Runtime-neutral port for privileged generation composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable
from uuid import UUID

from openloop.broker.models import (
    BrokerOwner,
    validate_lease_seconds,
    validate_token,
    validate_uuid,
)

from .models import (
    RunningGenerationAccess,
    StartSegmentPayload,
    StartSegmentResult,
)


@dataclass(frozen=True, slots=True)
class BrokerRpcPolicy:
    profile: str
    runtime_driver: str
    durable_state_driver: str
    execution_lease_seconds: int

    def __post_init__(self) -> None:
        validate_token("profile", self.profile)
        validate_token("runtime_driver", self.runtime_driver)
        validate_token("durable_state_driver", self.durable_state_driver)
        validate_lease_seconds(self.execution_lease_seconds)


class SegmentCoordinatorCode(str, Enum):
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    STATE_CONFLICT = "state_conflict"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INTERNAL = "internal"


class SegmentCoordinatorProblem(Exception):
    def __init__(
        self,
        code: SegmentCoordinatorCode,
        *,
        operation_id: UUID | None = None,
    ) -> None:
        if not isinstance(code, SegmentCoordinatorCode):
            raise TypeError("code must be SegmentCoordinatorCode")
        if operation_id is not None:
            validate_uuid("operation_id", operation_id)
        self.code = code
        self.operation_id = operation_id
        super().__init__("segment coordination failed")


@runtime_checkable
class SegmentCoordinator(Protocol):
    async def start_segment(
        self,
        owner: BrokerOwner,
        payload: StartSegmentPayload,
    ) -> StartSegmentResult: ...

    async def inspect_running_access(
        self,
        owner: BrokerOwner,
        job_id: UUID,
    ) -> RunningGenerationAccess | None: ...


__all__ = [
    "BrokerRpcPolicy",
    "SegmentCoordinator",
    "SegmentCoordinatorCode",
    "SegmentCoordinatorProblem",
]
