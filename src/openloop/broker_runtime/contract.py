"""Bounded, runtime-neutral values for privileged generation drivers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable
from uuid import UUID

from openloop.broker.models import POSTGRES_BIGINT_MAX
from openloop.tools.openhands_relay import RelayClientEndpoint


_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}\Z")


def _uuid(name: str, value: object) -> UUID:
    if not isinstance(value, UUID):
        raise TypeError(f"{name} must be a UUID")
    return value


def _generation(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= POSTGRES_BIGINT_MAX
    ):
        raise ValueError("generation must be a positive PostgreSQL BIGINT")
    return value


def _deadline(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("deadline must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("deadline must be timezone-aware UTC")
    if value.microsecond:
        raise ValueError("deadline must have whole-second precision")
    return value


def _token(name: str, value: object) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 32-256 character base64url token")
    return value


class RuntimeDriverError(RuntimeError):
    """A generation runtime operation failed closed."""


class RuntimeUnavailable(RuntimeDriverError):
    """The configured runtime cannot safely realize its fixed profile."""


class RuntimeIdentityConflict(RuntimeDriverError):
    """A discovered resource does not have the complete expected identity."""


class RuntimeExpired(RuntimeDriverError):
    """The original absolute generation deadline has elapsed."""


class RuntimeHealthFailure(RuntimeDriverError):
    """The generation failed its fixed relay health gate."""


class RuntimeResourceState(str, Enum):
    ABSENT = "absent"
    CREATED = "created"
    RUNNING = "running"
    EXITED = "exited"


@dataclass(frozen=True, slots=True)
class GenerationRuntimeIdentity:
    operation_id: UUID
    job_id: UUID
    generation: int
    deadline: datetime

    def __post_init__(self) -> None:
        _uuid("operation_id", self.operation_id)
        _uuid("job_id", self.job_id)
        _generation(self.generation)
        _deadline(self.deadline)

    @property
    def deadline_epoch(self) -> int:
        return int(self.deadline.timestamp())

    @property
    def opaque_handle(self) -> str:
        return (
            "docker-openhands:v1:"
            f"{self.operation_id}:{self.job_id}:{self.generation}:"
            f"{self.deadline_epoch}"
        )


@dataclass(frozen=True, slots=True, repr=False)
class OpenHandsGenerationSpec:
    operation_id: UUID
    job_id: UUID
    conversation_id: UUID
    generation: int
    deadline: datetime
    relay_capability: str = field(repr=False)
    session_api_key: str = field(repr=False)
    conversation_secret: str = field(repr=False)

    def __post_init__(self) -> None:
        _uuid("operation_id", self.operation_id)
        _uuid("job_id", self.job_id)
        _uuid("conversation_id", self.conversation_id)
        _generation(self.generation)
        _deadline(self.deadline)
        _token("relay_capability", self.relay_capability)
        _token("session_api_key", self.session_api_key)
        _token("conversation_secret", self.conversation_secret)

    @property
    def identity(self) -> GenerationRuntimeIdentity:
        return GenerationRuntimeIdentity(
            operation_id=self.operation_id,
            job_id=self.job_id,
            generation=self.generation,
            deadline=self.deadline,
        )

    def __repr__(self) -> str:
        return (
            "OpenHandsGenerationSpec("
            f"operation_id={str(self.operation_id)!r}, "
            f"job_id={str(self.job_id)!r}, "
            f"conversation_id={str(self.conversation_id)!r}, "
            f"generation={self.generation}, deadline={self.deadline!r}, "
            "relay_capability=<redacted>, session_api_key=<redacted>, "
            "conversation_secret=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class GenerationObservation:
    identity: GenerationRuntimeIdentity
    network: RuntimeResourceState
    agent: RuntimeResourceState
    relay: RuntimeResourceState
    artifacts_ready: bool
    workspace_ready: bool
    healthy: bool
    expired: bool

    def __post_init__(self) -> None:
        if not isinstance(self.identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        for name in ("network", "agent", "relay"):
            if not isinstance(getattr(self, name), RuntimeResourceState):
                raise TypeError(f"{name} must be a RuntimeResourceState")
        for name in ("artifacts_ready", "workspace_ready", "healthy", "expired"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")

    @property
    def complete(self) -> bool:
        return (
            self.network is RuntimeResourceState.CREATED
            and self.agent is RuntimeResourceState.RUNNING
            and self.relay is RuntimeResourceState.RUNNING
            and self.artifacts_ready
            and self.workspace_ready
            and self.healthy
            and not self.expired
        )


@dataclass(frozen=True, slots=True, repr=False)
class EnsuredGeneration:
    handle: str
    endpoint: RelayClientEndpoint = field(repr=False)
    observation: GenerationObservation

    def __post_init__(self) -> None:
        if self.handle != self.observation.identity.opaque_handle:
            raise ValueError("runtime handle does not match generation identity")
        if not isinstance(self.endpoint, RelayClientEndpoint):
            raise TypeError("endpoint must be a RelayClientEndpoint")
        if not self.observation.complete:
            raise ValueError("ensured generation must be complete and healthy")

    def __repr__(self) -> str:
        return (
            f"EnsuredGeneration(handle={self.handle!r}, endpoint=<redacted>, "
            f"observation={self.observation!r})"
        )


@dataclass(frozen=True, slots=True)
class ReleaseObservation:
    identity: GenerationRuntimeIdentity
    released: bool
    durable_state_preserved: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        if type(self.released) is not bool:
            raise TypeError("released must be a bool")
        if type(self.durable_state_preserved) is not bool:
            raise TypeError("durable_state_preserved must be a bool")


@runtime_checkable
class RuntimeDriver(Protocol):
    async def ensure(self, spec: OpenHandsGenerationSpec) -> EnsuredGeneration: ...

    async def inspect(
        self, identity: GenerationRuntimeIdentity
    ) -> GenerationObservation: ...

    async def release(
        self, identity: GenerationRuntimeIdentity
    ) -> ReleaseObservation: ...


__all__ = [
    "EnsuredGeneration",
    "GenerationObservation",
    "GenerationRuntimeIdentity",
    "OpenHandsGenerationSpec",
    "ReleaseObservation",
    "RuntimeDriver",
    "RuntimeDriverError",
    "RuntimeExpired",
    "RuntimeHealthFailure",
    "RuntimeIdentityConflict",
    "RuntimeResourceState",
    "RuntimeUnavailable",
]
