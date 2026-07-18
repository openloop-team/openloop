"""Deterministic runtime-driver test double with no privileged side effects."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from openloop.tools.openhands_relay import RelayMode, compile_openhands_relay

from .contract import (
    EnsuredGeneration,
    GenerationObservation,
    GenerationRuntimeIdentity,
    OpenHandsGenerationSpec,
    ReleaseObservation,
    RuntimeDriver,
    RuntimeExpired,
    RuntimeIdentityConflict,
    RuntimeResourceState,
)


class InMemoryRuntimeDriver(RuntimeDriver):
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._specs: dict[GenerationRuntimeIdentity, OpenHandsGenerationSpec] = {}

    def _expired(self, identity: GenerationRuntimeIdentity) -> bool:
        return self._clock() >= identity.deadline

    async def ensure(self, spec: OpenHandsGenerationSpec) -> EnsuredGeneration:
        if not isinstance(spec, OpenHandsGenerationSpec):
            raise TypeError("spec must be an OpenHandsGenerationSpec")
        identity = spec.identity
        if self._expired(identity):
            raise RuntimeExpired("generation execution deadline has elapsed")
        existing = self._specs.get(identity)
        if existing is not None and existing != spec:
            raise RuntimeIdentityConflict(
                "generation identity was reused with different runtime inputs"
            )
        self._specs[identity] = spec
        compiled = compile_openhands_relay(
            job_id=spec.job_id,
            generation=spec.generation,
            conversation_id=spec.conversation_id,
            relay_capability=spec.relay_capability,
            session_api_key=spec.session_api_key,
            mode=RelayMode.RUNNING,
        )
        observation = await self.inspect(identity)
        return EnsuredGeneration(
            handle=identity.opaque_handle,
            endpoint=compiled.endpoint,
            observation=observation,
        )

    async def inspect(
        self, identity: GenerationRuntimeIdentity
    ) -> GenerationObservation:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        present = identity in self._specs
        state = (
            RuntimeResourceState.RUNNING
            if present
            else RuntimeResourceState.ABSENT
        )
        return GenerationObservation(
            identity=identity,
            network=(
                RuntimeResourceState.CREATED
                if present
                else RuntimeResourceState.ABSENT
            ),
            agent=state,
            relay=state,
            artifacts_ready=present,
            workspace_ready=present,
            healthy=present and not self._expired(identity),
            expired=self._expired(identity),
        )

    async def release(
        self, identity: GenerationRuntimeIdentity
    ) -> ReleaseObservation:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        self._specs.pop(identity, None)
        return ReleaseObservation(identity=identity, released=True)


__all__ = ["InMemoryRuntimeDriver"]
