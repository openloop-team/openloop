"""Deterministic runtime-driver test double with no privileged side effects."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from openloop.tools.openhands_relay import (
    RelayClientEndpoint,
    RelayMode,
    compile_openhands_relay,
)

from .contract import (
    EnsuredGeneration,
    GenerationObservation,
    GenerationRuntimeIdentity,
    OpenHandsGenerationSpec,
    QuiescedGeneration,
    ReleaseObservation,
    RuntimeDriver,
    RuntimeExpired,
    RuntimeIdentityConflict,
    RuntimeResourceState,
    RuntimeUnavailable,
)


class InMemoryRuntimeDriver(RuntimeDriver):
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        maximum_lifetime_seconds: int = 86_400,
    ) -> None:
        if (
            isinstance(maximum_lifetime_seconds, bool)
            or not isinstance(maximum_lifetime_seconds, int)
            or not 1 <= maximum_lifetime_seconds <= 86_400
        ):
            raise ValueError("maximum_lifetime_seconds must be in 1-86400")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._maximum_lifetime_seconds = maximum_lifetime_seconds
        self._specs: dict[GenerationRuntimeIdentity, OpenHandsGenerationSpec] = {}
        self._modes: dict[GenerationRuntimeIdentity, RelayMode] = {}

    @property
    def maximum_lifetime_seconds(self) -> int:
        return self._maximum_lifetime_seconds

    @staticmethod
    def _endpoint(
        spec: OpenHandsGenerationSpec, mode: RelayMode
    ) -> RelayClientEndpoint:
        if not isinstance(spec, OpenHandsGenerationSpec):
            raise TypeError("spec must be an OpenHandsGenerationSpec")
        return compile_openhands_relay(
            job_id=spec.job_id,
            generation=spec.generation,
            conversation_id=spec.conversation_id,
            relay_capability=spec.relay_capability,
            session_api_key=spec.session_api_key,
            mode=mode,
        ).endpoint

    def describe_endpoint(
        self, spec: OpenHandsGenerationSpec
    ) -> RelayClientEndpoint:
        return self._endpoint(spec, RelayMode.RUNNING)

    def _expired(self, identity: GenerationRuntimeIdentity) -> bool:
        return self._clock() >= identity.deadline

    async def ensure(self, spec: OpenHandsGenerationSpec) -> EnsuredGeneration:
        if not isinstance(spec, OpenHandsGenerationSpec):
            raise TypeError("spec must be an OpenHandsGenerationSpec")
        identity = spec.identity
        if self._expired(identity):
            raise RuntimeExpired("generation execution deadline has elapsed")
        if (
            identity.deadline - self._clock()
        ).total_seconds() > self._maximum_lifetime_seconds:
            raise RuntimeUnavailable("generation deadline exceeds runtime maximum")
        existing = self._specs.get(identity)
        if existing is not None and existing != spec:
            raise RuntimeIdentityConflict(
                "generation identity was reused with different runtime inputs"
            )
        if self._modes.get(identity) is RelayMode.CHECKPOINT:
            raise RuntimeIdentityConflict(
                "quiesced generation cannot return to running mode"
            )
        self._specs[identity] = spec
        self._modes[identity] = RelayMode.RUNNING
        observation = await self.inspect(identity)
        return EnsuredGeneration(
            handle=identity.opaque_handle,
            endpoint=self.describe_endpoint(spec),
            observation=observation,
        )

    async def quiesce(
        self, spec: OpenHandsGenerationSpec
    ) -> QuiescedGeneration:
        if not isinstance(spec, OpenHandsGenerationSpec):
            raise TypeError("spec must be an OpenHandsGenerationSpec")
        identity = spec.identity
        if self._expired(identity):
            raise RuntimeExpired("generation execution deadline has elapsed")
        existing = self._specs.get(identity)
        if existing is None:
            raise RuntimeUnavailable("generation is not present")
        if existing != spec:
            raise RuntimeIdentityConflict(
                "generation identity was reused with different runtime inputs"
            )
        self._modes[identity] = RelayMode.CHECKPOINT
        observation = await self.inspect(identity)
        return QuiescedGeneration(
            handle=identity.opaque_handle,
            endpoint=self._endpoint(spec, RelayMode.CHECKPOINT),
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
        self._modes.pop(identity, None)
        return ReleaseObservation(identity=identity, released=True)


__all__ = ["InMemoryRuntimeDriver"]
