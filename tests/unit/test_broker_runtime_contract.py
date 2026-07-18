from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from openloop.broker_runtime import (
    GenerationRuntimeIdentity,
    InMemoryRuntimeDriver,
    OpenHandsGenerationSpec,
    RuntimeDriver,
    RuntimeExpired,
    RuntimeIdentityConflict,
    RuntimeResourceState,
)


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")
JOB_ID = UUID("22222222-2222-4222-8222-222222222222")
CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")
RELAY_CAPABILITY = "r" * 43
SESSION_KEY = "s" * 43
CONVERSATION_SECRET = "c" * 43


def _spec(**changes) -> OpenHandsGenerationSpec:
    values = {
        "operation_id": OPERATION_ID,
        "job_id": JOB_ID,
        "conversation_id": CONVERSATION_ID,
        "generation": 1,
        "deadline": NOW + timedelta(minutes=5),
        "relay_capability": RELAY_CAPABILITY,
        "session_api_key": SESSION_KEY,
        "conversation_secret": CONVERSATION_SECRET,
    }
    values.update(changes)
    return OpenHandsGenerationSpec(**values)


def test_spec_redacts_every_runtime_credential():
    rendered = repr(_spec())
    for secret in (RELAY_CAPABILITY, SESSION_KEY, CONVERSATION_SECRET):
        assert secret not in rendered
    assert rendered.count("<redacted>") == 3


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("operation_id", "not-a-uuid", "operation_id"),
        ("job_id", "not-a-uuid", "job_id"),
        ("conversation_id", "not-a-uuid", "conversation_id"),
        ("generation", 0, "generation"),
        ("generation", True, "generation"),
        ("deadline", NOW.replace(tzinfo=None), "timezone-aware UTC"),
        ("deadline", NOW + timedelta(microseconds=1), "whole-second"),
        ("relay_capability", "short", "relay_capability"),
        ("session_api_key", "x" * 42 + "=", "session_api_key"),
        ("conversation_secret", "x\n" * 22, "conversation_secret"),
    ],
)
def test_spec_rejects_malformed_values(field, value, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _spec(**{field: value})


def test_identity_handle_is_deterministic_and_contains_no_runtime_primitive():
    identity = _spec().identity
    assert identity == GenerationRuntimeIdentity(
        OPERATION_ID, JOB_ID, 1, NOW + timedelta(minutes=5)
    )
    assert identity.opaque_handle == (
        "docker-openhands:v1:"
        "11111111-1111-4111-8111-111111111111:"
        "22222222-2222-4222-8222-222222222222:1:1784376300"
    )
    assert "container" not in identity.opaque_handle
    assert "network" not in identity.opaque_handle


async def test_memory_driver_satisfies_contract_and_replays_ensure():
    driver = InMemoryRuntimeDriver(clock=lambda: NOW)
    assert isinstance(driver, RuntimeDriver)

    first = await driver.ensure(_spec())
    second = await driver.ensure(_spec())

    assert first == second
    assert first.observation.complete
    assert first.observation.network is RuntimeResourceState.CREATED
    assert RELAY_CAPABILITY not in repr(first)
    assert SESSION_KEY not in repr(first)


async def test_memory_driver_rejects_same_identity_with_different_secrets():
    driver = InMemoryRuntimeDriver(clock=lambda: NOW)
    original = _spec()
    await driver.ensure(original)

    with pytest.raises(RuntimeIdentityConflict):
        await driver.ensure(replace(original, relay_capability="z" * 43))


async def test_memory_driver_refuses_expired_generation_and_release_is_idempotent():
    driver = InMemoryRuntimeDriver(clock=lambda: NOW + timedelta(hours=1))
    spec = _spec()
    with pytest.raises(RuntimeExpired):
        await driver.ensure(spec)

    first = await driver.release(spec.identity)
    second = await driver.release(spec.identity)
    assert first == second
    observation = await driver.inspect(spec.identity)
    assert observation.agent is RuntimeResourceState.ABSENT
    assert observation.expired is True
