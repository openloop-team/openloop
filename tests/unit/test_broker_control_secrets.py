from uuid import UUID

import pytest

from openloop.broker.models import BrokerOwner
from openloop.broker_control.secrets import (
    RuntimeSecretAuthority,
    RuntimeSecretProblem,
    RuntimeSecretRootRing,
)


OWNER = BrokerOwner("tenant-a", "workload-a")
JOB_ID = UUID("00000000-0000-4000-8000-000000000301")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000302")
DURABLE_REF = f"local-openhands:v1:{JOB_ID}"
ROOT_V1 = bytes(range(32))
ROOT_V2 = bytes(range(32, 64))


def _authority(*, current="runtime-v1") -> RuntimeSecretAuthority:
    return RuntimeSecretAuthority(
        RuntimeSecretRootRing(
            {"runtime-v1": ROOT_V1, "runtime-v2": ROOT_V2},
            current_version=current,
        )
    )


def test_runtime_secrets_are_deterministic_verified_and_redacted():
    authority = _authority()
    durable_digest = authority.durable_digest_for(
        OWNER,
        JOB_ID,
        CONVERSATION_ID,
        DURABLE_REF,
        "runtime-v1",
    )

    values = authority.derive(
        OWNER,
        JOB_ID,
        CONVERSATION_ID,
        1,
        DURABLE_REF,
        runtime_key_version="runtime-v1",
        durable_key_version="runtime-v1",
    )

    assert values == authority.derive(
        OWNER,
        JOB_ID,
        CONVERSATION_ID,
        1,
        DURABLE_REF,
        runtime_key_version="runtime-v1",
        durable_key_version="runtime-v1",
    )
    assert len(values.relay_capability) == 43
    assert len(values.session_api_key) == 43
    assert len(values.conversation_secret) == 43
    assert values.relay_capability == (
        "AqHmbqyhp-UnSu28EBCIu1UjnOM-K3ogf-l5lPqzZ6E"
    )
    assert values.session_api_key == (
        "py1KIaudV-8pmV-V3TgZrRFI9O7rVXiOmhaKSjB7w1Q"
    )
    assert values.conversation_secret == (
        "iUdqTee-GlA4w7lGZ63FZsmNIcooe3PROwuVnBznCuQ"
    )
    assert values.capability_digest == (
        "659419450471d565466a9843cac8f7748bfc56e3326f51ed0174ad123e474a5b"
    )
    assert values.durable_digest == (
        "83a649ed8d1a50dcc68e72d73f1346bbfab924d7566299385b330221c960b193"
    )
    assert values.durable_digest == durable_digest
    assert authority.verify_durable(values, durable_digest)
    assert authority.verify_capability(values, values.capability_digest)
    rendered = repr(values)
    for protected in (
        values.relay_capability,
        values.session_api_key,
        values.conversation_secret,
        values.capability_digest,
        values.durable_digest,
    ):
        assert protected not in rendered


def test_conversation_is_stable_while_generation_credentials_rotate():
    authority = _authority(current="runtime-v2")
    first = authority.derive(
        OWNER,
        JOB_ID,
        CONVERSATION_ID,
        1,
        DURABLE_REF,
        runtime_key_version="runtime-v1",
        durable_key_version="runtime-v1",
    )
    second = authority.derive(
        OWNER,
        JOB_ID,
        CONVERSATION_ID,
        2,
        DURABLE_REF,
        runtime_key_version="runtime-v2",
        durable_key_version="runtime-v1",
    )

    assert first.conversation_secret == second.conversation_secret
    assert first.durable_digest == second.durable_digest
    assert first.relay_capability != second.relay_capability
    assert first.session_api_key != second.session_api_key
    assert first.capability_digest != second.capability_digest


@pytest.mark.parametrize(
    "change",
    [
        {"owner": BrokerOwner("tenant-b", "workload-a")},
        {"owner": BrokerOwner("tenant-a", "workload-b")},
        {"job_id": UUID("00000000-0000-4000-8000-000000000303")},
        {"conversation_id": UUID("00000000-0000-4000-8000-000000000304")},
        {"durable_state_ref": "local-openhands:v1:different"},
    ],
)
def test_runtime_secret_context_fields_are_domain_separated(change):
    authority = _authority()
    values = dict(
        owner=OWNER,
        job_id=JOB_ID,
        conversation_id=CONVERSATION_ID,
        generation=1,
        durable_state_ref=DURABLE_REF,
        runtime_key_version="runtime-v1",
        durable_key_version="runtime-v1",
    )
    original = authority.derive(**values)
    values.update(change)
    changed = authority.derive(**values)
    assert changed != original


def test_runtime_secret_verification_rejects_wrong_digest():
    values = _authority().derive(
        OWNER,
        JOB_ID,
        CONVERSATION_ID,
        1,
        DURABLE_REF,
        runtime_key_version="runtime-v1",
        durable_key_version="runtime-v1",
    )
    assert not _authority().verify_durable(values, "0" * 64)
    assert not _authority().verify_capability(values, "0" * 64)


@pytest.mark.parametrize(
    ("roots", "current"),
    [
        ({}, "runtime-v1"),
        ({"runtime-v1": b"short"}, "runtime-v1"),
        ({"runtime-v1": ROOT_V1}, "missing"),
    ],
)
def test_runtime_root_ring_rejects_invalid_configuration(roots, current):
    with pytest.raises(RuntimeSecretProblem):
        RuntimeSecretRootRing(roots, current_version=current)


def test_runtime_secret_authority_rejects_unknown_retired_version_safely():
    authority = RuntimeSecretAuthority(
        RuntimeSecretRootRing(
            {"runtime-v2": ROOT_V2},
            current_version="runtime-v2",
        )
    )
    with pytest.raises(RuntimeSecretProblem) as captured:
        authority.derive(
            OWNER,
            JOB_ID,
            CONVERSATION_ID,
            1,
            DURABLE_REF,
            runtime_key_version="runtime-v1",
            durable_key_version="runtime-v1",
        )
    assert ROOT_V1.hex() not in str(captured.value)
