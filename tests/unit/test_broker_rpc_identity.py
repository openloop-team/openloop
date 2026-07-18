from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker_rpc.identity import (
    IdentityProblem,
    WorkloadIdentityIssuer,
    WorkloadIdentityToken,
    WorkloadIdentityVerifier,
    WorkloadIntent,
)


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
OWNER = BrokerOwner("tenant-a", "workload-a")
WORKER_ID = UUID("00000000-0000-4000-8000-000000000101")
ASSIGNMENT_ID = UUID("00000000-0000-4000-8000-000000000102")
JTI = UUID("00000000-0000-4000-8000-000000000103")


def _issuer(private_key, *, ttl_seconds=300):
    return WorkloadIdentityIssuer(
        private_key=private_key,
        key_id="issuer-v1",
        issuer="openloop-control",
        audience="openloop:broker-control",
        clock=lambda: NOW,
        id_factory=lambda: JTI,
        ttl_seconds=ttl_seconds,
    )


def _issue(private_key, **changes):
    values = dict(
        owner=OWNER,
        worker_instance_id=WORKER_ID,
        assignment_id=ASSIGNMENT_ID,
        isolation_mode=IsolationMode.DEDICATED,
        required_isolation=IsolationMode.SHARED,
        intents=frozenset(
            {
                WorkloadIntent.CREATE_JOB,
                WorkloadIntent.INSPECT_JOB,
                WorkloadIntent.START_SEGMENT,
            }
        ),
    )
    values.update(changes)
    return _issuer(private_key).issue(**values)


def _verifier(private_key, *, now=NOW, keys=None):
    return WorkloadIdentityVerifier(
        public_keys=keys or {"issuer-v1": private_key.public_key()},
        issuer="openloop-control",
        audience="openloop:broker-control",
        clock=lambda: now,
    )


def test_ed25519_identity_round_trip_is_strict_and_redacted():
    private_key = Ed25519PrivateKey.generate()
    token = _issue(private_key)
    principal = _verifier(private_key).verify(token)
    assert principal.owner == OWNER
    assert principal.worker_instance_id == WORKER_ID
    assert principal.assignment_id == ASSIGNMENT_ID
    assert principal.isolation_mode is IsolationMode.DEDICATED
    assert principal.required_isolation is IsolationMode.SHARED
    assert principal.intents == frozenset(
        {
            WorkloadIntent.CREATE_JOB,
            WorkloadIntent.INSPECT_JOB,
            WorkloadIntent.START_SEGMENT,
        }
    )
    assert principal.key_id == "issuer-v1"
    assert principal.jwt_id == JTI
    assert token.value not in repr(token)
    assert token.value not in str(token)


def test_required_isolation_cannot_exceed_actual_worker_placement():
    private_key = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError):
        _issue(
            private_key,
            isolation_mode=IsolationMode.SHARED,
            required_isolation=IsolationMode.DEDICATED,
        )


def test_verifier_rejects_algorithm_confusion_and_unknown_key():
    private_key = Ed25519PrivateKey.generate()
    good = _issue(private_key)
    payload = jwt.decode(
        good.value,
        options={"verify_signature": False},
        algorithms=["EdDSA"],
    )
    confused = jwt.encode(
        payload,
        "not-an-ed25519-key-but-long-enough",
        algorithm="HS256",
        headers={"kid": "issuer-v1", "typ": "JWT"},
    )
    with pytest.raises(IdentityProblem):
        _verifier(private_key).verify(WorkloadIdentityToken(confused))
    with pytest.raises(IdentityProblem):
        WorkloadIdentityVerifier(
            public_keys={"other": Ed25519PrivateKey.generate().public_key()},
            issuer="openloop-control",
            audience="openloop:broker-control",
            clock=lambda: NOW,
        ).verify(good)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda claims: {**claims, "aud": ["openloop:broker-control"]},
        lambda claims: {**claims, "extra": "not-approved"},
        lambda claims: {**claims, "iat": True},
        lambda claims: {**claims, "intents": ["CREATE_JOB", "CREATE_JOB"]},
        lambda claims: {**claims, "isolation_mode": "DISPOSABLE"},
        lambda claims: {**claims, "sub": "x" * 257},
    ],
)
def test_verifier_rejects_noncanonical_or_unapproved_claims(mutation):
    private_key = Ed25519PrivateKey.generate()
    good = _issue(private_key)
    claims = jwt.decode(
        good.value,
        options={"verify_signature": False},
        algorithms=["EdDSA"],
    )
    token = jwt.encode(
        mutation(claims),
        private_key,
        algorithm="EdDSA",
        headers={"kid": "issuer-v1", "typ": "JWT"},
    )
    with pytest.raises(IdentityProblem):
        _verifier(private_key).verify(WorkloadIdentityToken(token))


def test_verifier_enforces_five_minute_lifetime_and_clock_skew():
    private_key = Ed25519PrivateKey.generate()
    token = _issue(private_key)
    assert _verifier(private_key, now=NOW + timedelta(seconds=329)).verify(token)
    with pytest.raises(IdentityProblem):
        _verifier(private_key, now=NOW + timedelta(seconds=331)).verify(token)
    with pytest.raises(ValueError):
        _issuer(private_key, ttl_seconds=301)


def test_verifier_accepts_overlapping_keys_during_rotation():
    old = Ed25519PrivateKey.generate()
    new = Ed25519PrivateKey.generate()
    old_token = _issue(old)
    verifier = _verifier(
        old,
        keys={"issuer-v1": old.public_key(), "issuer-v2": new.public_key()},
    )
    assert verifier.verify(old_token).owner == OWNER
