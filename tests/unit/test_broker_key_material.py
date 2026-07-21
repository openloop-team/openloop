"""Socket-free unit tests for the config-distributed broker key material.

These cover the pure helpers `build_broker_client`/`build_broker_service` use to
turn the external-mode configuration surface (base64 seeds and publics) into
usable keys, and the decision-11 cross-boundary reuse gate that keeps a held
broker root (capability/runtime) from secretly reproducing a key the app trusts
it *not* to hold (a receipt or identity public).
"""

import base64

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import SecretStr

from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker_rpc.identity import (
    WorkloadIdentityIssuer,
    WorkloadIdentityVerifier,
    WorkloadIntent,
)
from openloop.wiring.broker import (
    _decode_identity_seed,
    _decode_public_keys,
    _decode_roots,
    _derive_receipt_key,
    _reject_cross_boundary_reuse,
)

_DOMAIN = "broker-receipt"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _fixed_clock():
    from datetime import UTC, datetime

    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    return lambda: stamp


def test_identity_seed_and_public_round_trip():
    # A base64 32-byte seed decodes to the private key that signs tokens the
    # matching decoded public verifies — the whole external-mode identity seam.
    seed = bytes(range(32))
    private = _decode_identity_seed(SecretStr(_b64(seed)))
    assert isinstance(private, Ed25519PrivateKey)

    public_raw = private.public_key().public_bytes_raw()
    publics = _decode_public_keys("identity", {"identity-v1": _b64(public_raw)})
    assert isinstance(publics["identity-v1"], Ed25519PublicKey)

    clock = _fixed_clock()
    issuer = WorkloadIdentityIssuer(
        private_key=private,
        key_id="identity-v1",
        issuer="openloop-app",
        audience="openloop-broker",
        clock=clock,
    )
    verifier = WorkloadIdentityVerifier(
        public_keys=publics,
        issuer="openloop-app",
        audience="openloop-broker",
        clock=clock,
    )
    token = issuer.issue(
        owner=BrokerOwner("openloop", "coding-worker"),
        worker_instance_id=__import__("uuid").uuid4(),
        assignment_id=__import__("uuid").uuid4(),
        isolation_mode=IsolationMode.DEDICATED,
        required_isolation=IsolationMode.SHARED,
        intents={WorkloadIntent.CREATE_JOB},
    )
    principal = verifier.verify(token)
    assert principal.owner == BrokerOwner("openloop", "coding-worker")


def test_decode_identity_seed_rejects_malformed_base64():
    with pytest.raises(ValueError, match="not valid base64"):
        _decode_identity_seed(SecretStr("not base64 !!!"))


def test_decode_identity_seed_rejects_wrong_length():
    with pytest.raises(ValueError, match="32 bytes"):
        _decode_identity_seed(SecretStr(_b64(b"tooshort")))


def test_decode_public_keys_rejects_empty_map():
    with pytest.raises(ValueError, match="must be set"):
        _decode_public_keys("identity", {})


def test_decode_public_keys_rejects_malformed_base64():
    with pytest.raises(ValueError, match="not valid base64"):
        _decode_public_keys("receipt", {"v1": "%%%not base64%%%"})


def test_decode_public_keys_rejects_wrong_length():
    with pytest.raises(ValueError, match="32 bytes"):
        _decode_public_keys("receipt", {"v1": _b64(b"short")})


def test_decode_roots_rejects_reused_root_across_versions():
    encoded = _b64(bytes([5]) * 32)

    with pytest.raises(ValueError, match="fake rotation"):
        _decode_roots(
            "receipt",
            {"receipt-v1": SecretStr(encoded), "receipt-v2": SecretStr(encoded)},
            "receipt-v2",
        )


def test_cross_boundary_reuse_positive_receipt_hkdf_match():
    # A held broker root that HKDF-derives (under the receipt domain) the very
    # receipt public the broker trusts is a shared trust line — reject.
    root = bytes([7]) * 32
    receipt_pub = _derive_receipt_key(root, _DOMAIN, "receipt-key-v1").public_key()
    with pytest.raises(ValueError, match=_DOMAIN):
        _reject_cross_boundary_reuse(
            {"capability": {"cap-v1": root}},
            {"receipt-key-v1": receipt_pub},
            {"identity-v1": Ed25519PrivateKey.generate().public_key()},
            receipt_domain=_DOMAIN,
        )


def test_cross_boundary_reuse_positive_identity_direct_seed_match():
    # An operator pasting a held root directly as the identity seed: the root's
    # own Ed25519 public equals a configured identity public — reject.
    root = bytes([9]) * 32
    identity_pub = Ed25519PrivateKey.from_private_bytes(root).public_key()
    with pytest.raises(ValueError, match=_DOMAIN):
        _reject_cross_boundary_reuse(
            {"runtime": {"runtime-v1": root}},
            {"receipt-key-v1": Ed25519PrivateKey.generate().public_key()},
            {"identity-v1": identity_pub},
            receipt_domain=_DOMAIN,
        )


def test_cross_boundary_reuse_positive_identity_hkdf_match():
    # A held root HKDF-derived under the receipt domain (keyed by the identity
    # key id) equalling a configured identity public — reject.
    root = bytes([11]) * 32
    identity_pub = _derive_receipt_key(root, _DOMAIN, "identity-v1").public_key()
    with pytest.raises(ValueError, match=_DOMAIN):
        _reject_cross_boundary_reuse(
            {"capability": {"cap-v1": root}},
            {"receipt-key-v1": Ed25519PrivateKey.generate().public_key()},
            {"identity-v1": identity_pub},
            receipt_domain=_DOMAIN,
        )


def test_cross_boundary_reuse_negative_distinct_roots_and_publics_pass():
    # Independent held roots and unrelated configured publics must NOT trip the
    # gate (no false positive that would block a correct external deployment).
    held = {
        "capability": {"cap-v1": bytes([1]) * 32},
        "runtime": {"runtime-v1": bytes([2]) * 32},
    }
    receipt_publics = {
        "receipt-key-v1": Ed25519PrivateKey.generate().public_key(),
    }
    identity_publics = {
        "identity-v1": Ed25519PrivateKey.generate().public_key(),
    }
    _reject_cross_boundary_reuse(
        held, receipt_publics, identity_publics, receipt_domain=_DOMAIN
    )
