from dataclasses import replace
from uuid import UUID

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openloop.broker.models import (
    SignedCheckpointReceipt,
    VerifiedCheckpointReceipt,
)
from openloop.broker_control.receipts import (
    CheckpointReceiptIssuer,
    CheckpointReceiptProblem,
    CheckpointReceiptVerifier,
    receipt_key,
)
from openloop.broker_rpc.keys import VerificationKeySet


JOB_ID = UUID("00000000-0000-4000-8000-000000000701")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000702")


def _receipt(**changes):
    values = {
        "issuer": "checkpoint-store",
        "receipt_id": "receipt-0001",
        "tenant_id": "tenant-a",
        "job_id": JOB_ID,
        "conversation_id": CONVERSATION_ID,
        "generation": 1,
        "barrier_id": "barrier-0001",
        "artifact_id": "artifact-0001",
        "base_commit": "a" * 40,
        "ciphertext_sha256": "b" * 64,
        "plaintext_sha256": "c" * 64,
        "byte_count": 1024,
        "store_version": "store-v1",
        "envelope_version": "envelope-v1",
        "key_version": "artifact-v1",
        "durable_write_sequence": 7,
    }
    values.update(changes)
    return VerifiedCheckpointReceipt(**values)


def _pair(private_key, *, keys=None):
    issuer = CheckpointReceiptIssuer(
        private_key=private_key,
        key_id="receipt-v1",
        issuer="checkpoint-store",
    )
    verifier = CheckpointReceiptVerifier(
        public_keys=VerificationKeySet(
            keys or {"receipt-v1": private_key.public_key()}
        ),
        issuer="checkpoint-store",
    )
    return issuer, verifier


def test_signed_receipt_round_trip_is_strict_redacted_and_lookup_keyed():
    private_key = Ed25519PrivateKey.generate()
    issuer, verifier = _pair(private_key)
    signed = issuer.issue(_receipt())

    verified = verifier.verify(signed)

    assert verified == _receipt()
    assert receipt_key(verified).tenant_id == "tenant-a"
    assert receipt_key(verified).barrier_id == "barrier-0001"
    assert signed.value not in repr(signed)
    assert signed.value not in str(signed)


def test_verifier_accepts_old_receipt_while_rotation_keys_overlap():
    old = Ed25519PrivateKey.generate()
    new = Ed25519PrivateKey.generate()
    old_issuer, verifier = _pair(
        old,
        keys={"receipt-v1": old.public_key(), "receipt-v2": new.public_key()},
    )

    assert verifier.verify(old_issuer.issue(_receipt())) == _receipt()


def test_verifier_rejects_unknown_key_algorithm_confusion_and_claim_changes():
    private_key = Ed25519PrivateKey.generate()
    issuer, verifier = _pair(private_key)
    good = issuer.issue(_receipt())
    claims = jwt.decode(
        good.value,
        options={"verify_signature": False},
        algorithms=["EdDSA"],
    )

    confused = jwt.encode(
        claims,
        "not-an-ed25519-key-but-long-enough",
        algorithm="HS256",
        headers={"kid": "receipt-v1", "typ": "OPENLOOP-CHECKPOINT+JWT"},
    )
    extra = jwt.encode(
        {**claims, "unexpected": "value"},
        private_key,
        algorithm="EdDSA",
        headers={"kid": "receipt-v1", "typ": "OPENLOOP-CHECKPOINT+JWT"},
    )
    wrong_tenant = jwt.encode(
        {**claims, "tenant_id": ""},
        private_key,
        algorithm="EdDSA",
        headers={"kid": "receipt-v1", "typ": "OPENLOOP-CHECKPOINT+JWT"},
    )

    for value in (confused, extra, wrong_tenant):
        with pytest.raises(CheckpointReceiptProblem):
            verifier.verify(SignedCheckpointReceipt(value))

    other = Ed25519PrivateKey.generate()
    _, wrong_verifier = _pair(other)
    with pytest.raises(CheckpointReceiptProblem):
        wrong_verifier.verify(good)


def test_issuer_refuses_claims_for_another_issuer():
    private_key = Ed25519PrivateKey.generate()
    issuer, _ = _pair(private_key)

    with pytest.raises(ValueError, match="issuer"):
        issuer.issue(replace(_receipt(), issuer="other-store"))


def test_opaque_signed_receipt_rejects_non_ascii_and_oversized_input():
    with pytest.raises(ValueError, match="encoding"):
        SignedCheckpointReceipt("header.payload.signaturé")
    with pytest.raises(ValueError, match="encoding"):
        SignedCheckpointReceipt("A" * (16 * 1024 + 1))
