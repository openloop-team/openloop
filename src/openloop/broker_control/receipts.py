"""Strict Ed25519 checkpoint-receipt issuance and verification boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from openloop.broker.models import (
    SignedCheckpointReceipt,
    VerifiedCheckpointReceipt,
    validate_identifier,
    validate_tenant_id,
    validate_uuid,
)
from openloop.broker_rpc.keys import VerificationKeySet


RECEIPT_FORMAT_VERSION = 1
RECEIPT_AUDIENCE = "openloop:checkpoint-receipt"
RECEIPT_TYPE = "OPENLOOP-CHECKPOINT+JWT"


class CheckpointReceiptProblem(Exception):
    """Safe failure raised when a checkpoint receipt is not authentic/canonical."""

    def __init__(self) -> None:
        super().__init__("checkpoint receipt rejected")


@dataclass(frozen=True, slots=True)
class CheckpointReceiptKey:
    """Deterministic lookup identity used by recovery after caller loss."""

    tenant_id: str
    job_id: UUID
    conversation_id: UUID
    generation: int
    barrier_id: str

    def __post_init__(self) -> None:
        validate_tenant_id(self.tenant_id)
        validate_uuid("job_id", self.job_id)
        validate_uuid("conversation_id", self.conversation_id)
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("generation must be an integer")
        if self.generation <= 0:
            raise ValueError("generation must be positive")
        validate_identifier("barrier_id", self.barrier_id)


def receipt_key(receipt: VerifiedCheckpointReceipt) -> CheckpointReceiptKey:
    if not isinstance(receipt, VerifiedCheckpointReceipt):
        raise TypeError("receipt must be VerifiedCheckpointReceipt")
    return CheckpointReceiptKey(
        receipt.tenant_id,
        receipt.job_id,
        receipt.conversation_id,
        receipt.generation,
        receipt.barrier_id,
    )


@runtime_checkable
class CheckpointReceiptLocator(Protocol):
    async def lookup(
        self, key: CheckpointReceiptKey
    ) -> SignedCheckpointReceipt | None: ...


class CheckpointReceiptIssuer:
    """Trusted checkpoint-store signer; the broker should hold only its public key."""

    def __init__(
        self,
        *,
        private_key: Ed25519PrivateKey,
        key_id: str,
        issuer: str,
        audience: str = RECEIPT_AUDIENCE,
    ) -> None:
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("private_key must be Ed25519PrivateKey")
        validate_identifier("key_id", key_id)
        validate_identifier("issuer", issuer)
        validate_identifier("audience", audience)
        self._private_key = private_key
        self._key_id = key_id
        self._issuer = issuer
        self._audience = audience

    def issue(self, receipt: VerifiedCheckpointReceipt) -> SignedCheckpointReceipt:
        if not isinstance(receipt, VerifiedCheckpointReceipt):
            raise TypeError("receipt must be VerifiedCheckpointReceipt")
        if receipt.issuer != self._issuer:
            raise ValueError("receipt issuer does not match signer")
        payload = {
            "iss": receipt.issuer,
            "aud": self._audience,
            "ver": RECEIPT_FORMAT_VERSION,
            "receipt_id": receipt.receipt_id,
            "tenant_id": receipt.tenant_id,
            "job_id": str(receipt.job_id),
            "conversation_id": str(receipt.conversation_id),
            "generation": receipt.generation,
            "barrier_id": receipt.barrier_id,
            "artifact_id": receipt.artifact_id,
            "base_commit": receipt.base_commit,
            "ciphertext_sha256": receipt.ciphertext_sha256,
            "plaintext_sha256": receipt.plaintext_sha256,
            "byte_count": receipt.byte_count,
            "store_version": receipt.store_version,
            "envelope_version": receipt.envelope_version,
            "key_version": receipt.key_version,
            "durable_write_sequence": receipt.durable_write_sequence,
        }
        encoded = jwt.encode(
            payload,
            self._private_key,
            algorithm="EdDSA",
            headers={"kid": self._key_id, "typ": RECEIPT_TYPE},
        )
        return SignedCheckpointReceipt(encoded)


class CheckpointReceiptVerifier:
    _HEADER_FIELDS = frozenset({"alg", "typ", "kid"})
    _CLAIM_FIELDS = frozenset(
        {
            "iss",
            "aud",
            "ver",
            "receipt_id",
            "tenant_id",
            "job_id",
            "conversation_id",
            "generation",
            "barrier_id",
            "artifact_id",
            "base_commit",
            "ciphertext_sha256",
            "plaintext_sha256",
            "byte_count",
            "store_version",
            "envelope_version",
            "key_version",
            "durable_write_sequence",
        }
    )

    def __init__(
        self,
        *,
        public_keys: VerificationKeySet,
        issuer: str,
        audience: str = RECEIPT_AUDIENCE,
    ) -> None:
        if not isinstance(public_keys, VerificationKeySet):
            raise TypeError("public_keys must be VerificationKeySet")
        validate_identifier("issuer", issuer)
        validate_identifier("audience", audience)
        self._public_keys = public_keys
        self._issuer = issuer
        self._audience = audience

    @staticmethod
    def _uuid(value: object) -> UUID:
        if not isinstance(value, str):
            raise CheckpointReceiptProblem()
        parsed = UUID(value)
        if str(parsed) != value:
            raise CheckpointReceiptProblem()
        return parsed

    @staticmethod
    def _integer(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CheckpointReceiptProblem()
        return value

    def verify(
        self, token: SignedCheckpointReceipt
    ) -> VerifiedCheckpointReceipt:
        if not isinstance(token, SignedCheckpointReceipt):
            raise TypeError("token must be SignedCheckpointReceipt")
        try:
            header = jwt.get_unverified_header(token.value)
            if set(header) != self._HEADER_FIELDS:
                raise CheckpointReceiptProblem()
            if header["alg"] != "EdDSA" or header["typ"] != RECEIPT_TYPE:
                raise CheckpointReceiptProblem()
            key_id = header["kid"]
            validate_identifier("key_id", key_id)
            key = self._public_keys.snapshot().get(key_id)
            if key is None:
                raise CheckpointReceiptProblem()
            claims = jwt.decode(
                token.value,
                key,
                algorithms=["EdDSA"],
                options={
                    "verify_aud": False,
                    "verify_iss": False,
                    "verify_exp": False,
                    "verify_nbf": False,
                    "verify_iat": False,
                },
            )
            if set(claims) != self._CLAIM_FIELDS:
                raise CheckpointReceiptProblem()
            if (
                claims["iss"] != self._issuer
                or claims["aud"] != self._audience
                or self._integer(claims["ver"]) != RECEIPT_FORMAT_VERSION
            ):
                raise CheckpointReceiptProblem()
            return VerifiedCheckpointReceipt(
                issuer=claims["iss"],
                receipt_id=claims["receipt_id"],
                tenant_id=claims["tenant_id"],
                job_id=self._uuid(claims["job_id"]),
                conversation_id=self._uuid(claims["conversation_id"]),
                generation=self._integer(claims["generation"]),
                barrier_id=claims["barrier_id"],
                artifact_id=claims["artifact_id"],
                base_commit=claims["base_commit"],
                ciphertext_sha256=claims["ciphertext_sha256"],
                plaintext_sha256=claims["plaintext_sha256"],
                byte_count=self._integer(claims["byte_count"]),
                store_version=claims["store_version"],
                envelope_version=claims["envelope_version"],
                key_version=claims["key_version"],
                durable_write_sequence=self._integer(
                    claims["durable_write_sequence"]
                ),
            )
        except CheckpointReceiptProblem:
            raise
        except (
            jwt.PyJWTError,
            KeyError,
            TypeError,
            ValueError,
            UnicodeError,
        ) as error:
            raise CheckpointReceiptProblem() from error


__all__ = [
    "CheckpointReceiptIssuer",
    "CheckpointReceiptKey",
    "CheckpointReceiptLocator",
    "CheckpointReceiptProblem",
    "CheckpointReceiptVerifier",
    "RECEIPT_AUDIENCE",
    "RECEIPT_FORMAT_VERSION",
    "receipt_key",
]
