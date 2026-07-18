"""Versioned domain-separated credentials for privileged runtimes."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
import hmac as stdlib_hmac
import re
import struct
from uuid import UUID

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from openloop.broker.models import (
    BrokerOwner,
    validate_identifier,
    validate_opaque_ref,
    validate_positive_bigint,
    validate_sha256,
    validate_uuid,
)


_CONVERSATION_INFO = b"openloop.runtime-secrets.v1/conversation"
_RELAY_INFO = b"openloop.runtime-secrets.v1/relay"
_SESSION_INFO = b"openloop.runtime-secrets.v1/session"
_CAPABILITY_VERIFY_INFO = b"openloop.runtime-secrets.v1/capability-verification"
_DURABLE_VERIFY_INFO = b"openloop.runtime-secrets.v1/durable-verification"
_BASE64URL_43 = re.compile(r"[A-Za-z0-9_-]{43}\Z")


class RuntimeSecretProblem(Exception):
    """A runtime secret root, version, or integrity check was rejected."""

    def __init__(self) -> None:
        super().__init__("runtime secret material rejected")


class RuntimeSecretRootRing:
    def __init__(
        self,
        roots: Mapping[str, bytes],
        *,
        current_version: str,
    ) -> None:
        try:
            validate_identifier("current_version", current_version)
        except (TypeError, ValueError) as error:
            raise RuntimeSecretProblem() from error
        if not isinstance(roots, Mapping) or not roots:
            raise RuntimeSecretProblem()
        validated: dict[str, bytes] = {}
        for version, root in roots.items():
            try:
                validate_identifier("key_version", version)
            except (TypeError, ValueError) as error:
                raise RuntimeSecretProblem() from error
            if not isinstance(root, bytes) or not 32 <= len(root) <= 64:
                raise RuntimeSecretProblem()
            validated[version] = bytes(root)
        if len(validated) != len(roots) or current_version not in validated:
            raise RuntimeSecretProblem()
        self._roots = validated
        self.current_version = current_version

    def _root(self, version: str) -> bytes:
        try:
            validate_identifier("key_version", version)
        except (TypeError, ValueError) as error:
            raise RuntimeSecretProblem() from error
        root = self._roots.get(version)
        if root is None:
            raise RuntimeSecretProblem()
        return root


def _length_prefixed(value: bytes) -> bytes:
    if len(value) > 65535:
        raise RuntimeSecretProblem()
    return struct.pack(">H", len(value)) + value


def _subkey(root: bytes, info: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    ).derive(root)


def _mac(key: bytes, value: bytes) -> bytes:
    signer = hmac.HMAC(key, hashes.SHA256())
    signer.update(value)
    return signer.finalize()


def _token(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _context(
    domain: bytes,
    key_version: str,
    owner: BrokerOwner,
    job_id: UUID,
    conversation_id: UUID,
    *,
    generation: int | None = None,
    durable_state_ref: str | None = None,
) -> bytes:
    if not isinstance(owner, BrokerOwner):
        raise TypeError("owner must be a BrokerOwner")
    validate_identifier("key_version", key_version)
    validate_uuid("job_id", job_id)
    validate_uuid("conversation_id", conversation_id)
    values = [
        _length_prefixed(domain),
        _length_prefixed(key_version.encode("utf-8")),
        _length_prefixed(owner.tenant_id.encode("utf-8")),
        _length_prefixed(owner.workload_subject.encode("utf-8")),
        _length_prefixed(job_id.bytes),
        _length_prefixed(conversation_id.bytes),
    ]
    if generation is not None:
        validate_positive_bigint("generation", generation)
        values.append(struct.pack(">Q", generation))
    if durable_state_ref is not None:
        validate_opaque_ref("durable_state_ref", durable_state_ref)
        values.append(_length_prefixed(durable_state_ref.encode("utf-8")))
    return b"".join(values)


@dataclass(frozen=True, slots=True, repr=False)
class DerivedRuntimeSecrets:
    runtime_key_version: str
    durable_key_version: str
    relay_capability: str = field(repr=False)
    session_api_key: str = field(repr=False)
    conversation_secret: str = field(repr=False)
    capability_digest: str = field(repr=False)
    durable_digest: str = field(repr=False)

    def __post_init__(self) -> None:
        validate_identifier("runtime_key_version", self.runtime_key_version)
        validate_identifier("durable_key_version", self.durable_key_version)
        for name in (
            "relay_capability",
            "session_api_key",
            "conversation_secret",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or _BASE64URL_43.fullmatch(value) is None
            ):
                raise ValueError(f"{name} must be a 43-character base64url token")
            try:
                decoded = base64.urlsafe_b64decode(value + "=")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} encoding is invalid") from error
            if len(decoded) != 32:
                raise ValueError(f"{name} encoding is invalid")
        validate_sha256("capability_digest", self.capability_digest)
        validate_sha256("durable_digest", self.durable_digest)

    def __repr__(self) -> str:
        return (
            "DerivedRuntimeSecrets("
            f"runtime_key_version={self.runtime_key_version!r}, "
            f"durable_key_version={self.durable_key_version!r}, "
            "relay_capability=<redacted>, session_api_key=<redacted>, "
            "conversation_secret=<redacted>, capability_digest=<redacted>, "
            "durable_digest=<redacted>)"
        )


class RuntimeSecretAuthority:
    def __init__(self, roots: RuntimeSecretRootRing) -> None:
        if not isinstance(roots, RuntimeSecretRootRing):
            raise TypeError("roots must be RuntimeSecretRootRing")
        self._roots = roots

    @property
    def current_version(self) -> str:
        return self._roots.current_version

    def _conversation(
        self,
        owner: BrokerOwner,
        job_id: UUID,
        conversation_id: UUID,
        durable_key_version: str,
    ) -> bytes:
        root = self._roots._root(durable_key_version)
        context = _context(
            _CONVERSATION_INFO,
            durable_key_version,
            owner,
            job_id,
            conversation_id,
        )
        return _mac(_subkey(root, _CONVERSATION_INFO), context)

    def durable_digest_for(
        self,
        owner: BrokerOwner,
        job_id: UUID,
        conversation_id: UUID,
        durable_state_ref: str,
        durable_key_version: str,
    ) -> str:
        root = self._roots._root(durable_key_version)
        conversation = self._conversation(
            owner,
            job_id,
            conversation_id,
            durable_key_version,
        )
        context = _context(
            _DURABLE_VERIFY_INFO,
            durable_key_version,
            owner,
            job_id,
            conversation_id,
            durable_state_ref=durable_state_ref,
        )
        return _mac(
            _subkey(root, _DURABLE_VERIFY_INFO),
            context + _length_prefixed(conversation),
        ).hex()

    def derive(
        self,
        owner: BrokerOwner,
        job_id: UUID,
        conversation_id: UUID,
        generation: int,
        durable_state_ref: str,
        *,
        runtime_key_version: str,
        durable_key_version: str,
    ) -> DerivedRuntimeSecrets:
        validate_positive_bigint("generation", generation)
        validate_opaque_ref("durable_state_ref", durable_state_ref)
        runtime_root = self._roots._root(runtime_key_version)
        conversation = self._conversation(
            owner,
            job_id,
            conversation_id,
            durable_key_version,
        )
        relay_context = _context(
            _RELAY_INFO,
            runtime_key_version,
            owner,
            job_id,
            conversation_id,
            generation=generation,
        )
        session_context = _context(
            _SESSION_INFO,
            runtime_key_version,
            owner,
            job_id,
            conversation_id,
            generation=generation,
        )
        relay = _mac(_subkey(runtime_root, _RELAY_INFO), relay_context)
        session = _mac(_subkey(runtime_root, _SESSION_INFO), session_context)
        capability_context = _context(
            _CAPABILITY_VERIFY_INFO,
            runtime_key_version,
            owner,
            job_id,
            conversation_id,
            generation=generation,
        )
        capability_digest = _mac(
            _subkey(runtime_root, _CAPABILITY_VERIFY_INFO),
            capability_context
            + _length_prefixed(relay)
            + _length_prefixed(session),
        ).hex()
        durable_digest = self.durable_digest_for(
            owner,
            job_id,
            conversation_id,
            durable_state_ref,
            durable_key_version,
        )
        return DerivedRuntimeSecrets(
            runtime_key_version=runtime_key_version,
            durable_key_version=durable_key_version,
            relay_capability=_token(relay),
            session_api_key=_token(session),
            conversation_secret=_token(conversation),
            capability_digest=capability_digest,
            durable_digest=durable_digest,
        )

    @staticmethod
    def verify_durable(
        values: DerivedRuntimeSecrets,
        expected_digest: str,
    ) -> bool:
        if not isinstance(values, DerivedRuntimeSecrets):
            raise TypeError("values must be DerivedRuntimeSecrets")
        validate_sha256("expected_digest", expected_digest)
        return stdlib_hmac.compare_digest(values.durable_digest, expected_digest)

    @staticmethod
    def verify_capability(
        values: DerivedRuntimeSecrets,
        expected_digest: str,
    ) -> bool:
        if not isinstance(values, DerivedRuntimeSecrets):
            raise TypeError("values must be DerivedRuntimeSecrets")
        validate_sha256("expected_digest", expected_digest)
        return stdlib_hmac.compare_digest(
            values.capability_digest,
            expected_digest,
        )


__all__ = [
    "DerivedRuntimeSecrets",
    "RuntimeSecretAuthority",
    "RuntimeSecretProblem",
    "RuntimeSecretRootRing",
]
