"""Versioned domain-separated job control capabilities."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
import hmac as stdlib_hmac
import os
import re
import struct
from uuid import UUID

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from openloop.broker.models import (
    BrokerOwner,
    IsolationMode,
    JobAuthorizationMetadata,
    validate_identifier,
    validate_positive_bigint,
    validate_uuid,
)

from .keys import KeyFileProblem, load_private_bytes


_CAPABILITY_TEXT = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_ROOT_TEXT = re.compile(rb"[A-Za-z0-9_-]{43}\n?\Z")
_DERIVE_INFO = b"openloop.job-control.v1/capability"
_VERIFY_INFO = b"openloop.job-control.v1/verification"
_CONTEXT_DOMAIN = b"openloop.job-control.v1"


class CapabilityProblem(Exception):
    def __init__(self) -> None:
        super().__init__("job capability rejected")


@dataclass(frozen=True, slots=True)
class JobCapability:
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _CAPABILITY_TEXT.fullmatch(
            self.value
        ) is None:
            raise ValueError("job capability encoding is invalid")
        try:
            decoded = base64.urlsafe_b64decode(self.value + "=")
        except (ValueError, TypeError) as error:
            raise ValueError("job capability encoding is invalid") from error
        if len(decoded) != 32:
            raise ValueError("job capability encoding is invalid")

    def _bytes(self) -> bytes:
        return base64.urlsafe_b64decode(self.value + "=")


def _encode_root(data: bytes) -> bytes:
    if _ROOT_TEXT.fullmatch(data) is None:
        raise CapabilityProblem()
    text = data[:-1] if data.endswith(b"\n") else data
    try:
        root = base64.urlsafe_b64decode(text + b"=")
    except (ValueError, TypeError) as error:
        raise CapabilityProblem() from error
    if len(root) != 32:
        raise CapabilityProblem()
    return root


class CapabilityRootRing:
    def __init__(
        self, roots: Mapping[str, bytes], *, current_version: str
    ) -> None:
        try:
            validate_identifier("current_version", current_version)
        except (TypeError, ValueError) as error:
            raise CapabilityProblem() from error
        if not isinstance(roots, Mapping) or not roots:
            raise CapabilityProblem()
        validated: dict[str, bytes] = {}
        for version, root in roots.items():
            try:
                validate_identifier("key_version", version)
            except (TypeError, ValueError) as error:
                raise CapabilityProblem() from error
            if not isinstance(root, bytes) or len(root) != 32:
                raise CapabilityProblem()
            validated[version] = bytes(root)
        if current_version not in validated:
            raise CapabilityProblem()
        self._roots = validated
        self.current_version = current_version

    @classmethod
    def load(
        cls,
        paths: Mapping[str, str | os.PathLike[str]],
        *,
        current_version: str,
        expected_uid: int | None = None,
    ) -> "CapabilityRootRing":
        try:
            roots = {
                version: _encode_root(
                    load_private_bytes(path, expected_uid=expected_uid)
                )
                for version, path in paths.items()
            }
            return cls(roots, current_version=current_version)
        except CapabilityProblem:
            raise
        except KeyFileProblem as error:
            raise CapabilityProblem() from error

    def _root(self, version: str) -> bytes:
        root = self._roots.get(version)
        if root is None:
            raise CapabilityProblem()
        return root


def _subkey(root: bytes, info: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    ).derive(root)


def _length_prefixed(value: bytes) -> bytes:
    if len(value) > 65535:
        raise CapabilityProblem()
    return struct.pack(">H", len(value)) + value


def _context(
    owner: BrokerOwner, job_id: UUID, key_version: str, epoch: int
) -> bytes:
    if not isinstance(owner, BrokerOwner):
        raise TypeError("owner must be a BrokerOwner")
    validate_uuid("job_id", job_id)
    validate_identifier("key_version", key_version)
    validate_positive_bigint("epoch", epoch)
    return b"".join(
        (
            _length_prefixed(_CONTEXT_DOMAIN),
            _length_prefixed(key_version.encode("utf-8")),
            _length_prefixed(owner.tenant_id.encode("utf-8")),
            _length_prefixed(owner.workload_subject.encode("utf-8")),
            _length_prefixed(job_id.bytes),
            struct.pack(">Q", epoch),
        )
    )


def _mac(key: bytes, data: bytes) -> bytes:
    signer = hmac.HMAC(key, hashes.SHA256())
    signer.update(data)
    return signer.finalize()


class JobCapabilityAuthority:
    def __init__(self, roots: CapabilityRootRing) -> None:
        if not isinstance(roots, CapabilityRootRing):
            raise TypeError("roots must be CapabilityRootRing")
        self._roots = roots

    def _values(
        self, owner: BrokerOwner, job_id: UUID, key_version: str, epoch: int
    ) -> tuple[bytes, str]:
        root = self._roots._root(key_version)
        context = _context(owner, job_id, key_version, epoch)
        capability = _mac(_subkey(root, _DERIVE_INFO), context)
        digest = _mac(_subkey(root, _VERIFY_INFO), capability).hex()
        return capability, digest

    def issue_metadata(
        self,
        owner: BrokerOwner,
        job_id: UUID,
        minimum_isolation: IsolationMode,
    ) -> JobAuthorizationMetadata:
        if not isinstance(minimum_isolation, IsolationMode):
            raise TypeError("minimum_isolation must be IsolationMode")
        version = self._roots.current_version
        _, digest = self._values(owner, job_id, version, 1)
        return JobAuthorizationMetadata(version, 1, digest)

    __call__ = issue_metadata

    def digest_for(
        self, owner: BrokerOwner, job_id: UUID, key_version: str, epoch: int
    ) -> str:
        return self._values(owner, job_id, key_version, epoch)[1]

    def derive(
        self,
        owner: BrokerOwner,
        job_id: UUID,
        metadata: JobAuthorizationMetadata,
    ) -> JobCapability:
        if not isinstance(metadata, JobAuthorizationMetadata):
            raise TypeError("metadata must be JobAuthorizationMetadata")
        capability, digest = self._values(
            owner, job_id, metadata.key_version, metadata.epoch
        )
        if not stdlib_hmac.compare_digest(digest, metadata.capability_digest):
            raise CapabilityProblem()
        encoded = base64.urlsafe_b64encode(capability).rstrip(b"=").decode("ascii")
        return JobCapability(encoded)

    def verify(
        self,
        owner: BrokerOwner,
        job_id: UUID,
        metadata: JobAuthorizationMetadata,
        capability: JobCapability,
    ) -> bool:
        if not isinstance(metadata, JobAuthorizationMetadata):
            raise TypeError("metadata must be JobAuthorizationMetadata")
        if not isinstance(capability, JobCapability):
            raise TypeError("capability must be JobCapability")
        root = self._roots._root(metadata.key_version)
        candidate = _mac(_subkey(root, _VERIFY_INFO), capability._bytes()).hex()
        return stdlib_hmac.compare_digest(
            candidate, metadata.capability_digest
        )

