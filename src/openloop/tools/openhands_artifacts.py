"""Authenticated, host-only workspace artifacts for OpenHands cold resume."""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from openloop.tools.openhands_state import (
    OpenHandsKeyDeriver,
    OpenHandsStateError,
    OpenHandsStateLayout,
    validate_state_identifier,
)


_MAGIC = b"OLWART01"
_ENVELOPE_VERSION = 1
_TAG_BYTES = 16
_NONCE_BYTES = 12
_MAX_HEADER_BYTES = 16_384
_MAX_MANIFEST_BYTES = 1_048_576
_CHUNK_BYTES = 1024 * 1024


class WorkspaceArtifactError(RuntimeError):
    """Base error for artifact storage and verification failures."""


class WorkspaceArtifactConflict(WorkspaceArtifactError):
    """A deterministic artifact identity already contains different data."""


class WorkspaceArtifactVerificationError(WorkspaceArtifactError):
    """An artifact did not authenticate or match its expected identity."""


@dataclass(frozen=True, slots=True)
class WorkspaceArtifactIdentity:
    job_id: str
    conversation_id: str
    segment_id: str
    kind: str

    def __post_init__(self) -> None:
        for field in ("job_id", "conversation_id", "segment_id", "kind"):
            object.__setattr__(
                self,
                field,
                validate_state_identifier(getattr(self, field), field=field),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "conversation_id": self.conversation_id,
            "segment_id": self.segment_id,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceArtifactManifest:
    format: str
    base_commit: str
    plaintext_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_state_identifier(self.format, field="artifact format")
        if (
            len(self.base_commit) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in self.base_commit)
        ):
            raise WorkspaceArtifactError(
                "artifact base_commit must be a lowercase Git object ID"
            )
        if self.plaintext_sha256 is not None and (
            len(self.plaintext_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.plaintext_sha256)
        ):
            raise WorkspaceArtifactError("invalid artifact plaintext SHA-256")

    def with_plaintext_sha256(self, digest: str) -> "WorkspaceArtifactManifest":
        return WorkspaceArtifactManifest(
            format=self.format,
            base_commit=self.base_commit,
            plaintext_sha256=digest,
        )


@dataclass(frozen=True, slots=True)
class WorkspaceArtifact:
    identity: WorkspaceArtifactIdentity
    key: str
    ciphertext_sha256: str
    ciphertext_bytes: int
    envelope_version: int
    master_key_id: str


class VerifiedWorkspaceArtifact:
    """A verified plaintext stream whose scratch file is removed on close."""

    def __init__(
        self,
        manifest: WorkspaceArtifactManifest,
        stream: BinaryIO,
        scratch_path: Path,
    ) -> None:
        self.manifest = manifest
        self.stream = stream
        self._scratch_path = scratch_path

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()
        self._scratch_path.unlink(missing_ok=True)

    def __enter__(self) -> "VerifiedWorkspaceArtifact":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


class WorkspaceArtifactStore:
    """Filesystem-backed encrypted store with deterministic, path-safe keys."""

    def __init__(
        self,
        layout: OpenHandsStateLayout,
        keys: OpenHandsKeyDeriver,
        *,
        scratch_root: str | Path | None = None,
    ) -> None:
        self.layout = layout
        self.keys = keys
        selected = (
            Path(scratch_root)
            if scratch_root is not None
            else Path(tempfile.gettempdir()) / "openloop" / "openhands-artifact-scratch"
        )
        if selected.is_symlink():
            raise WorkspaceArtifactError("artifact scratch root cannot be a symlink")
        selected.mkdir(mode=0o700, parents=True, exist_ok=True)
        selected.chmod(0o700)
        self.scratch_root = selected.resolve(strict=True)

    def put_atomic(
        self,
        identity: WorkspaceArtifactIdentity,
        plaintext_stream: BinaryIO,
        manifest: WorkspaceArtifactManifest,
    ) -> WorkspaceArtifact:
        """Encrypt and atomically publish one immutable deterministic artifact."""
        plaintext_path, plaintext_sha256 = self._spool_plaintext(plaintext_stream)
        completed_manifest = manifest.with_plaintext_sha256(plaintext_sha256)
        if (
            manifest.plaintext_sha256 is not None
            and manifest.plaintext_sha256 != plaintext_sha256
        ):
            plaintext_path.unlink(missing_ok=True)
            raise WorkspaceArtifactError("provided plaintext SHA-256 does not match")

        artifact_path = self._artifact_path(identity, create=True)
        lock_path = artifact_path.with_suffix(artifact_path.suffix + ".lock")
        try:
            with self._locked(lock_path):
                if artifact_path.exists():
                    existing = self._descriptor_for(identity, artifact_path)
                    with self.open_verified(existing, identity) as verified:
                        if verified.manifest != completed_manifest:
                            raise WorkspaceArtifactConflict(
                                "artifact identity already contains different content"
                            )
                    return existing

                temporary = self._encrypt_to_temporary(
                    identity, completed_manifest, plaintext_path, artifact_path.parent
                )
                try:
                    os.replace(temporary, artifact_path)
                    self._fsync_directory(artifact_path.parent)
                finally:
                    temporary.unlink(missing_ok=True)
                return self._descriptor_for(identity, artifact_path)
        finally:
            plaintext_path.unlink(missing_ok=True)

    def open_verified(
        self,
        descriptor: WorkspaceArtifact,
        expected_identity: WorkspaceArtifactIdentity,
    ) -> VerifiedWorkspaceArtifact:
        """Authenticate fully before returning any plaintext bytes."""
        if descriptor.identity != expected_identity:
            raise WorkspaceArtifactVerificationError("artifact identity mismatch")
        artifact_path = self._artifact_path(expected_identity, create=False)
        expected_key = artifact_path.relative_to(self.layout.root).as_posix()
        if descriptor.key != expected_key:
            raise WorkspaceArtifactVerificationError("artifact key mismatch")
        if artifact_path.is_symlink():
            raise WorkspaceArtifactVerificationError("artifact cannot be a symlink")

        scratch_path: Path | None = None
        scratch_stream: BinaryIO | None = None
        try:
            with self._open_nofollow(artifact_path) as encrypted:
                digest, size = self._hash_stream(encrypted)
                if digest != descriptor.ciphertext_sha256:
                    raise WorkspaceArtifactVerificationError(
                        "artifact ciphertext checksum mismatch"
                    )
                if size != descriptor.ciphertext_bytes:
                    raise WorkspaceArtifactVerificationError(
                        "artifact ciphertext size mismatch"
                    )
                encrypted.seek(0)
                header_prefix, header, ciphertext_bytes, tag = self._read_envelope(
                    encrypted, size
                )
                if header.get("envelope_version") != _ENVELOPE_VERSION:
                    raise WorkspaceArtifactVerificationError(
                        "unsupported artifact envelope version"
                    )
                if descriptor.envelope_version != _ENVELOPE_VERSION:
                    raise WorkspaceArtifactVerificationError(
                        "artifact descriptor version mismatch"
                    )
                if header.get("master_key_id") != self.keys.master_key_id:
                    raise WorkspaceArtifactVerificationError(
                        "artifact master-key identifier mismatch"
                    )
                if descriptor.master_key_id != self.keys.master_key_id:
                    raise WorkspaceArtifactVerificationError(
                        "artifact descriptor key identifier mismatch"
                    )
                try:
                    nonce = base64.b64decode(
                        str(header["nonce"]).encode("ascii"),
                        altchars=b"-_",
                        validate=True,
                    )
                except Exception as exc:  # noqa: BLE001 — normalize envelope errors
                    raise WorkspaceArtifactVerificationError(
                        "invalid artifact nonce"
                    ) from exc
                if len(nonce) != _NONCE_BYTES:
                    raise WorkspaceArtifactVerificationError("invalid artifact nonce")

                scratch_path, scratch_stream = self._new_scratch()
                decryptor = Cipher(
                    algorithms.AES(self.keys.artifact_key(expected_identity.job_id)),
                    modes.GCM(nonce, tag),
                ).decryptor()
                decryptor.authenticate_additional_data(
                    self._associated_data(header_prefix, expected_identity)
                )
                remaining = ciphertext_bytes
                while remaining:
                    chunk = encrypted.read(min(_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise WorkspaceArtifactVerificationError(
                            "truncated artifact ciphertext"
                        )
                    remaining -= len(chunk)
                    scratch_stream.write(decryptor.update(chunk))
                scratch_stream.write(decryptor.finalize())
                scratch_stream.flush()

            manifest = self._verify_plaintext_payload(
                scratch_stream, expected_identity
            )
            return VerifiedWorkspaceArtifact(manifest, scratch_stream, scratch_path)
        except InvalidTag as exc:
            if scratch_stream is not None:
                scratch_stream.close()
            if scratch_path is not None:
                scratch_path.unlink(missing_ok=True)
            raise WorkspaceArtifactVerificationError(
                "artifact authentication failed"
            ) from exc
        except Exception:
            if scratch_stream is not None:
                scratch_stream.close()
            if scratch_path is not None:
                scratch_path.unlink(missing_ok=True)
            raise

    def delete(self, identity: WorkspaceArtifactIdentity) -> bool:
        artifact_path = self._artifact_path(identity, create=True)
        lock_path = artifact_path.with_suffix(artifact_path.suffix + ".lock")
        with self._locked(lock_path):
            try:
                artifact_path.unlink()
            except FileNotFoundError:
                return False
            self._fsync_directory(artifact_path.parent)
            return True

    def list_orphans(self, older_than: datetime) -> list[WorkspaceArtifactIdentity]:
        """List old deterministic artifacts; malformed/symlink entries are ignored."""
        cutoff = older_than.timestamp()
        found: list[WorkspaceArtifactIdentity] = []
        for job_dir in self.layout.jobs_root.iterdir():
            if job_dir.is_symlink() or not job_dir.is_dir():
                continue
            artifacts = job_dir / "artifacts"
            if artifacts.is_symlink() or not artifacts.is_dir():
                continue
            for conversation_dir in artifacts.iterdir():
                if conversation_dir.is_symlink() or not conversation_dir.is_dir():
                    continue
                for artifact in conversation_dir.glob("*.artifact"):
                    try:
                        if artifact.is_symlink() or artifact.stat().st_mtime >= cutoff:
                            continue
                    except OSError:
                        continue
                    try:
                        segment_id, kind = artifact.stem.rsplit(".", 1)
                        found.append(
                            WorkspaceArtifactIdentity(
                                job_id=job_dir.name,
                                conversation_id=conversation_dir.name,
                                segment_id=segment_id,
                                kind=kind,
                            )
                        )
                    except (OpenHandsStateError, ValueError):
                        continue
        return found

    def _artifact_path(
        self, identity: WorkspaceArtifactIdentity, *, create: bool
    ) -> Path:
        job = self.layout.for_job(identity.job_id)
        conversation_dir = job.artifacts / identity.conversation_id
        if conversation_dir.is_symlink():
            raise WorkspaceArtifactError("artifact directory cannot be a symlink")
        if create:
            conversation_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
            conversation_dir.chmod(0o700)
        resolved_parent = conversation_dir.resolve(strict=create)
        try:
            resolved_parent.relative_to(job.artifacts.resolve(strict=True))
        except ValueError as exc:
            raise WorkspaceArtifactError("artifact path escapes job directory") from exc
        return resolved_parent / f"{identity.segment_id}.{identity.kind}.artifact"

    def _spool_plaintext(self, source: BinaryIO) -> tuple[Path, str]:
        path, stream = self._new_scratch()
        digest = hashlib.sha256()
        try:
            while True:
                chunk = source.read(_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise WorkspaceArtifactError("artifact plaintext stream must be binary")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
            return path, digest.hexdigest()
        except Exception:
            stream.close()
            path.unlink(missing_ok=True)
            raise
        finally:
            if not stream.closed:
                stream.close()

    def _encrypt_to_temporary(
        self,
        identity: WorkspaceArtifactIdentity,
        manifest: WorkspaceArtifactManifest,
        plaintext_path: Path,
        destination_dir: Path,
    ) -> Path:
        nonce = os.urandom(_NONCE_BYTES)
        header = _canonical_json(
            {
                "envelope_version": _ENVELOPE_VERSION,
                "master_key_id": self.keys.master_key_id,
                "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
            }
        )
        header_prefix = _MAGIC + struct.pack(">I", len(header)) + header
        manifest_bytes = _canonical_json(
            {
                "identity": identity.to_dict(),
                "format": manifest.format,
                "base_commit": manifest.base_commit,
                "plaintext_sha256": manifest.plaintext_sha256,
            }
        )
        fd, raw_path = tempfile.mkstemp(
            prefix=".artifact-", suffix=".tmp", dir=destination_dir
        )
        path = Path(raw_path)
        os.chmod(path, 0o600)
        try:
            with os.fdopen(fd, "wb") as output, plaintext_path.open("rb") as plaintext:
                output.write(header_prefix)
                encryptor = Cipher(
                    algorithms.AES(self.keys.artifact_key(identity.job_id)),
                    modes.GCM(nonce),
                ).encryptor()
                encryptor.authenticate_additional_data(
                    self._associated_data(header_prefix, identity)
                )
                output.write(encryptor.update(struct.pack(">I", len(manifest_bytes))))
                output.write(encryptor.update(manifest_bytes))
                while chunk := plaintext.read(_CHUNK_BYTES):
                    output.write(encryptor.update(chunk))
                output.write(encryptor.finalize())
                output.write(encryptor.tag)
                output.flush()
                os.fsync(output.fileno())
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def _descriptor_for(
        self, identity: WorkspaceArtifactIdentity, path: Path
    ) -> WorkspaceArtifact:
        with self._open_nofollow(path) as stream:
            digest, size = self._hash_stream(stream)
        return WorkspaceArtifact(
            identity=identity,
            key=path.relative_to(self.layout.root).as_posix(),
            ciphertext_sha256=digest,
            ciphertext_bytes=size,
            envelope_version=_ENVELOPE_VERSION,
            master_key_id=self.keys.master_key_id,
        )

    @staticmethod
    def _associated_data(
        header_prefix: bytes, identity: WorkspaceArtifactIdentity
    ) -> bytes:
        return header_prefix + b"\0" + _canonical_json(identity.to_dict())

    @staticmethod
    def _read_envelope(
        stream: BinaryIO, total_size: int
    ) -> tuple[bytes, dict, int, bytes]:
        fixed = stream.read(len(_MAGIC) + 4)
        if len(fixed) != len(_MAGIC) + 4 or fixed[: len(_MAGIC)] != _MAGIC:
            raise WorkspaceArtifactVerificationError("invalid artifact envelope")
        header_length = struct.unpack(">I", fixed[len(_MAGIC) :])[0]
        if not 0 < header_length <= _MAX_HEADER_BYTES:
            raise WorkspaceArtifactVerificationError("invalid artifact header length")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise WorkspaceArtifactVerificationError("truncated artifact header")
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkspaceArtifactVerificationError("invalid artifact header") from exc
        if not isinstance(header, dict):
            raise WorkspaceArtifactVerificationError("invalid artifact header")
        prefix = fixed + header_bytes
        ciphertext_bytes = total_size - len(prefix) - _TAG_BYTES
        if ciphertext_bytes <= 4:
            raise WorkspaceArtifactVerificationError("truncated artifact payload")
        stream.seek(total_size - _TAG_BYTES)
        tag = stream.read(_TAG_BYTES)
        if len(tag) != _TAG_BYTES:
            raise WorkspaceArtifactVerificationError("truncated artifact tag")
        stream.seek(len(prefix))
        return prefix, header, ciphertext_bytes, tag

    @staticmethod
    def _verify_plaintext_payload(
        stream: BinaryIO, expected_identity: WorkspaceArtifactIdentity
    ) -> WorkspaceArtifactManifest:
        stream.seek(0)
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            raise WorkspaceArtifactVerificationError("missing encrypted manifest")
        manifest_length = struct.unpack(">I", length_bytes)[0]
        if not 0 < manifest_length <= _MAX_MANIFEST_BYTES:
            raise WorkspaceArtifactVerificationError("invalid encrypted manifest length")
        encoded = stream.read(manifest_length)
        if len(encoded) != manifest_length:
            raise WorkspaceArtifactVerificationError("truncated encrypted manifest")
        try:
            raw = json.loads(encoded)
            if raw["identity"] != expected_identity.to_dict():
                raise WorkspaceArtifactVerificationError(
                    "encrypted artifact identity mismatch"
                )
            manifest = WorkspaceArtifactManifest(
                format=raw["format"],
                base_commit=raw["base_commit"],
                plaintext_sha256=raw["plaintext_sha256"],
            )
        except WorkspaceArtifactVerificationError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize authenticated data errors
            raise WorkspaceArtifactVerificationError(
                "invalid encrypted artifact manifest"
            ) from exc

        payload_offset = stream.tell()
        digest = hashlib.sha256()
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
        if digest.hexdigest() != manifest.plaintext_sha256:
            raise WorkspaceArtifactVerificationError(
                "artifact plaintext checksum mismatch"
            )
        stream.seek(payload_offset)
        return manifest

    def _new_scratch(self) -> tuple[Path, BinaryIO]:
        fd, raw_path = tempfile.mkstemp(
            prefix="openhands-artifact-", suffix=".tmp", dir=self.scratch_root
        )
        path = Path(raw_path)
        os.chmod(path, 0o600)
        return path, os.fdopen(fd, "w+b")

    @staticmethod
    def _hash_stream(stream: BinaryIO) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size

    @staticmethod
    def _open_nofollow(path: Path) -> BinaryIO:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise WorkspaceArtifactVerificationError(
                "artifact is missing or unsafe"
            ) from exc
        return os.fdopen(fd, "rb")

    @staticmethod
    @contextlib.contextmanager
    def _locked(path: Path) -> Iterator[None]:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise WorkspaceArtifactError("artifact lock is unsafe") from exc
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
