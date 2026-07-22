"""Atomic private and cross-process checkpoint-receipt sidecars."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import secrets
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from openloop.broker.models import (
    SignedCheckpointReceipt,
    VerifiedCheckpointReceipt,
)
from openloop.tools.openhands_artifacts import (
    WorkspaceArtifact,
    WorkspaceArtifactIdentity,
    WorkspaceArtifactStore,
)
from openloop.tools.openhands_state import validate_state_identifier

from .receipts import (
    CheckpointReceiptIssuer,
    CheckpointReceiptKey,
    CheckpointReceiptVerifier,
)


_CHECKPOINT_KEY_DOMAIN = b"openloop-checkpoint-key-v1\0"
_ARTIFACT_ID_DOMAIN = b"openloop-checkpoint-artifact-id-v1\0"
_RECEIPT_ID_DOMAIN = b"openloop-checkpoint-receipt-id-v1\0"
_STORE_VERSION = "local-workspace-v1"
_MAX_RECEIPT_BYTES = 16 * 1024
_SHARED_DIRECTORY_MODE = 0o2750
_SHARED_DIRECTORY_MODES = frozenset({_SHARED_DIRECTORY_MODE, 0o750})
_SHARED_RECEIPT_MODE = 0o440
_ORPHANED_TEMP_MIN_AGE_SECONDS = 60.0


class LocalCheckpointReceiptProblem(RuntimeError):
    """Redacted failure at the local checkpoint-store boundary."""

    def __init__(self) -> None:
        super().__init__("local checkpoint receipt operation rejected")


class LocalCheckpointReceiptConflict(LocalCheckpointReceiptProblem):
    """The immutable checkpoint identity already names different evidence."""


def canonical_checkpoint_key_json(key: CheckpointReceiptKey) -> str:
    """Return the stable, path-independent JSON representation of ``key``."""
    if not isinstance(key, CheckpointReceiptKey):
        raise TypeError("key must be CheckpointReceiptKey")
    return json.dumps(
        {
            "barrier_id": key.barrier_id,
            "conversation_id": str(key.conversation_id),
            "generation": key.generation,
            "job_id": str(key.job_id),
            "tenant_id": key.tenant_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def checkpoint_digest(key: CheckpointReceiptKey) -> str:
    encoded = canonical_checkpoint_key_json(key).encode("utf-8")
    return hashlib.sha256(_CHECKPOINT_KEY_DOMAIN + encoded).hexdigest()


def checkpoint_artifact_identity(
    key: CheckpointReceiptKey,
) -> WorkspaceArtifactIdentity:
    digest = checkpoint_digest(key)
    identity = WorkspaceArtifactIdentity(
        job_id=str(key.job_id),
        conversation_id=str(key.conversation_id),
        segment_id=f"g{key.generation}-{digest}",
        kind="checkpoint",
    )
    for name, value in identity.to_dict().items():
        validate_state_identifier(value, field=name)
    return identity


def _dedicated_receipt_relpath(key: CheckpointReceiptKey) -> Path:
    """Return the mountable receipt path shared by publisher and locator."""
    if not isinstance(key, CheckpointReceiptKey):
        raise TypeError("key must be CheckpointReceiptKey")
    return Path(
        _validate_component(key.tenant_id),
        _validate_component(str(key.job_id)),
        LocalCheckpointReceiptStore._filename(key),
    )


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise LocalCheckpointReceiptProblem()
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )


def _validate_component(value: str) -> str:
    try:
        return validate_state_identifier(value, field="checkpoint path component")
    except Exception as error:
        raise LocalCheckpointReceiptProblem() from error


class LocalCheckpointReceiptStore:
    """Trusted publisher and exact locator for immutable local receipt sidecars."""

    def __init__(
        self,
        *,
        artifact_store: WorkspaceArtifactStore,
        issuer: CheckpointReceiptIssuer,
        historical_verifier: CheckpointReceiptVerifier,
        expected_uid: int,
        expected_gid: int,
        receipt_root: Path | None = None,
        shared_gid: int | None = None,
    ) -> None:
        if not isinstance(artifact_store, WorkspaceArtifactStore):
            raise TypeError("artifact_store must be WorkspaceArtifactStore")
        if not isinstance(issuer, CheckpointReceiptIssuer):
            raise TypeError("issuer must be CheckpointReceiptIssuer")
        if not isinstance(historical_verifier, CheckpointReceiptVerifier):
            raise TypeError(
                "historical_verifier must be CheckpointReceiptVerifier"
            )
        expected_ownership = (
            ("expected_uid", expected_uid),
            ("expected_gid", expected_gid),
        )
        if shared_gid is not None:
            expected_ownership += (("shared_gid", shared_gid),)
        for name, value in expected_ownership:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if receipt_root is not None and not isinstance(receipt_root, Path):
            raise TypeError("receipt_root must be Path or None")
        if receipt_root is not None and shared_gid is None:
            raise ValueError("shared_gid is required when receipt_root is set")
        self._artifacts = artifact_store
        self._issuer = issuer
        self._historical_verifier = historical_verifier
        self._uid = expected_uid
        self._gid = expected_gid
        self._receipt_root = receipt_root
        self._shared_gid = shared_gid

    async def publish(
        self, key: CheckpointReceiptKey, descriptor: WorkspaceArtifact
    ) -> SignedCheckpointReceipt:
        if not isinstance(key, CheckpointReceiptKey):
            raise TypeError("key must be CheckpointReceiptKey")
        if not isinstance(descriptor, WorkspaceArtifact):
            raise TypeError("descriptor must be WorkspaceArtifact")
        return await asyncio.to_thread(self._publish, key, descriptor)

    async def lookup(
        self, key: CheckpointReceiptKey
    ) -> SignedCheckpointReceipt | None:
        if not isinstance(key, CheckpointReceiptKey):
            raise TypeError("key must be CheckpointReceiptKey")
        return await asyncio.to_thread(self._lookup, key)

    def _publish(
        self, key: CheckpointReceiptKey, descriptor: WorkspaceArtifact
    ) -> SignedCheckpointReceipt:
        try:
            identity = checkpoint_artifact_identity(key)
            if descriptor.identity != identity:
                raise LocalCheckpointReceiptConflict()
            with self._artifacts.open_verified(descriptor, identity) as verified:
                manifest = verified.manifest
            if manifest.plaintext_sha256 is None:
                raise LocalCheckpointReceiptProblem()

            artifact_id = hashlib.sha256(
                _ARTIFACT_ID_DOMAIN + descriptor.key.encode("utf-8")
            ).hexdigest()
            receipt_material = json.dumps(
                {
                    "artifact_id": artifact_id,
                    "base_commit": manifest.base_commit,
                    "byte_count": descriptor.ciphertext_bytes,
                    "checkpoint": json.loads(canonical_checkpoint_key_json(key)),
                    "ciphertext_sha256": descriptor.ciphertext_sha256,
                    "envelope_version": descriptor.envelope_version,
                    "key_version": descriptor.master_key_id,
                    "plaintext_sha256": manifest.plaintext_sha256,
                    "store_version": _STORE_VERSION,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            receipt = VerifiedCheckpointReceipt(
                issuer=self._issuer.issuer,
                receipt_id=hashlib.sha256(
                    _RECEIPT_ID_DOMAIN + receipt_material
                ).hexdigest(),
                tenant_id=key.tenant_id,
                job_id=key.job_id,
                conversation_id=key.conversation_id,
                generation=key.generation,
                barrier_id=key.barrier_id,
                artifact_id=artifact_id,
                base_commit=manifest.base_commit,
                ciphertext_sha256=descriptor.ciphertext_sha256,
                plaintext_sha256=manifest.plaintext_sha256,
                byte_count=descriptor.ciphertext_bytes,
                store_version=_STORE_VERSION,
                envelope_version=f"workspace-envelope-v{descriptor.envelope_version}",
                key_version=descriptor.master_key_id,
                durable_write_sequence=key.generation,
            )
            signed = self._issuer.issue(receipt)
            return self._publish_sidecar(key, signed, receipt)
        except LocalCheckpointReceiptProblem:
            raise
        except Exception as error:
            raise LocalCheckpointReceiptProblem() from error

    def _lookup(self, key: CheckpointReceiptKey) -> SignedCheckpointReceipt | None:
        try:
            with self._receipt_directory(key, create=False) as directory_fd:
                if directory_fd is None:
                    return None
                return self._read_sidecar(directory_fd, self._filename(key))
        except FileNotFoundError:
            return None
        except LocalCheckpointReceiptProblem:
            raise
        except Exception as error:
            raise LocalCheckpointReceiptProblem() from error

    def _publish_sidecar(
        self,
        key: CheckpointReceiptKey,
        signed: SignedCheckpointReceipt,
        expected: VerifiedCheckpointReceipt,
    ) -> SignedCheckpointReceipt:
        payload = signed.value.encode("ascii")
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise LocalCheckpointReceiptProblem()
        stored = self._publish_artifact_sidecar(key, signed, expected, payload)
        if self._receipt_root is not None:
            self._publish_dedicated_sidecar(key, stored)
        return stored

    def _publish_artifact_sidecar(
        self,
        key: CheckpointReceiptKey,
        signed: SignedCheckpointReceipt,
        expected: VerifiedCheckpointReceipt,
        payload: bytes,
    ) -> SignedCheckpointReceipt:
        with self._receipt_directory(key, create=True) as directory_fd:
            if directory_fd is None:
                raise LocalCheckpointReceiptProblem()
            lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            lock_flags |= getattr(os, "O_NOFOLLOW", 0)
            lock_fd = os.open(
                ".checkpoint-receipts.lock",
                lock_flags,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                self._validate_file(lock_fd, mode=0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                name = self._filename(key)
                existing = self._read_sidecar(directory_fd, name)
                if existing is not None:
                    if self._historical_verifier.verify(existing) != expected:
                        raise LocalCheckpointReceiptConflict()
                    return existing

                # The exclusive flock means any temp file here was orphaned by
                # a crashed publish; sweep them all, not just this PID's name.
                for entry in os.listdir(directory_fd):
                    if entry.startswith(".") and entry.endswith(".tmp"):
                        try:
                            os.unlink(entry, dir_fd=directory_fd)
                        except FileNotFoundError:
                            pass
                temporary = f".{name}.{os.getpid()}.tmp"
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                )
                temporary_fd = os.open(temporary, flags, 0o400, dir_fd=directory_fd)
                try:
                    os.fchmod(temporary_fd, 0o400)
                    view = memoryview(payload)
                    while view:
                        written = os.write(temporary_fd, view)
                        if written <= 0:
                            raise LocalCheckpointReceiptProblem()
                        view = view[written:]
                    os.fsync(temporary_fd)
                    self._validate_file(temporary_fd, mode=0o400)
                finally:
                    os.close(temporary_fd)
                try:
                    os.replace(
                        temporary,
                        name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                    os.fsync(directory_fd)
                finally:
                    try:
                        os.unlink(temporary, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
                return signed
            finally:
                os.close(lock_fd)

    def _publish_dedicated_sidecar(
        self, key: CheckpointReceiptKey, signed: SignedCheckpointReceipt
    ) -> None:
        if self._receipt_root is None or self._shared_gid is None:
            raise LocalCheckpointReceiptProblem()
        payload = signed.value.encode("ascii")
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise LocalCheckpointReceiptProblem()
        relative = _dedicated_receipt_relpath(key)
        opened: list[int] = []
        temporary: str | None = None
        try:
            current = os.open(self._receipt_root, _directory_flags())
            opened.append(current)
            self._validate_shared_directory(current, expected_uid=os.getuid())
            for component in relative.parts[:-1]:
                current = self._open_shared_child(current, component)
                opened.append(current)

            name = relative.name
            self._sweep_orphaned_temporaries(current)
            if self._shared_sidecar_matches(current, name, payload):
                return
            temporary = (
                f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
            )
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            descriptor = os.open(temporary, flags, 0o400, dir_fd=current)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                os.fchown(descriptor, -1, self._shared_gid)
                os.fchmod(descriptor, _SHARED_RECEIPT_MODE)
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise LocalCheckpointReceiptProblem()
                    view = view[written:]
                os.fsync(descriptor)
                self._validate_shared_file(descriptor, expected_uid=os.getuid())
                os.replace(temporary, name, src_dir_fd=current, dst_dir_fd=current)
                temporary = None
                os.fsync(current)
            finally:
                os.close(descriptor)
        except LocalCheckpointReceiptProblem:
            raise
        except Exception as error:
            raise LocalCheckpointReceiptProblem() from error
        finally:
            if temporary is not None and opened:
                try:
                    os.unlink(temporary, dir_fd=opened[-1])
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            for descriptor in reversed(opened):
                os.close(descriptor)

    def _open_shared_child(self, parent_fd: int, name: str) -> int:
        if self._shared_gid is None:
            raise LocalCheckpointReceiptProblem()
        _validate_component(name)
        created = False
        try:
            os.mkdir(name, 0o750, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as error:
            raise LocalCheckpointReceiptProblem() from error
        try:
            descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
        except OSError as error:
            raise LocalCheckpointReceiptProblem() from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise LocalCheckpointReceiptProblem()
            os.fchown(descriptor, -1, self._shared_gid)
            os.fchmod(descriptor, _SHARED_DIRECTORY_MODE)
            self._validate_shared_directory(descriptor, expected_uid=os.getuid())
            os.fsync(descriptor)
            if created:
                os.fsync(parent_fd)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def _shared_sidecar_matches(
        self, directory_fd: int, name: str, expected: bytes
    ) -> bool:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except OSError:
            return False
        try:
            self._validate_shared_file(
                descriptor,
                expected_uid=os.getuid(),
                allow_unlinked=True,
            )
            chunks: list[bytes] = []
            remaining = _MAX_RECEIPT_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            return len(payload) <= _MAX_RECEIPT_BYTES and payload == expected
        except Exception:
            return False
        finally:
            os.close(descriptor)

    @staticmethod
    def _sweep_orphaned_temporaries(directory_fd: int) -> None:
        now = time.time()
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for entry in os.listdir(directory_fd):
            if not entry.startswith(".") or not entry.endswith(".tmp"):
                continue
            try:
                descriptor = os.open(entry, flags, dir_fd=directory_fd)
            except OSError:
                continue
            try:
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_nlink != 1
                    or now - info.st_mtime < _ORPHANED_TEMP_MIN_AGE_SECONDS
                ):
                    continue
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                current = os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)
                locked = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(current.st_mode)
                    or (current.st_dev, current.st_ino)
                    != (locked.st_dev, locked.st_ino)
                ):
                    continue
                try:
                    os.unlink(entry, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            finally:
                os.close(descriptor)

    def _validate_shared_directory(
        self, descriptor: int, *, expected_uid: int
    ) -> None:
        if self._shared_gid is None:
            raise LocalCheckpointReceiptProblem()
        info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != expected_uid
            or info.st_gid != self._shared_gid
            or stat.S_IMODE(info.st_mode) not in _SHARED_DIRECTORY_MODES
        ):
            raise LocalCheckpointReceiptProblem()

    def _validate_shared_file(
        self,
        descriptor: int,
        *,
        expected_uid: int,
        allow_unlinked: bool = False,
    ) -> None:
        if self._shared_gid is None:
            raise LocalCheckpointReceiptProblem()
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != expected_uid
            or info.st_gid != self._shared_gid
            or stat.S_IMODE(info.st_mode) != _SHARED_RECEIPT_MODE
            or info.st_nlink not in ({0, 1} if allow_unlinked else {1})
        ):
            raise LocalCheckpointReceiptProblem()

    @staticmethod
    def _filename(key: CheckpointReceiptKey) -> str:
        return f"v1-{checkpoint_digest(key)}.checkpoint-receipt.jwt"

    @contextmanager
    def _receipt_directory(
        self, key: CheckpointReceiptKey, *, create: bool
    ) -> Iterator[int | None]:
        components = (
            _validate_component(str(key.job_id)),
            "artifacts",
            _validate_component(str(key.conversation_id)),
            "receipts",
        )
        opened: list[int] = []
        try:
            current = os.open(self._artifacts.layout.jobs_root, _directory_flags())
            opened.append(current)
            self._validate_directory(current)
            for index, component in enumerate(components):
                try:
                    child = os.open(component, _directory_flags(), dir_fd=current)
                except FileNotFoundError:
                    if not create or index != len(components) - 1:
                        yield None
                        return
                    os.mkdir(component, 0o700, dir_fd=current)
                    os.fsync(current)
                    child = os.open(component, _directory_flags(), dir_fd=current)
                opened.append(child)
                self._validate_directory(child)
                current = child
            yield current
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    def _validate_directory(self, descriptor: int) -> None:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != self._uid
            or info.st_gid != self._gid
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise LocalCheckpointReceiptProblem()

    def _validate_file(self, descriptor: int, *, mode: int) -> None:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != self._uid
            or info.st_gid != self._gid
            or stat.S_IMODE(info.st_mode) != mode
            or info.st_nlink != 1
        ):
            raise LocalCheckpointReceiptProblem()

    def _read_sidecar(
        self, directory_fd: int, name: str
    ) -> SignedCheckpointReceipt | None:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise LocalCheckpointReceiptProblem() from error
        try:
            self._validate_file(descriptor, mode=0o400)
            chunks: list[bytes] = []
            remaining = _MAX_RECEIPT_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > _MAX_RECEIPT_BYTES:
                raise LocalCheckpointReceiptProblem()
            try:
                value = payload.decode("ascii")
            except UnicodeDecodeError as error:
                raise LocalCheckpointReceiptProblem() from error
            return SignedCheckpointReceipt(value)
        except LocalCheckpointReceiptProblem:
            raise
        except Exception as error:
            raise LocalCheckpointReceiptProblem() from error
        finally:
            os.close(descriptor)


class ReadOnlyCheckpointReceiptLocator:
    """Broker-side receipt locator over the dedicated shared receipt tree."""

    def __init__(
        self,
        *,
        root: Path,
        verifier: CheckpointReceiptVerifier,
        expected_uid: int,
        expected_gid: int,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be Path")
        if not isinstance(verifier, CheckpointReceiptVerifier):
            raise TypeError("verifier must be CheckpointReceiptVerifier")
        for name, value in (
            ("expected_uid", expected_uid),
            ("expected_gid", expected_gid),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        self._root = root
        self._verifier = verifier
        self._uid = expected_uid
        self._gid = expected_gid

    async def lookup(
        self, key: CheckpointReceiptKey
    ) -> SignedCheckpointReceipt | None:
        if not isinstance(key, CheckpointReceiptKey):
            raise TypeError("key must be CheckpointReceiptKey")
        return await asyncio.to_thread(self._lookup, key)

    def _lookup(self, key: CheckpointReceiptKey) -> SignedCheckpointReceipt | None:
        relative = _dedicated_receipt_relpath(key)
        opened: list[int] = []
        try:
            try:
                current = os.open(self._root, _directory_flags())
            except FileNotFoundError as error:
                # The configured mount itself is an availability boundary, not
                # evidence that one particular checkpoint is absent.
                raise LocalCheckpointReceiptProblem() from error
            opened.append(current)
            self._validate_directory(current)
            for component in relative.parts[:-1]:
                try:
                    current = os.open(component, _directory_flags(), dir_fd=current)
                except FileNotFoundError:
                    return None
                opened.append(current)
                self._validate_directory(current)
            signed = self._read_verified(opened[-1], relative.name)
            return signed
        except LocalCheckpointReceiptProblem:
            raise
        except Exception as error:
            raise LocalCheckpointReceiptProblem() from error
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    def _validate_directory(self, descriptor: int) -> None:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != self._uid
            or info.st_gid != self._gid
            or stat.S_IMODE(info.st_mode) not in _SHARED_DIRECTORY_MODES
        ):
            raise LocalCheckpointReceiptProblem()

    def _read_verified(
        self, directory_fd: int, name: str
    ) -> SignedCheckpointReceipt | None:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise LocalCheckpointReceiptProblem() from error
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != self._uid
                or info.st_gid != self._gid
                or stat.S_IMODE(info.st_mode) != _SHARED_RECEIPT_MODE
                # A concurrent atomic replacement can unlink this already-open
                # inode. Its descriptor remains immutable, and the signature is
                # verified below, so link counts zero and one are both safe.
                or info.st_nlink not in {0, 1}
            ):
                raise LocalCheckpointReceiptProblem()
            chunks: list[bytes] = []
            remaining = _MAX_RECEIPT_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > _MAX_RECEIPT_BYTES:
                raise LocalCheckpointReceiptProblem()
            try:
                signed = SignedCheckpointReceipt(payload.decode("ascii"))
                self._verifier.verify(signed)
            except Exception as error:
                raise LocalCheckpointReceiptProblem() from error
            return signed
        except LocalCheckpointReceiptProblem:
            raise
        except Exception as error:
            raise LocalCheckpointReceiptProblem() from error
        finally:
            os.close(descriptor)


__all__ = [
    "LocalCheckpointReceiptConflict",
    "LocalCheckpointReceiptProblem",
    "LocalCheckpointReceiptStore",
    "ReadOnlyCheckpointReceiptLocator",
    "canonical_checkpoint_key_json",
    "checkpoint_artifact_identity",
    "checkpoint_digest",
]
