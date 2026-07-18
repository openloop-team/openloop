"""Fail-closed Ed25519 key loading with atomic verification-key rotation."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import stat
from threading import RLock
from types import MappingProxyType

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from openloop.broker.models import validate_identifier


MAX_KEY_FILE_BYTES = 16 * 1024


class KeyFileProblem(Exception):
    def __init__(self) -> None:
        super().__init__("broker key file rejected")


def _read_key_file(
    path: str | os.PathLike[str],
    *,
    expected_uid: int | None,
    private: bool,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(os.fspath(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise KeyFileProblem()
        if expected_uid is not None and before.st_uid != expected_uid:
            raise KeyFileProblem()
        if private:
            if before.st_mode & 0o077:
                raise KeyFileProblem()
        elif before.st_mode & 0o022:
            raise KeyFileProblem()
        if not 0 < before.st_size <= MAX_KEY_FILE_BYTES:
            raise KeyFileProblem()
        chunks: list[bytes] = []
        remaining = MAX_KEY_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(4096, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_uid,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_uid,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if not stable or len(data) != before.st_size:
            raise KeyFileProblem()
        return data
    except KeyFileProblem:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise KeyFileProblem() from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def load_ed25519_private_key(
    path: str | os.PathLike[str], *, expected_uid: int | None = None
) -> Ed25519PrivateKey:
    data = _read_key_file(path, expected_uid=expected_uid, private=True)
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError) as error:
        raise KeyFileProblem() from error
    if not isinstance(key, Ed25519PrivateKey):
        raise KeyFileProblem()
    return key


def load_private_bytes(
    path: str | os.PathLike[str], *, expected_uid: int | None = None
) -> bytes:
    return _read_key_file(path, expected_uid=expected_uid, private=True)


def load_ed25519_public_key(
    path: str | os.PathLike[str], *, expected_uid: int | None = None
) -> Ed25519PublicKey:
    data = _read_key_file(path, expected_uid=expected_uid, private=False)
    try:
        key = serialization.load_pem_public_key(data)
    except (ValueError, TypeError) as error:
        raise KeyFileProblem() from error
    if not isinstance(key, Ed25519PublicKey):
        raise KeyFileProblem()
    return key


class VerificationKeySet:
    def __init__(self, keys: Mapping[str, Ed25519PublicKey]) -> None:
        self._lock = RLock()
        self._keys = self._validated(keys)

    @staticmethod
    def _validated(
        keys: Mapping[str, Ed25519PublicKey],
    ) -> dict[str, Ed25519PublicKey]:
        if not isinstance(keys, Mapping) or not keys:
            raise KeyFileProblem()
        validated: dict[str, Ed25519PublicKey] = {}
        for key_id, key in keys.items():
            try:
                validate_identifier("key_id", key_id)
            except (TypeError, ValueError) as error:
                raise KeyFileProblem() from error
            if not isinstance(key, Ed25519PublicKey):
                raise KeyFileProblem()
            validated[key_id] = key
        return validated

    @classmethod
    def load(
        cls,
        paths: Mapping[str, str | os.PathLike[str]],
        *,
        expected_uid: int | None = None,
    ) -> "VerificationKeySet":
        keys = {
            key_id: load_ed25519_public_key(path, expected_uid=expected_uid)
            for key_id, path in paths.items()
        }
        return cls(keys)

    def reload(
        self,
        paths: Mapping[str, str | os.PathLike[str]],
        *,
        expected_uid: int | None = None,
    ) -> None:
        replacement = self.load(paths, expected_uid=expected_uid)._keys
        with self._lock:
            self._keys = replacement

    def snapshot(self) -> Mapping[str, Ed25519PublicKey]:
        with self._lock:
            return MappingProxyType(dict(self._keys))
