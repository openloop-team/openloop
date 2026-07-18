"""Development-only descriptor-safe local OpenHands durable state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import stat
from uuid import UUID

from openloop.broker.models import (
    validate_identifier,
    validate_opaque_ref,
    validate_sha256,
    validate_uuid,
)


class LocalDurableStateProblem(Exception):
    """The development local durable-state boundary failed closed."""

    def __init__(self) -> None:
        super().__init__("local durable state rejected")


def _local_ref(job_id: UUID) -> str:
    validate_uuid("job_id", job_id)
    return f"local-openhands:v1:{job_id}"


def _bounded_identity(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= 2**31 - 1:
        raise ValueError(f"{name} is out of range")
    return value


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise LocalDurableStateProblem()
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )


def _check_directory(descriptor: int, *, uid: int, gid: int) -> None:
    try:
        info = os.fstat(descriptor)
    except OSError as error:
        raise LocalDurableStateProblem() from error
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != uid
        or info.st_gid != gid
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise LocalDurableStateProblem()


def _open_root(path: Path, *, uid: int, gid: int) -> int:
    try:
        descriptor = os.open(path, _directory_flags())
    except OSError as error:
        raise LocalDurableStateProblem() from error
    try:
        _check_directory(descriptor, uid=uid, gid=gid)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _ensure_child(parent: int, name: str, *, uid: int, gid: int) -> int:
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise LocalDurableStateProblem()
    try:
        os.mkdir(name, 0o700, dir_fd=parent)
    except FileExistsError:
        pass
    except OSError as error:
        raise LocalDurableStateProblem() from error
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent)
    except OSError as error:
        raise LocalDurableStateProblem() from error
    try:
        _check_directory(descriptor, uid=uid, gid=gid)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@dataclass(frozen=True, slots=True, repr=False)
class LocalDurableBinding:
    state_root: Path = field(repr=False)
    uid: int
    gid: int

    def __post_init__(self) -> None:
        if not isinstance(self.state_root, Path) or not self.state_root.is_absolute():
            raise ValueError("state_root must be an absolute pathlib.Path")
        _bounded_identity("uid", self.uid)
        _bounded_identity("gid", self.gid)

    def __repr__(self) -> str:
        return (
            "LocalDurableBinding(state_root=<redacted>, "
            f"uid={self.uid}, gid={self.gid})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class DurableStateDescriptor:
    job_id: UUID
    durable_state_ref: str = field(repr=False)
    durable_key_version: str
    durable_digest: str = field(repr=False)

    def __post_init__(self) -> None:
        validate_uuid("job_id", self.job_id)
        validate_opaque_ref("durable_state_ref", self.durable_state_ref)
        validate_identifier("durable_key_version", self.durable_key_version)
        validate_sha256("durable_digest", self.durable_digest)
        if self.durable_state_ref != _local_ref(self.job_id):
            raise ValueError("durable_state_ref does not match job identity")

    def __repr__(self) -> str:
        return (
            "DurableStateDescriptor("
            f"job_id={str(self.job_id)!r}, durable_state_ref=<redacted>, "
            f"durable_key_version={self.durable_key_version!r}, "
            "durable_digest=<redacted>)"
        )


class LocalDurableStateAdapter:
    """Single-node development state; never a production durable authority."""

    development_only = True

    def __init__(self, *, state_root: Path, uid: int, gid: int) -> None:
        if not isinstance(state_root, Path) or not state_root.is_absolute():
            raise LocalDurableStateProblem()
        uid = _bounded_identity("uid", uid)
        gid = _bounded_identity("gid", gid)
        try:
            info = state_root.lstat()
        except OSError as error:
            raise LocalDurableStateProblem() from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise LocalDurableStateProblem()
        resolved = state_root.resolve()
        if resolved == Path(resolved.anchor):
            raise LocalDurableStateProblem()
        descriptor = _open_root(resolved, uid=uid, gid=gid)
        os.close(descriptor)
        self.binding = LocalDurableBinding(resolved, uid, gid)

    @staticmethod
    def reference(job_id: UUID) -> str:
        return _local_ref(job_id)

    def describe(
        self,
        job_id: UUID,
        durable_key_version: str,
        durable_digest: str,
    ) -> DurableStateDescriptor:
        return DurableStateDescriptor(
            job_id=job_id,
            durable_state_ref=self.reference(job_id),
            durable_key_version=durable_key_version,
            durable_digest=durable_digest,
        )

    def _ensure(self, descriptor: DurableStateDescriptor) -> None:
        if not isinstance(descriptor, DurableStateDescriptor):
            raise TypeError("descriptor must be DurableStateDescriptor")
        if descriptor.durable_state_ref != self.reference(descriptor.job_id):
            raise LocalDurableStateProblem()
        root = _open_root(
            self.binding.state_root,
            uid=self.binding.uid,
            gid=self.binding.gid,
        )
        opened = [root]
        try:
            job = _ensure_child(
                root,
                str(descriptor.job_id),
                uid=self.binding.uid,
                gid=self.binding.gid,
            )
            opened.append(job)
            state = _ensure_child(
                job,
                "agent-server",
                uid=self.binding.uid,
                gid=self.binding.gid,
            )
            opened.append(state)
        finally:
            for item in reversed(opened):
                try:
                    os.close(item)
                except OSError:
                    pass

    async def ensure(self, descriptor: DurableStateDescriptor) -> None:
        await asyncio.to_thread(self._ensure, descriptor)


__all__ = [
    "DurableStateDescriptor",
    "LocalDurableBinding",
    "LocalDurableStateAdapter",
    "LocalDurableStateProblem",
]
