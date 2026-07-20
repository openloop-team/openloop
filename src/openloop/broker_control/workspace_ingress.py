"""Bounded local workspace staging for broker-owned Docker generations.

The control RPC intentionally carries no host paths or large workspace payloads.
For the co-process canary, the unprivileged worker stages a prepared checkout in
this private store under the broker-minted ``(job_id, generation)`` identity.
The Docker runtime consumes that immutable snapshot before it creates any
network or container.

The staging tree is never mounted into a generation.  Copying preserves regular
files, directories, and symlinks without following symlinks; special files are
rejected.  A per-generation consumed marker makes runtime ``ensure`` replay a
no-op after launch, while a crash before the marker causes the still-containerless
workspace to be rebuilt from the immutable snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from openloop.broker.models import POSTGRES_BIGINT_MAX
from openloop.broker_runtime.contract import GenerationRuntimeIdentity


_COPY_CHUNK_BYTES = 1024 * 1024
_MANIFEST = "manifest.json"
_TREE = "tree"
_CONSUMED = "consumed-operation"


class WorkspaceIngressProblem(RuntimeError):
    """A workspace could not be staged or materialized safely."""


@dataclass(frozen=True, slots=True)
class StagedWorkspace:
    job_id: UUID
    generation: int
    sha256: str
    file_count: int
    byte_count: int


class _Bounds:
    def __init__(self, *, max_files: int, max_bytes: int) -> None:
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.file_count = 0
        self.byte_count = 0

    def add(self, size: int) -> None:
        self.file_count += 1
        self.byte_count += size
        if self.file_count > self.max_files or self.byte_count > self.max_bytes:
            raise WorkspaceIngressProblem("workspace seed exceeds fixed bounds")


def _validate_generation(generation: int) -> None:
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or not 1 <= generation <= POSTGRES_BIGINT_MAX
    ):
        raise ValueError("generation must be a positive PostgreSQL BIGINT")


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise WorkspaceIngressProblem("workspace ingress directory is not private")


def _safe_mode(mode: int, *, directory: bool = False) -> int:
    if directory:
        return 0o700
    return 0o700 if mode & 0o111 else 0o600


def _copy_snapshot(
    source: Path,
    destination: Path,
    *,
    max_files: int,
    max_bytes: int,
) -> tuple[str, int, int]:
    """Copy and hash one tree without following links or accepting devices."""
    source_info = source.lstat()
    if not stat.S_ISDIR(source_info.st_mode) or source.is_symlink():
        raise WorkspaceIngressProblem("workspace seed source must be a directory")
    destination.mkdir(mode=0o700)
    bounds = _Bounds(max_files=max_files, max_bytes=max_bytes)
    digest = hashlib.sha256(b"openloop-workspace-seed-v1\0")

    def visit(src: Path, dst: Path, relative: Path) -> None:
        try:
            entries = sorted(os.scandir(src), key=lambda entry: entry.name)
        except OSError as exc:
            raise WorkspaceIngressProblem("workspace seed cannot be enumerated") from exc
        for entry in entries:
            name = entry.name
            if not name or name in {".", ".."} or "\0" in name:
                raise WorkspaceIngressProblem("workspace seed contains an invalid name")
            rel = relative / name
            rendered = rel.as_posix().encode("utf-8")
            src_path = src / name
            dst_path = dst / name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise WorkspaceIngressProblem("workspace seed entry changed") from exc
            mode = info.st_mode
            if stat.S_ISDIR(mode):
                bounds.add(0)
                digest.update(b"d\0" + rendered + b"\0")
                dst_path.mkdir(mode=_safe_mode(mode, directory=True))
                visit(src_path, dst_path, rel)
                continue
            if stat.S_ISLNK(mode):
                try:
                    target = os.readlink(src_path)
                except OSError as exc:
                    raise WorkspaceIngressProblem("workspace symlink changed") from exc
                encoded_target = os.fsencode(target)
                if b"\0" in encoded_target:
                    raise WorkspaceIngressProblem("workspace symlink is invalid")
                bounds.add(len(encoded_target))
                digest.update(
                    b"l\0" + rendered + b"\0" + encoded_target + b"\0"
                )
                os.symlink(target, dst_path)
                continue
            if not stat.S_ISREG(mode) or info.st_nlink != 1:
                raise WorkspaceIngressProblem(
                    "workspace seed contains a special or hard-linked file"
                )
            bounds.add(info.st_size)
            digest.update(
                b"f\0"
                + rendered
                + b"\0"
                + str(_safe_mode(mode)).encode("ascii")
                + b"\0"
                + str(info.st_size).encode("ascii")
                + b"\0"
            )
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                source_fd = os.open(src_path, flags)
                target_fd = os.open(
                    dst_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    _safe_mode(mode),
                )
            except OSError as exc:
                raise WorkspaceIngressProblem("workspace file cannot be copied") from exc
            try:
                copied = 0
                while True:
                    chunk = os.read(source_fd, _COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    copied += len(chunk)
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(target_fd, view)
                        view = view[written:]
                after = os.fstat(source_fd)
                if copied != info.st_size or any(
                    getattr(after, field) != getattr(info, field)
                    for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
                ):
                    raise WorkspaceIngressProblem("workspace file changed while copied")
                os.fsync(target_fd)
            finally:
                os.close(source_fd)
                os.close(target_fd)

    visit(source, destination, Path())
    return digest.hexdigest(), bounds.file_count, bounds.byte_count


def _write_atomic(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, mode)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary_path.unlink(missing_ok=True)


class LocalWorkspaceIngress:
    """Private, deterministic seed store shared by adapter and Docker driver."""

    def __init__(
        self,
        root: Path,
        *,
        max_files: int = 100_000,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("workspace ingress root must be an absolute path")
        if max_files < 1 or max_bytes < 1:
            raise ValueError("workspace ingress bounds must be positive")
        self.root = root
        self.max_files = max_files
        self.max_bytes = max_bytes
        # All generations of one job share a lock so pruning the final seed's
        # parent cannot race staging the next generation.
        self._locks: dict[UUID, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        _private_directory(root)

    def _lock(self, job_id: UUID, generation: int) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, threading.Lock())

    def _generation_root(self, job_id: UUID, generation: int) -> Path:
        if not isinstance(job_id, UUID):
            raise TypeError("job_id must be a UUID")
        _validate_generation(generation)
        return self.root / str(job_id) / str(generation)

    @staticmethod
    def _read_manifest(root: Path) -> StagedWorkspace:
        try:
            raw = json.loads((root / _MANIFEST).read_text(encoding="ascii"))
            return StagedWorkspace(
                job_id=UUID(raw["job_id"]),
                generation=raw["generation"],
                sha256=raw["sha256"],
                file_count=raw["file_count"],
                byte_count=raw["byte_count"],
            )
        except Exception as exc:
            raise WorkspaceIngressProblem("workspace seed manifest is invalid") from exc

    def stage(self, job_id: UUID, generation: int, source: Path) -> StagedWorkspace:
        target = self._generation_root(job_id, generation)
        source_resolved = source.resolve()
        root_resolved = self.root.resolve()
        if source_resolved == root_resolved or root_resolved.is_relative_to(source_resolved):
            raise WorkspaceIngressProblem("workspace seed cannot contain ingress storage")
        with self._lock(job_id, generation):
            _private_directory(target.parent)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{generation}.", dir=target.parent)
            )
            try:
                tree = temporary / _TREE
                sha256, file_count, byte_count = _copy_snapshot(
                    source,
                    tree,
                    max_files=self.max_files,
                    max_bytes=self.max_bytes,
                )
                staged = StagedWorkspace(
                    job_id, generation, sha256, file_count, byte_count
                )
                manifest = json.dumps(
                    {
                        "byte_count": byte_count,
                        "file_count": file_count,
                        "generation": generation,
                        "job_id": str(job_id),
                        "sha256": sha256,
                        "version": 1,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                _write_atomic(temporary / _MANIFEST, manifest)
                if target.exists():
                    existing = self._read_manifest(target)
                    if existing != staged:
                        raise WorkspaceIngressProblem(
                            "workspace seed replay conflicts with existing content"
                        )
                    return existing
                os.replace(temporary, target)
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return staged
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)

    def materialize(
        self, identity: GenerationRuntimeIdentity, destination: Path
    ) -> None:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        target = self._generation_root(identity.job_id, identity.generation)
        with self._lock(identity.job_id, identity.generation):
            staged = self._read_manifest(target)
            consumed = target / _CONSUMED
            if consumed.exists():
                try:
                    operation_id = consumed.read_text(encoding="ascii")
                except OSError as exc:
                    raise WorkspaceIngressProblem(
                        "workspace seed consumed marker is invalid"
                    ) from exc
                if operation_id != str(identity.operation_id):
                    raise WorkspaceIngressProblem(
                        "workspace seed was consumed by another operation"
                    )
                return
            if not destination.is_dir() or destination.is_symlink():
                raise WorkspaceIngressProblem(
                    "runtime workspace destination is not a directory"
                )
            for child in destination.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            destination.rmdir()
            copied = _copy_snapshot(
                target / _TREE,
                destination,
                max_files=self.max_files,
                max_bytes=self.max_bytes,
            )
            if copied != (staged.sha256, staged.file_count, staged.byte_count):
                raise WorkspaceIngressProblem("materialized workspace seed mismatch")
            _write_atomic(consumed, str(identity.operation_id).encode("ascii"))

    def discard(self, identity: GenerationRuntimeIdentity) -> None:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        target = self._generation_root(identity.job_id, identity.generation)
        with self._lock(identity.job_id, identity.generation):
            shutil.rmtree(target, ignore_errors=True)
            try:
                target.parent.rmdir()
            except OSError:
                # Another staged generation still owns the job directory.
                pass
