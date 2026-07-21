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

When the broker runs as a *separate process* (external mode) the app stages and
the broker materializes across a uid boundary.  Three keyword knobs carry that
split without changing the co-process default (all unset → byte-for-byte today):

- ``shared_gid`` (stage side): directories are setgid group-mode ``0o2750`` owned
  by the shared gid so children inherit the group; files ``0o640``/``0o750`` and
  the manifest ``0o440`` — the broker can read them without owning them.
- ``expected_stage_uid`` (materialize side): directory/manifest validation
  compares ``st_uid`` against the app's uid (not ``os.getuid()``) and ``st_gid``
  against ``shared_gid``.
- ``marker_root`` (materialize side): the consumed/discarded markers live in a
  broker-private sibling tree — checked *first*, so an idempotent replay works
  after the producer has pruned the (app-owned) staged tree, and ``discard``
  never deletes app-owned data.

The tree walk is descriptor-anchored in **both** modes: each child is opened
relative to its parent's fd with ``O_NOFOLLOW`` (directories additionally
``O_DIRECTORY``) and re-``fstat``-ed, so a producer that swaps a checked
directory or file for a symlink mid-walk cannot steer the (differently
privileged) consumer outside the staged tree.  Legitimate staged symlinks are
still copied as symlinks (read descriptor-relative, never followed).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from openloop.broker.models import POSTGRES_BIGINT_MAX
from openloop.broker_runtime.contract import GenerationRuntimeIdentity


_COPY_CHUNK_BYTES = 1024 * 1024
_MANIFEST = "manifest.json"
_TREE = "tree"
_CONSUMED = "consumed-operation"
_DISCARDED = "discarded"

# Descriptor-anchored open flags. Directories additionally require O_DIRECTORY so
# a swap to a symlink or file fails the open outright; both refuse to follow a
# final-component symlink (O_NOFOLLOW) and never leak to a child process.
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_DIR_OPEN_FLAGS = _FILE_OPEN_FLAGS | getattr(os, "O_DIRECTORY", 0)


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


def _owned_directory(path: Path, *, mode: int, gid: int | None) -> None:
    """Create + validate a store directory we own.

    Owner-only (``gid`` unset) is today's private ``0o700`` directory, validated
    but never chmod-ed (a wrong pre-existing mode still raises).  Shared mode
    forces the setgid group-handoff bits ``mkdir`` masks off plus the shared gid,
    so children inherit the group across the process boundary.
    """
    path.mkdir(mode=mode & 0o777, parents=True, exist_ok=True)
    if gid is not None:
        os.chown(path, -1, gid)
        os.chmod(path, mode)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != mode
        or (gid is not None and info.st_gid != gid)
    ):
        raise WorkspaceIngressProblem("workspace ingress directory is not private")


def _safe_mode(mode: int, *, directory: bool = False, shared: bool = False) -> int:
    if directory:
        return 0o2750 if shared else 0o700
    if shared:
        return 0o750 if mode & 0o111 else 0o640
    return 0o700 if mode & 0o111 else 0o600


def _copy_snapshot(
    source: Path,
    destination: Path,
    *,
    max_files: int,
    max_bytes: int,
    shared: bool = False,
) -> tuple[str, int, int]:
    """Copy and hash one tree without following links or accepting devices.

    The walk is descriptor-anchored: each child is enumerated and opened relative
    to its parent's fd, and re-``fstat``-ed against the pre-open observation so a
    mid-walk type swap or inode substitution raises.  ``shared`` selects the
    group-handoff write modes for the *destination* only — the content digest is
    computed from the mode-invariant canonical ``_safe_mode`` so a shared staging
    tree and its owner-mode materialization hash identically.
    """
    bounds = _Bounds(max_files=max_files, max_bytes=max_bytes)
    digest = hashlib.sha256(b"openloop-workspace-seed-v1\0")

    def visit(src_fd: int, dst_fd: int, relative: Path) -> None:
        try:
            names = sorted(os.listdir(src_fd))
        except OSError as exc:
            raise WorkspaceIngressProblem("workspace seed cannot be enumerated") from exc
        for name in names:
            if not name or name in {".", ".."} or "\0" in name:
                raise WorkspaceIngressProblem("workspace seed contains an invalid name")
            rel = relative / name
            rendered = rel.as_posix().encode("utf-8")
            try:
                info = os.stat(name, dir_fd=src_fd, follow_symlinks=False)
            except OSError as exc:
                raise WorkspaceIngressProblem("workspace seed entry changed") from exc
            mode = info.st_mode
            if stat.S_ISDIR(mode):
                bounds.add(0)
                digest.update(b"d\0" + rendered + b"\0")
                try:
                    child_src_fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=src_fd)
                except OSError as exc:
                    raise WorkspaceIngressProblem(
                        "workspace seed directory changed to a non-directory"
                    ) from exc
                try:
                    opened = os.fstat(child_src_fd)
                    if not stat.S_ISDIR(opened.st_mode) or (
                        opened.st_dev,
                        opened.st_ino,
                    ) != (info.st_dev, info.st_ino):
                        raise WorkspaceIngressProblem(
                            "workspace seed directory changed between scan and open"
                        )
                    os.mkdir(
                        name,
                        _safe_mode(mode, directory=True, shared=shared),
                        dir_fd=dst_fd,
                    )
                    child_dst_fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=dst_fd)
                    try:
                        if shared:
                            os.fchmod(
                                child_dst_fd,
                                _safe_mode(mode, directory=True, shared=True),
                            )
                        visit(child_src_fd, child_dst_fd, rel)
                    finally:
                        os.close(child_dst_fd)
                finally:
                    os.close(child_src_fd)
                continue
            if stat.S_ISLNK(mode):
                try:
                    target = os.readlink(name, dir_fd=src_fd)
                except OSError as exc:
                    raise WorkspaceIngressProblem("workspace symlink changed") from exc
                encoded_target = os.fsencode(target)
                if b"\0" in encoded_target:
                    raise WorkspaceIngressProblem("workspace symlink is invalid")
                bounds.add(len(encoded_target))
                digest.update(b"l\0" + rendered + b"\0" + encoded_target + b"\0")
                os.symlink(target, name, dir_fd=dst_fd)
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
            try:
                source_fd = os.open(name, _FILE_OPEN_FLAGS, dir_fd=src_fd)
            except OSError as exc:
                raise WorkspaceIngressProblem("workspace file cannot be copied") from exc
            target_fd = -1
            try:
                opened = os.fstat(source_fd)
                if not stat.S_ISREG(opened.st_mode) or (
                    opened.st_dev,
                    opened.st_ino,
                ) != (info.st_dev, info.st_ino):
                    raise WorkspaceIngressProblem(
                        "workspace file changed between scan and open"
                    )
                try:
                    target_fd = os.open(
                        name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        _safe_mode(mode, shared=shared),
                        dir_fd=dst_fd,
                    )
                except OSError as exc:
                    raise WorkspaceIngressProblem(
                        "workspace file cannot be copied"
                    ) from exc
                if shared:
                    os.fchmod(target_fd, _safe_mode(mode, shared=shared))
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
                if target_fd >= 0:
                    os.close(target_fd)

    try:
        root_src_fd = os.open(source, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise WorkspaceIngressProblem(
            "workspace seed source must be a directory"
        ) from exc
    try:
        root_mode = _safe_mode(0, directory=True, shared=shared)
        destination.mkdir(mode=root_mode & 0o777)
        if shared:
            os.chmod(destination, root_mode)
        root_dst_fd = os.open(destination, _DIR_OPEN_FLAGS)
        try:
            visit(root_src_fd, root_dst_fd, Path())
        finally:
            os.close(root_dst_fd)
    finally:
        os.close(root_src_fd)
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
        shared_gid: int | None = None,
        expected_stage_uid: int | None = None,
        marker_root: Path | None = None,
    ) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("workspace ingress root must be an absolute path")
        if max_files < 1 or max_bytes < 1:
            raise ValueError("workspace ingress bounds must be positive")
        if marker_root is not None and (
            not isinstance(marker_root, Path) or not marker_root.is_absolute()
        ):
            raise ValueError("workspace ingress marker root must be an absolute path")
        self.root = root
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.shared_gid = shared_gid
        self.expected_stage_uid = expected_stage_uid
        self.marker_root = marker_root
        # All generations of one job share a lock so pruning the final seed's
        # parent cannot race staging the next generation.
        self._locks: dict[UUID, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        if expected_stage_uid is not None:
            # Materialize side: the app owns the staging root — validate, do not
            # create (the broker cannot chown to the app's uid anyway).
            self._require_stage_ownership(root, directory=True)
        else:
            self._prepare_owned_directory(root)
        if marker_root is not None:
            # Broker-private consumed/discarded marker tree.
            _owned_directory(marker_root, mode=0o700, gid=None)

    def _prepare_owned_directory(self, path: Path) -> None:
        """Create + validate a store directory on the owner (stage) side."""
        if self.shared_gid is not None:
            _owned_directory(path, mode=0o2750, gid=self.shared_gid)
        else:
            _owned_directory(path, mode=0o700, gid=None)

    def _require_stage_ownership(self, path: Path, *, directory: bool) -> None:
        """Materialize side: confirm a staged path is owned by the app identity.

        Active only when ``expected_stage_uid`` is set (external broker).  In the
        co-process default (unset) this is a no-op, preserving today's checks.
        """
        if self.expected_stage_uid is None:
            return
        try:
            info = path.lstat()
        except OSError as exc:
            raise WorkspaceIngressProblem("staged workspace path is missing") from exc
        type_ok = (
            stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        )
        if not type_ok:
            raise WorkspaceIngressProblem(
                "staged workspace path has an unexpected type"
            )
        if info.st_uid != self.expected_stage_uid:
            raise WorkspaceIngressProblem(
                "staged workspace is not owned by the app uid"
            )
        if self.shared_gid is not None and info.st_gid != self.shared_gid:
            raise WorkspaceIngressProblem("staged workspace has an unexpected group")

    def _lock(self, job_id: UUID, generation: int) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, threading.Lock())

    def _generation_root(self, job_id: UUID, generation: int) -> Path:
        if not isinstance(job_id, UUID):
            raise TypeError("job_id must be a UUID")
        _validate_generation(generation)
        return self.root / str(job_id) / str(generation)

    def _marker_gen_dir(self, job_id: UUID, generation: int) -> Path:
        assert self.marker_root is not None
        return self.marker_root / str(job_id) / str(generation)

    def _ensure_marker_gen_dir(self, job_id: UUID, generation: int) -> Path:
        """Broker-private ``marker_root/<job>/<generation>`` (0700 throughout)."""
        assert self.marker_root is not None
        job_dir = self.marker_root / str(job_id)
        job_dir.mkdir(mode=0o700, exist_ok=True)
        gen_dir = job_dir / str(generation)
        gen_dir.mkdir(mode=0o700, exist_ok=True)
        return gen_dir

    def _read_consumed_marker(self, job_id: UUID, generation: int) -> str | None:
        path = self._marker_gen_dir(job_id, generation) / _CONSUMED
        try:
            return path.read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise WorkspaceIngressProblem(
                "workspace seed consumed marker is invalid"
            ) from exc

    def _write_consumed_marker(
        self, job_id: UUID, generation: int, operation_id: str
    ) -> None:
        gen_dir = self._ensure_marker_gen_dir(job_id, generation)
        _write_atomic(gen_dir / _CONSUMED, operation_id.encode("ascii"))

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

    def _read_verified_manifest(self, target: Path) -> StagedWorkspace:
        """Materialize side: confirm the manifest is app-owned, then read it."""
        self._require_stage_ownership(target / _MANIFEST, directory=False)
        return self._read_manifest(target)

    def stage(self, job_id: UUID, generation: int, source: Path) -> StagedWorkspace:
        target = self._generation_root(job_id, generation)
        source_resolved = source.resolve()
        root_resolved = self.root.resolve()
        if source_resolved == root_resolved or root_resolved.is_relative_to(source_resolved):
            raise WorkspaceIngressProblem("workspace seed cannot contain ingress storage")
        shared = self.shared_gid is not None
        with self._lock(job_id, generation):
            self._prepare_owned_directory(target.parent)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{generation}.", dir=target.parent)
            )
            try:
                if shared:
                    # Group-handoff the temp generation root BEFORE populating and
                    # renaming, or the renamed generation root (mkdtemp is always
                    # 0700) would fail broker-side ownership validation.
                    os.chown(temporary, -1, self.shared_gid)
                    os.chmod(temporary, 0o2750)
                tree = temporary / _TREE
                sha256, file_count, byte_count = _copy_snapshot(
                    source,
                    tree,
                    max_files=self.max_files,
                    max_bytes=self.max_bytes,
                    shared=shared,
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
                _write_atomic(
                    temporary / _MANIFEST,
                    manifest,
                    mode=0o440 if shared else 0o400,
                )
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

    def _materialize_tree(
        self, target: Path, destination: Path, staged: StagedWorkspace
    ) -> None:
        self._require_stage_ownership(target, directory=True)
        self._require_stage_ownership(target / _TREE, directory=True)
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

    def materialize(
        self, identity: GenerationRuntimeIdentity, destination: Path
    ) -> None:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        target = self._generation_root(identity.job_id, identity.generation)
        operation_id = str(identity.operation_id)
        with self._lock(identity.job_id, identity.generation):
            if self.marker_root is not None:
                # Marker-first: an idempotent replay short-circuits with no
                # manifest, which survives the producer pruning the staged tree.
                recorded = self._read_consumed_marker(
                    identity.job_id, identity.generation
                )
                if recorded is not None:
                    if recorded != operation_id:
                        raise WorkspaceIngressProblem(
                            "workspace seed was consumed by another operation"
                        )
                    return
                staged = self._read_verified_manifest(target)
                self._materialize_tree(target, destination, staged)
                self._write_consumed_marker(
                    identity.job_id, identity.generation, operation_id
                )
                return
            # Default (co-process): manifest-then-marker, in-tree marker.
            staged = self._read_verified_manifest(target)
            consumed = target / _CONSUMED
            if consumed.exists():
                try:
                    recorded = consumed.read_text(encoding="ascii")
                except OSError as exc:
                    raise WorkspaceIngressProblem(
                        "workspace seed consumed marker is invalid"
                    ) from exc
                if recorded != operation_id:
                    raise WorkspaceIngressProblem(
                        "workspace seed was consumed by another operation"
                    )
                return
            self._materialize_tree(target, destination, staged)
            _write_atomic(consumed, operation_id.encode("ascii"))

    def discard(self, identity: GenerationRuntimeIdentity) -> None:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        target = self._generation_root(identity.job_id, identity.generation)
        with self._lock(identity.job_id, identity.generation):
            if self.marker_root is not None:
                # Marker-only: the producer owns and deletes the tree; the
                # consumer records intent and never rmtrees app-owned data.
                gen_dir = self._ensure_marker_gen_dir(
                    identity.job_id, identity.generation
                )
                _write_atomic(gen_dir / _DISCARDED, b"")
                return
            shutil.rmtree(target, ignore_errors=True)
            try:
                target.parent.rmdir()
            except OSError:
                # Another staged generation still owns the job directory.
                pass

    def prune(self, job_id: UUID, generation: int) -> None:
        """Producer-side deletion of one staged generation (+ empty parent)."""
        target = self._generation_root(job_id, generation)
        with self._lock(job_id, generation):
            shutil.rmtree(target, ignore_errors=True)
            try:
                target.parent.rmdir()
            except OSError:
                pass

    def prune_stale(self, *, max_age_seconds: int) -> int:
        """Reap staged generations whose manifest is older than the cutoff.

        Defensive by design: entries that vanish concurrently are skipped, never
        raised on.  Returns the number of generations removed.
        """
        cutoff = time.time() - max_age_seconds
        removed = 0
        try:
            with os.scandir(self.root) as scan:
                job_entries = list(scan)
        except OSError:
            return removed
        for job_entry in job_entries:
            try:
                job_id = UUID(job_entry.name)
            except ValueError:
                continue
            with self._lock(job_id, 0):
                try:
                    with os.scandir(job_entry.path) as scan:
                        gen_entries = list(scan)
                except OSError:
                    continue
                for gen_entry in gen_entries:
                    manifest = Path(gen_entry.path) / _MANIFEST
                    try:
                        mtime = manifest.stat().st_mtime
                    except OSError:
                        continue
                    if mtime < cutoff:
                        shutil.rmtree(gen_entry.path, ignore_errors=True)
                        removed += 1
                try:
                    os.rmdir(job_entry.path)
                except OSError:
                    pass
        return removed
