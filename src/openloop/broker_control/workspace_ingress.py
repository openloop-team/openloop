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

import errno
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


def _temporary_generation(name: str) -> int | None:
    """Return the generation encoded by ``stage``'s mkdtemp name shape."""
    if not name.startswith("."):
        return None
    generation_text, separator, suffix = name[1:].partition(".")
    if (
        not separator
        or not suffix
        or not suffix.isascii()
        or any(
            not (character.isalnum() or character == "_")
            for character in suffix
        )
    ):
        return None
    try:
        generation = int(generation_text)
        _validate_generation(generation)
    except (TypeError, ValueError):
        return None
    # stage() renders the integer directly, so aliases such as `.01.*` are not
    # directories it can create and must not be treated as cleanup targets.
    return generation if generation_text == str(generation) else None


def _safe_component(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise WorkspaceIngressProblem("workspace ingress path component is invalid")


def _check_directory_descriptor(
    descriptor: int,
    *,
    label: str,
    uid: int,
    gid: int | None,
    mode: int,
) -> os.stat_result:
    try:
        info = os.fstat(descriptor)
    except OSError as exc:
        raise WorkspaceIngressProblem(
            f"{label} directory cannot be inspected"
        ) from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != uid
        or stat.S_IMODE(info.st_mode) != mode
        or (gid is not None and info.st_gid != gid)
    ):
        raise WorkspaceIngressProblem(f"{label} directory metadata is invalid")
    return info


def _open_validated_directory(
    path: str | Path,
    *,
    label: str,
    uid: int,
    gid: int | None,
    mode: int,
    dir_fd: int | None = None,
) -> int:
    try:
        if dir_fd is None:
            descriptor = os.open(path, _DIR_OPEN_FLAGS)
        else:
            descriptor = os.open(path, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        raise WorkspaceIngressProblem(
            f"{label} path is not a safe directory"
        ) from exc
    try:
        _check_directory_descriptor(
            descriptor, label=label, uid=uid, gid=gid, mode=mode
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_owned_directory(
    path: str | Path,
    *,
    label: str,
    uid: int,
    device: int,
    dir_fd: int | None = None,
) -> int:
    """Open an app-owned directory without interpreting mode as lifecycle state."""
    try:
        if dir_fd is None:
            descriptor = os.open(path, _DIR_OPEN_FLAGS)
        else:
            descriptor = os.open(path, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        raise WorkspaceIngressProblem(f"{label} path is not a safe directory") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != uid
            or info.st_dev != device
        ):
            raise WorkspaceIngressProblem(f"{label} directory metadata is invalid")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _check_regular_metadata(
    info: os.stat_result,
    *,
    label: str,
    uid: int,
    gid: int | None,
    mode: int,
) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != uid
        or stat.S_IMODE(info.st_mode) != mode
        or (gid is not None and info.st_gid != gid)
    ):
        raise WorkspaceIngressProblem(f"{label} file metadata is invalid")


def _owned_directory(path: Path, *, mode: int, gid: int | None) -> None:
    """Create + validate a store directory we own.

    Owner-only (``gid`` unset) is today's private ``0o700`` directory, validated
    but never chmod-ed (a wrong pre-existing mode still raises).  Shared mode
    forces the setgid group-handoff bits ``mkdir`` masks off plus the shared gid,
    so children inherit the group across the process boundary.
    """
    try:
        path.mkdir(mode=mode & 0o777, parents=True, exist_ok=True)
        descriptor = os.open(path, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise WorkspaceIngressProblem(
            "workspace ingress directory is not safely accessible"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.getuid():
            raise WorkspaceIngressProblem(
                "workspace ingress directory is not privately owned"
            )
        if gid is not None:
            os.fchown(descriptor, -1, gid)
            os.fchmod(descriptor, mode)
        _check_directory_descriptor(
            descriptor,
            label="workspace ingress",
            uid=os.getuid(),
            gid=gid,
            mode=mode,
        )
    except WorkspaceIngressProblem:
        raise
    except OSError as exc:
        raise WorkspaceIngressProblem(
            "workspace ingress directory permissions cannot be applied"
        ) from exc
    finally:
        os.close(descriptor)


def _ensure_owned_child(
    parent_fd: int,
    name: str,
    *,
    mode: int,
    gid: int | None,
    label: str,
) -> int:
    """Create/open one owned child without following the final component."""
    _safe_component(name)
    try:
        os.mkdir(name, mode & 0o777, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise WorkspaceIngressProblem(f"{label} directory cannot be created") from exc
    try:
        descriptor = os.open(name, _DIR_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise WorkspaceIngressProblem(f"{label} path is not a safe directory") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.getuid():
            raise WorkspaceIngressProblem(f"{label} directory is not privately owned")
        if gid is not None:
            os.fchown(descriptor, -1, gid)
            os.fchmod(descriptor, mode)
        _check_directory_descriptor(
            descriptor,
            label=label,
            uid=os.getuid(),
            gid=gid,
            mode=mode,
        )
    except WorkspaceIngressProblem:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise WorkspaceIngressProblem(
            f"{label} directory permissions cannot be applied"
        ) from exc
    return descriptor


def _remove_tree_at(
    parent_fd: int,
    name: str,
    *,
    uid: int,
    gid: int | None,
    mode: int,
    missing_ok: bool = False,
    expected_identity: tuple[int, int] | None = None,
) -> bool:
    """Remove one tree relative to a trusted parent without following links."""
    _safe_component(name)
    try:
        descriptor = _open_validated_directory(
            name,
            dir_fd=parent_fd,
            label="workspace ingress cleanup",
            uid=uid,
            gid=gid,
            mode=mode,
        )
    except WorkspaceIngressProblem:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return False
        except OSError:
            pass
        raise
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise WorkspaceIngressProblem(
            "workspace ingress cleanup target cannot be inspected"
        ) from exc
    if expected_identity is not None and (
        before.st_dev,
        before.st_ino,
    ) != expected_identity:
        os.close(descriptor)
        raise WorkspaceIngressProblem(
            "workspace ingress cleanup target changed before removal"
        )
    try:
        try:
            entries = os.listdir(descriptor)
        except OSError as exc:
            raise WorkspaceIngressProblem(
                "workspace ingress cleanup cannot enumerate a directory"
            ) from exc
        for entry in entries:
            _safe_component(entry)
            try:
                info = os.stat(entry, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise WorkspaceIngressProblem(
                    "workspace ingress cleanup cannot inspect an entry"
                ) from exc
            if stat.S_ISDIR(info.st_mode):
                _remove_tree_at(
                    descriptor,
                    entry,
                    uid=uid,
                    gid=gid,
                    mode=mode,
                    missing_ok=True,
                )
            else:
                try:
                    os.unlink(entry, dir_fd=descriptor)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise WorkspaceIngressProblem(
                        "workspace ingress cleanup cannot unlink an entry"
                    ) from exc
    finally:
        os.close(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise WorkspaceIngressProblem(
                "workspace ingress cleanup target changed during removal"
            )
        os.rmdir(name, dir_fd=parent_fd)
    except WorkspaceIngressProblem:
        raise
    except FileNotFoundError:
        if missing_ok:
            return False
        raise WorkspaceIngressProblem("workspace ingress cleanup target vanished")
    except OSError as exc:
        raise WorkspaceIngressProblem(
            "workspace ingress cleanup cannot remove a directory"
        ) from exc
    return True


def _remove_owned_tree_at(
    parent_fd: int,
    name: str,
    *,
    uid: int,
    device: int,
    missing_ok: bool = False,
    expected_identity: tuple[int, int] | None = None,
) -> bool:
    """Remove one temp tree using ownership/confinement, independent of modes."""
    _safe_component(name)
    try:
        descriptor = _open_owned_directory(
            name,
            dir_fd=parent_fd,
            label="workspace ingress temp cleanup",
            uid=uid,
            device=device,
        )
    except WorkspaceIngressProblem:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return False
        except OSError:
            pass
        raise
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise WorkspaceIngressProblem(
            "workspace ingress temp cleanup target cannot be inspected"
        ) from exc
    if expected_identity is not None and (
        before.st_dev,
        before.st_ino,
    ) != expected_identity:
        os.close(descriptor)
        raise WorkspaceIngressProblem(
            "workspace ingress temp cleanup target changed before removal"
        )
    try:
        try:
            entries = os.listdir(descriptor)
        except OSError as exc:
            raise WorkspaceIngressProblem(
                "workspace ingress temp cleanup cannot enumerate a directory"
            ) from exc
        for entry in entries:
            _safe_component(entry)
            try:
                info = os.stat(entry, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise WorkspaceIngressProblem(
                    "workspace ingress temp cleanup cannot inspect an entry"
                ) from exc
            if info.st_uid != uid or info.st_dev != device:
                raise WorkspaceIngressProblem(
                    "workspace ingress temp cleanup entry metadata is invalid"
                )
            if stat.S_ISDIR(info.st_mode):
                _remove_owned_tree_at(
                    descriptor,
                    entry,
                    uid=uid,
                    device=device,
                    missing_ok=True,
                    expected_identity=(info.st_dev, info.st_ino),
                )
            else:
                try:
                    os.unlink(entry, dir_fd=descriptor)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise WorkspaceIngressProblem(
                        "workspace ingress temp cleanup cannot unlink an entry"
                    ) from exc
    finally:
        os.close(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_uid != uid
            or current.st_dev != device
            or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise WorkspaceIngressProblem(
                "workspace ingress temp cleanup target changed during removal"
            )
        os.rmdir(name, dir_fd=parent_fd)
    except WorkspaceIngressProblem:
        raise
    except FileNotFoundError:
        if missing_ok:
            return False
        raise WorkspaceIngressProblem(
            "workspace ingress temp cleanup target vanished"
        )
    except OSError as exc:
        raise WorkspaceIngressProblem(
            "workspace ingress temp cleanup cannot remove a directory"
        ) from exc
    return True


def _remove_empty_open_directory(
    parent_fd: int,
    name: str,
    descriptor: int,
) -> bool:
    """Remove an already-open child if it is still the same empty directory."""
    _safe_component(name)
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise WorkspaceIngressProblem(
            "workspace ingress directory cannot be inspected"
        ) from exc
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise WorkspaceIngressProblem(
                "workspace ingress directory changed before removal"
            )
        os.rmdir(name, dir_fd=parent_fd)
    except WorkspaceIngressProblem:
        raise
    except FileNotFoundError:
        return False
    except OSError as exc:
        if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
            return False
        raise WorkspaceIngressProblem(
            "workspace ingress directory cannot be removed"
        ) from exc
    return True


def _safe_mode(mode: int, *, directory: bool = False, shared: bool = False) -> int:
    if directory:
        return 0o2750 if shared else 0o700
    if shared:
        return 0o750 if mode & 0o111 else 0o640
    return 0o700 if mode & 0o111 else 0o600


def _copy_snapshot(
    source: Path | int,
    destination: Path,
    *,
    max_files: int,
    max_bytes: int,
    shared: bool = False,
    expected_source_uid: int | None = None,
    expected_source_gid: int | None = None,
    source_shared: bool = False,
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

    def validate_source(info: os.stat_result, *, entry_type: str) -> None:
        if expected_source_uid is None:
            return
        if info.st_uid != expected_source_uid or (
            expected_source_gid is not None and info.st_gid != expected_source_gid
        ):
            raise WorkspaceIngressProblem(
                "staged workspace entry has unexpected ownership"
            )
        if entry_type == "directory":
            expected_mode = _safe_mode(0, directory=True, shared=source_shared)
        elif entry_type == "file":
            expected_mode = _safe_mode(info.st_mode, shared=source_shared)
        else:
            return
        if stat.S_IMODE(info.st_mode) != expected_mode:
            raise WorkspaceIngressProblem(
                "staged workspace entry has unsafe permissions"
            )

    def visit(src_fd: int, dst_fd: int, relative: Path) -> None:
        try:
            names = sorted(os.listdir(src_fd))
        except OSError as exc:
            raise WorkspaceIngressProblem(
                "workspace seed cannot be enumerated"
            ) from exc
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
                validate_source(info, entry_type="directory")
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
                    validate_source(opened, entry_type="directory")
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
                validate_source(info, entry_type="symlink")
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
            validate_source(info, entry_type="file")
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
                raise WorkspaceIngressProblem(
                    "workspace file cannot be copied"
                ) from exc
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
                validate_source(opened, entry_type="file")
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
                    for field in (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_nlink",
                        "st_uid",
                        "st_gid",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                ):
                    raise WorkspaceIngressProblem("workspace file changed while copied")
                os.fsync(target_fd)
            finally:
                os.close(source_fd)
                if target_fd >= 0:
                    os.close(target_fd)

    try:
        root_src_fd = os.dup(source) if isinstance(source, int) else os.open(
            source, _DIR_OPEN_FLAGS
        )
    except OSError as exc:
        raise WorkspaceIngressProblem(
            "workspace seed source must be a directory"
        ) from exc
    try:
        root_info = os.fstat(root_src_fd)
        if not stat.S_ISDIR(root_info.st_mode):
            raise WorkspaceIngressProblem(
                "workspace seed source must be a directory"
            )
        validate_source(root_info, entry_type="directory")
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

    @property
    def _stage_uid(self) -> int:
        return (
            self.expected_stage_uid
            if self.expected_stage_uid is not None
            else os.getuid()
        )

    @property
    def _stage_directory_mode(self) -> int:
        return 0o2750 if self.shared_gid is not None else 0o700

    @property
    def _stage_manifest_mode(self) -> int:
        return 0o440 if self.shared_gid is not None else 0o400

    def _open_ingress_root(self) -> int:
        return _open_validated_directory(
            self.root,
            label="workspace ingress root",
            uid=self._stage_uid,
            gid=self.shared_gid,
            mode=self._stage_directory_mode,
        )

    def _open_generation_descriptors(
        self, job_id: UUID, generation: int
    ) -> tuple[int, int, int]:
        root_fd = self._open_ingress_root()
        try:
            job_fd = _open_validated_directory(
                str(job_id),
                dir_fd=root_fd,
                label="staged workspace job",
                uid=self._stage_uid,
                gid=self.shared_gid,
                mode=self._stage_directory_mode,
            )
        except BaseException:
            os.close(root_fd)
            raise
        try:
            generation_fd = _open_validated_directory(
                str(generation),
                dir_fd=job_fd,
                label="staged workspace generation",
                uid=self._stage_uid,
                gid=self.shared_gid,
                mode=self._stage_directory_mode,
            )
        except BaseException:
            os.close(job_fd)
            os.close(root_fd)
            raise
        return root_fd, job_fd, generation_fd

    def _require_stage_ownership(self, path: Path, *, directory: bool) -> None:
        """Materialize side: confirm a staged path is owned by the app identity.

        Active only when ``expected_stage_uid`` is set (external broker).  In the
        co-process default (unset) this is a no-op, preserving today's checks.
        """
        if self.expected_stage_uid is None:
            return
        if directory:
            descriptor = _open_validated_directory(
                path,
                label="staged workspace",
                uid=self._stage_uid,
                gid=self.shared_gid,
                mode=self._stage_directory_mode,
            )
            os.close(descriptor)
            return
        try:
            descriptor = os.open(path, _FILE_OPEN_FLAGS)
        except OSError as exc:
            raise WorkspaceIngressProblem(
                "staged workspace path is not a safe file"
            ) from exc
        try:
            _check_regular_metadata(
                os.fstat(descriptor),
                label="staged workspace",
                uid=self._stage_uid,
                gid=self.shared_gid,
                mode=self._stage_manifest_mode,
            )
        finally:
            os.close(descriptor)

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
    def _parse_manifest(payload: bytes) -> StagedWorkspace:
        try:
            raw = json.loads(payload.decode("ascii"))
            if raw.get("version") != 1:
                raise ValueError("unsupported manifest version")
            staged = StagedWorkspace(
                job_id=UUID(raw["job_id"]),
                generation=raw["generation"],
                sha256=raw["sha256"],
                file_count=raw["file_count"],
                byte_count=raw["byte_count"],
            )
            _validate_generation(staged.generation)
            if (
                not isinstance(staged.sha256, str)
                or len(staged.sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in staged.sha256
                )
                or isinstance(staged.file_count, bool)
                or not isinstance(staged.file_count, int)
                or staged.file_count < 0
                or isinstance(staged.byte_count, bool)
                or not isinstance(staged.byte_count, int)
                or staged.byte_count < 0
            ):
                raise ValueError("invalid manifest fields")
            return staged
        except Exception as exc:
            raise WorkspaceIngressProblem("workspace seed manifest is invalid") from exc

    @classmethod
    def _read_manifest(cls, root: Path) -> StagedWorkspace:
        try:
            payload = (root / _MANIFEST).read_bytes()
        except OSError as exc:
            raise WorkspaceIngressProblem("workspace seed manifest is invalid") from exc
        return cls._parse_manifest(payload)

    def _read_manifest_at(self, generation_fd: int) -> StagedWorkspace:
        try:
            descriptor = os.open(_MANIFEST, _FILE_OPEN_FLAGS, dir_fd=generation_fd)
        except OSError as exc:
            raise WorkspaceIngressProblem("workspace seed manifest is invalid") from exc
        try:
            before = os.fstat(descriptor)
            _check_regular_metadata(
                before,
                label="workspace seed manifest",
                uid=self._stage_uid,
                gid=self.shared_gid,
                mode=self._stage_manifest_mode,
            )
            if before.st_size > 16 * 1024:
                raise WorkspaceIngressProblem("workspace seed manifest is invalid")
            chunks: list[bytes] = []
            remaining = before.st_size + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 4096))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            after = os.fstat(descriptor)
            if len(payload) != before.st_size or any(
                getattr(after, field) != getattr(before, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_uid",
                    "st_gid",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            ):
                raise WorkspaceIngressProblem(
                    "workspace seed manifest changed while read"
                )
            return self._parse_manifest(payload)
        finally:
            os.close(descriptor)

    def stage(self, job_id: UUID, generation: int, source: Path) -> StagedWorkspace:
        target = self._generation_root(job_id, generation)
        source_resolved = source.resolve()
        root_resolved = self.root.resolve()
        if source_resolved == root_resolved or root_resolved.is_relative_to(
            source_resolved
        ):
            raise WorkspaceIngressProblem(
                "workspace seed cannot contain ingress storage"
            )
        shared = self.shared_gid is not None
        with self._lock(job_id, generation):
            root_fd = self._open_ingress_root()
            try:
                job_fd = _ensure_owned_child(
                    root_fd,
                    str(job_id),
                    mode=self._stage_directory_mode,
                    gid=self.shared_gid,
                    label="workspace ingress job",
                )
            finally:
                os.close(root_fd)
            os.close(job_fd)
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
        self, tree_fd: int, destination: Path, staged: StagedWorkspace
    ) -> None:
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
            tree_fd,
            destination,
            max_files=self.max_files,
            max_bytes=self.max_bytes,
            expected_source_uid=self._stage_uid,
            expected_source_gid=self.shared_gid,
            source_shared=self.shared_gid is not None,
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
            root_fd, job_fd, generation_fd = self._open_generation_descriptors(
                identity.job_id, identity.generation
            )
            try:
                staged = self._read_manifest_at(generation_fd)
                if (
                    staged.job_id != identity.job_id
                    or staged.generation != identity.generation
                ):
                    raise WorkspaceIngressProblem(
                        "workspace seed manifest identity does not match its path"
                    )
                if self.marker_root is None:
                    # Default (co-process): manifest-then-marker, in-tree marker.
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
                tree_fd = _open_validated_directory(
                    _TREE,
                    dir_fd=generation_fd,
                    label="staged workspace tree",
                    uid=self._stage_uid,
                    gid=self.shared_gid,
                    mode=self._stage_directory_mode,
                )
                try:
                    self._materialize_tree(tree_fd, destination, staged)
                finally:
                    os.close(tree_fd)
            finally:
                os.close(generation_fd)
                os.close(job_fd)
                os.close(root_fd)
            if self.marker_root is not None:
                self._write_consumed_marker(
                    identity.job_id, identity.generation, operation_id
                )
            else:
                _write_atomic(
                    target / _CONSUMED,
                    operation_id.encode("ascii"),
                )

    def discard(self, identity: GenerationRuntimeIdentity) -> None:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        self._generation_root(identity.job_id, identity.generation)
        with self._lock(identity.job_id, identity.generation):
            if self.marker_root is not None:
                # Marker-only: the producer owns and deletes the tree; the
                # consumer records intent and never rmtrees app-owned data.
                gen_dir = self._ensure_marker_gen_dir(
                    identity.job_id, identity.generation
                )
                _write_atomic(gen_dir / _DISCARDED, b"")
                return
            self._prune_locked(identity.job_id, identity.generation)

    def _prune_locked(self, job_id: UUID, generation: int) -> None:
        """Descriptor-rooted producer cleanup; caller holds the per-job lock."""
        root_fd = self._open_ingress_root()
        job_name = str(job_id)
        try:
            try:
                os.stat(job_name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            except OSError as exc:
                raise WorkspaceIngressProblem(
                    "workspace ingress job cannot be inspected"
                ) from exc
            job_fd = _open_validated_directory(
                job_name,
                dir_fd=root_fd,
                label="workspace ingress job",
                uid=self._stage_uid,
                gid=self.shared_gid,
                mode=self._stage_directory_mode,
            )
            try:
                _remove_tree_at(
                    job_fd,
                    str(generation),
                    uid=self._stage_uid,
                    gid=self.shared_gid,
                    mode=self._stage_directory_mode,
                    missing_ok=True,
                )
                _remove_empty_open_directory(root_fd, job_name, job_fd)
            finally:
                os.close(job_fd)
        finally:
            os.close(root_fd)

    def prune(self, job_id: UUID, generation: int) -> None:
        """Producer-side deletion of one staged generation (+ empty parent)."""
        self._generation_root(job_id, generation)
        with self._lock(job_id, generation):
            self._prune_locked(job_id, generation)

    def _prune_stale_temporary(
        self,
        job_fd: int,
        name: str,
        *,
        cutoff: float,
    ) -> bool:
        """Reap an interrupted ``stage`` tree without mode/umask assumptions."""
        try:
            parent = os.fstat(job_fd)
            descriptor = _open_owned_directory(
                name,
                dir_fd=job_fd,
                label="workspace ingress stale temporary stage",
                uid=self._stage_uid,
                device=parent.st_dev,
            )
        except (OSError, WorkspaceIngressProblem):
            return False

        try:
            identity = os.fstat(descriptor)
            if identity.st_mtime >= cutoff:
                return False
        except OSError:
            return False
        finally:
            os.close(descriptor)

        try:
            return _remove_owned_tree_at(
                job_fd,
                name,
                uid=self._stage_uid,
                device=parent.st_dev,
                expected_identity=(identity.st_dev, identity.st_ino),
            )
        except WorkspaceIngressProblem:
            return False

    def prune_stale(self, *, max_age_seconds: int) -> int:
        """Reap published stages and interrupted temp trees past the cutoff.

        Defensive by design: entries that vanish concurrently are skipped, never
        raised on. Returns the number of stage entries removed.
        """
        cutoff = time.time() - max_age_seconds
        removed = 0
        try:
            root_fd = self._open_ingress_root()
        except WorkspaceIngressProblem:
            return removed
        try:
            try:
                job_names = os.listdir(root_fd)
            except OSError:
                return removed
            for job_name in job_names:
                try:
                    job_id = UUID(job_name)
                except ValueError:
                    continue
                with self._lock(job_id, 0):
                    try:
                        job_fd = _open_validated_directory(
                            job_name,
                            dir_fd=root_fd,
                            label="workspace ingress stale job",
                            uid=self._stage_uid,
                            gid=self.shared_gid,
                            mode=self._stage_directory_mode,
                        )
                    except WorkspaceIngressProblem:
                        continue
                    try:
                        try:
                            generation_names = os.listdir(job_fd)
                        except OSError:
                            continue
                        for generation_name in generation_names:
                            if _temporary_generation(generation_name) is not None:
                                if self._prune_stale_temporary(
                                    job_fd,
                                    generation_name,
                                    cutoff=cutoff,
                                ):
                                    removed += 1
                                continue
                            try:
                                generation = int(generation_name)
                                _validate_generation(generation)
                            except (TypeError, ValueError):
                                continue
                            try:
                                generation_fd = _open_validated_directory(
                                    generation_name,
                                    dir_fd=job_fd,
                                    label="workspace ingress stale generation",
                                    uid=self._stage_uid,
                                    gid=self.shared_gid,
                                    mode=self._stage_directory_mode,
                                )
                            except WorkspaceIngressProblem:
                                continue
                            try:
                                identity = os.fstat(generation_fd)
                                try:
                                    manifest = os.stat(
                                        _MANIFEST,
                                        dir_fd=generation_fd,
                                        follow_symlinks=False,
                                    )
                                    _check_regular_metadata(
                                        manifest,
                                        label="workspace seed manifest",
                                        uid=self._stage_uid,
                                        gid=self.shared_gid,
                                        mode=self._stage_manifest_mode,
                                    )
                                except (OSError, WorkspaceIngressProblem):
                                    continue
                                if manifest.st_mtime >= cutoff:
                                    continue
                                try:
                                    deleted = _remove_tree_at(
                                        job_fd,
                                        generation_name,
                                        uid=self._stage_uid,
                                        gid=self.shared_gid,
                                        mode=self._stage_directory_mode,
                                        expected_identity=(
                                            identity.st_dev,
                                            identity.st_ino,
                                        ),
                                    )
                                except WorkspaceIngressProblem:
                                    continue
                                if deleted:
                                    removed += 1
                            finally:
                                os.close(generation_fd)
                        try:
                            _remove_empty_open_directory(root_fd, job_name, job_fd)
                        except WorkspaceIngressProblem:
                            pass
                    finally:
                        os.close(job_fd)
        finally:
            os.close(root_fd)
        return removed
