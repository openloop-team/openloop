"""Descriptor-anchored generation filesystem for the Docker runtime driver."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from openloop.tools.openhands_relay import (
    CompiledOpenHandsRelay,
    OpenHandsRelayProfileError,
    install_relay_artifacts,
)

from .contract import RuntimeIdentityConflict, RuntimeUnavailable
from .docker_policy import GenerationPaths


_EXPECTED_ROOT_ENTRIES = frozenset({"relay", "socket", "workspace"})
_EXPECTED_ARTIFACT_ENTRIES = frozenset({"haproxy.cfg", "relay-capability"})


def _directory_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeUnavailable("secure generation directories require O_NOFOLLOW")
    return flags | nofollow | getattr(os, "O_CLOEXEC", 0)


def _check_directory_descriptor(
    descriptor: int,
    *,
    name: str,
    uid: int,
    mode: int = 0o700,
) -> None:
    try:
        info = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeUnavailable(f"cannot inspect {name} directory") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeIdentityConflict(f"{name} is not a directory")
    if info.st_uid != uid:
        raise RuntimeIdentityConflict(f"{name} directory owner does not match")
    if stat.S_IMODE(info.st_mode) != mode:
        raise RuntimeIdentityConflict(f"{name} directory mode does not match")


def _open_trusted_root(path: Path, *, name: str, uid: int) -> int:
    try:
        descriptor = os.open(path, _directory_flags())
    except FileNotFoundError as exc:
        raise RuntimeUnavailable(f"configured {name} does not exist") from exc
    except OSError as exc:
        raise RuntimeUnavailable(f"configured {name} is not safely accessible") from exc
    try:
        _check_directory_descriptor(descriptor, name=name, uid=uid)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _ensure_child(parent_fd: int, name: str, *, label: str, uid: int) -> int:
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise RuntimeUnavailable(f"invalid derived {label} name")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise RuntimeUnavailable(f"cannot create {label} directory") from exc
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise RuntimeIdentityConflict(f"{label} path is not a safe directory") from exc
    try:
        _check_directory_descriptor(descriptor, name=label, uid=uid)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _entries(descriptor: int, *, label: str) -> frozenset[str]:
    try:
        return frozenset(os.listdir(descriptor))
    except (OSError, TypeError) as exc:
        raise RuntimeUnavailable(f"cannot inspect {label} directory") from exc


def _validate_artifact(
    directory_fd: int,
    name: str,
    expected: bytes,
    *,
    uid: int,
) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeIdentityConflict(
            f"relay artifact {name} is not safely readable"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeIdentityConflict(
                f"relay artifact {name} is not a regular file"
            )
        if info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o400:
            raise RuntimeIdentityConflict(
                f"relay artifact {name} metadata does not match"
            )
        chunks: list[bytes] = []
        remaining = len(expected) + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if b"".join(chunks) != expected:
            raise RuntimeIdentityConflict(
                f"relay artifact {name} content does not match"
            )
    finally:
        os.close(descriptor)


def prepare_generation_filesystem(
    paths: GenerationPaths,
    compiled: CompiledOpenHandsRelay,
    *,
    uid: int,
) -> None:
    """Create or validate the exact generation directory and relay artifacts."""
    runtime_fd = _open_trusted_root(paths.root.parents[1], name="runtime root", uid=uid)
    state_fd = _open_trusted_root(paths.state.parents[1], name="state root", uid=uid)
    opened: list[int] = [runtime_fd, state_fd]
    try:
        job_fd = _ensure_child(
            runtime_fd,
            paths.root.parent.name,
            label="runtime job",
            uid=uid,
        )
        opened.append(job_fd)
        generation_fd = _ensure_child(
            job_fd, paths.root.name, label="runtime generation", uid=uid
        )
        opened.append(generation_fd)
        artifact_fd = _ensure_child(
            generation_fd, paths.artifacts.name, label="relay artifacts", uid=uid
        )
        opened.append(artifact_fd)
        socket_fd = _ensure_child(
            generation_fd, paths.socket.name, label="relay socket", uid=uid
        )
        opened.append(socket_fd)
        workspace_fd = _ensure_child(
            generation_fd, paths.workspace.name, label="runtime workspace", uid=uid
        )
        opened.append(workspace_fd)

        state_job_fd = _ensure_child(
            state_fd, paths.state.parent.name, label="state job", uid=uid
        )
        opened.append(state_job_fd)
        state_runtime_fd = _ensure_child(
            state_job_fd, paths.state.name, label="agent state", uid=uid
        )
        opened.append(state_runtime_fd)

        generation_entries = _entries(generation_fd, label="runtime generation")
        if generation_entries != _EXPECTED_ROOT_ENTRIES:
            raise RuntimeIdentityConflict(
                "runtime generation directory entries do not match"
            )
        artifact_entries = _entries(artifact_fd, label="relay artifacts")
        if not artifact_entries:
            try:
                install_relay_artifacts(artifact_fd, compiled)
            except OpenHandsRelayProfileError as exc:
                raise RuntimeUnavailable(
                    "failed to install fixed relay artifacts"
                ) from exc
        elif artifact_entries != _EXPECTED_ARTIFACT_ENTRIES:
            raise RuntimeIdentityConflict(
                "relay artifact directory entries do not match"
            )

        _validate_artifact(
            artifact_fd,
            "haproxy.cfg",
            compiled.haproxy_config,
            uid=uid,
        )
        _validate_artifact(
            artifact_fd,
            "relay-capability",
            compiled.capability_file.payload,
            uid=uid,
        )
    finally:
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except OSError:
                pass


def generation_filesystem_observation(
    paths: GenerationPaths,
    *,
    uid: int,
) -> tuple[bool, bool]:
    """Return structural artifact/workspace readiness without following links."""
    try:
        artifact_info = paths.artifacts.lstat()
        workspace_info = paths.workspace.lstat()
    except FileNotFoundError:
        return False, False
    artifacts_ready = (
        stat.S_ISDIR(artifact_info.st_mode)
        and artifact_info.st_uid == uid
        and stat.S_IMODE(artifact_info.st_mode) == 0o700
        and frozenset(os.listdir(paths.artifacts)) == _EXPECTED_ARTIFACT_ENTRIES
    )
    workspace_ready = (
        stat.S_ISDIR(workspace_info.st_mode)
        and workspace_info.st_uid == uid
        and stat.S_IMODE(workspace_info.st_mode) == 0o700
    )
    return artifacts_ready, workspace_ready


def release_generation_filesystem(paths: GenerationPaths, *, uid: int) -> None:
    """Remove only disposable exact-generation paths; preserve durable state."""
    if not paths.root.exists():
        return
    root_info = paths.root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != uid:
        raise RuntimeIdentityConflict("runtime generation root identity does not match")
    entries = frozenset(os.listdir(paths.root))
    if not entries.issubset(_EXPECTED_ROOT_ENTRIES):
        raise RuntimeIdentityConflict(
            "runtime generation contains an unknown root entry"
        )

    if paths.artifacts.exists():
        info = paths.artifacts.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid:
            raise RuntimeIdentityConflict(
                "relay artifact directory identity does not match"
            )
        artifact_entries = frozenset(os.listdir(paths.artifacts))
        if not artifact_entries.issubset(_EXPECTED_ARTIFACT_ENTRIES):
            raise RuntimeIdentityConflict(
                "relay artifact directory contains an unknown entry"
            )
        for name in _EXPECTED_ARTIFACT_ENTRIES:
            target = paths.artifacts / name
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        paths.artifacts.rmdir()

    if paths.socket.exists():
        info = paths.socket.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid:
            raise RuntimeIdentityConflict(
                "relay socket directory identity does not match"
            )
        socket_entries = frozenset(os.listdir(paths.socket))
        if not socket_entries.issubset({"agent.sock"}):
            raise RuntimeIdentityConflict(
                "relay socket directory contains an unknown entry"
            )
        socket_path = paths.socket / "agent.sock"
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        paths.socket.rmdir()

    if paths.workspace.exists():
        info = paths.workspace.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid:
            raise RuntimeIdentityConflict("workspace directory identity does not match")
        # The agent is stopped before this call, so its untrusted tree is no
        # longer racing this descriptor-anchored generation identity check.
        shutil.rmtree(paths.workspace)

    paths.root.rmdir()
    try:
        paths.root.parent.rmdir()
    except OSError:
        pass


__all__ = [
    "generation_filesystem_observation",
    "prepare_generation_filesystem",
    "release_generation_filesystem",
]
