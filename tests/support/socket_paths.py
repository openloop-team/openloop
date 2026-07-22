"""Portable short roots for tests that bind deeply nested Unix sockets."""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
import secrets
import tempfile

from openloop.broker_rpc.server import MAX_UNIX_SOCKET_PATH_BYTES


TEST_SOCKET_TMPDIR_ENV = "OPENLOOP_TEST_SOCKET_TMPDIR"
_DIRECTORY_PREFIX = "ol-"
_RANDOM_NAME_CHARS = 8
_CREATE_ATTEMPTS = 100
# Covers runtime_root/<job>/<generation>/socket/agent.sock with a conservative
# 32-bit generation plus tests that place runtime_root beneath the fixture root.
LONGEST_BROKER_TEST_SOCKET_SUFFIX = Path(
    "runtime",
    "00000000-0000-0000-0000-000000000000",
    str(2**31 - 1),
    "socket",
    "agent.sock",
)


class ShortSocketPathProblem(RuntimeError):
    """No trusted temporary directory can satisfy the Unix-socket path budget."""


def _prospective_socket_path(parent: Path, suffix: Path) -> Path:
    return parent / f"{_DIRECTORY_PREFIX}{'x' * _RANDOM_NAME_CHARS}" / suffix


def _resolve_candidate(candidate: Path) -> Path:
    if not isinstance(candidate, Path) or not candidate.is_absolute():
        raise ShortSocketPathProblem("socket temporary root must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ShortSocketPathProblem(
            "socket temporary root does not exist"
        ) from error
    if not resolved.is_dir() or not os.access(resolved, os.W_OK | os.X_OK):
        raise ShortSocketPathProblem(
            "socket temporary root must be a writable directory"
        )
    return resolved


def _fits_budget(parent: Path, suffix: Path) -> bool:
    return (
        len(os.fsencode(_prospective_socket_path(parent, suffix)))
        <= MAX_UNIX_SOCKET_PATH_BYTES
    )


def select_short_socket_parent(
    *,
    override: Path | None = None,
    candidates: Iterable[Path] | None = None,
    required_suffix: Path = LONGEST_BROKER_TEST_SOCKET_SUFFIX,
) -> Path:
    """Select the shortest resolved writable parent that fits ``required_suffix``."""
    if not isinstance(required_suffix, Path) or required_suffix.is_absolute():
        raise TypeError("required_suffix must be a relative Path")
    configured = override
    if configured is None:
        rendered = os.environ.get(TEST_SOCKET_TMPDIR_ENV)
        configured = Path(rendered) if rendered else None
    if configured is not None:
        resolved = _resolve_candidate(configured)
        if not _fits_budget(resolved, required_suffix):
            raise ShortSocketPathProblem(
                "configured socket temporary root exceeds the path budget"
            )
        return resolved

    proposed = tuple(candidates) if candidates is not None else (
        Path("/tmp"),
        Path("/private/tmp"),
        Path(tempfile.gettempdir()),
    )
    viable: dict[Path, None] = {}
    for candidate in proposed:
        try:
            resolved = _resolve_candidate(candidate)
        except ShortSocketPathProblem:
            continue
        if _fits_budget(resolved, required_suffix):
            viable[resolved] = None
    if not viable:
        raise ShortSocketPathProblem(
            "no writable socket temporary root fits the path budget"
        )
    return min(
        viable,
        key=lambda path: (len(os.fsencode(path)), os.fsencode(path)),
    )


def create_short_socket_root(
    *,
    override: Path | None = None,
    candidates: Iterable[Path] | None = None,
    required_suffix: Path = LONGEST_BROKER_TEST_SOCKET_SUFFIX,
) -> Path:
    """Atomically create one owner-private short socket root."""
    parent = select_short_socket_parent(
        override=override,
        candidates=candidates,
        required_suffix=required_suffix,
    )
    for _attempt in range(_CREATE_ATTEMPTS):
        root = parent / f"{_DIRECTORY_PREFIX}{secrets.token_hex(4)}"
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            continue
        root.chmod(0o700)
        return root
    raise ShortSocketPathProblem("could not allocate a unique socket temporary root")


__all__ = [
    "LONGEST_BROKER_TEST_SOCKET_SUFFIX",
    "ShortSocketPathProblem",
    "TEST_SOCKET_TMPDIR_ENV",
    "create_short_socket_root",
    "select_short_socket_parent",
]
