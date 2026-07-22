"""Provision the split broker's host-backed directories without replacing data.

The command is intentionally stdlib-only so it is available in the application
image before either long-running service starts.  It creates or repairs only the
five top-level directories in the broker ownership matrix; it never deletes or
recursively changes an existing tree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import sys


class ProvisioningProblem(RuntimeError):
    """The requested ownership matrix cannot be established safely."""


@dataclass(frozen=True, slots=True)
class _DirectorySpec:
    path: Path
    uid: int
    gid: int
    mode: int


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be a numeric id") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{name} must be a non-negative numeric id")
    return parsed


def _numeric_id(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a numeric id") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative numeric id")
    return parsed


def _parser() -> argparse.ArgumentParser:
    root = os.environ.get("OPENLOOP_BROKER_ROOT")
    parser = argparse.ArgumentParser(
        prog="python -m openloop.broker_provision",
        description="Create or repair split-broker host directory ownership.",
    )
    parser.add_argument(
        "--root",
        default=root,
        help=(
            "common root used to derive control, ingress, runtime, state, and "
            "receipts paths (env: OPENLOOP_BROKER_ROOT)"
        ),
    )
    parser.add_argument(
        "--control-socket-dir",
        default=os.environ.get("BROKER_CONTROL_SOCKET_DIR"),
    )
    parser.add_argument(
        "--ingress-root",
        default=os.environ.get("BROKER_INGRESS_ROOT"),
    )
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get("BROKER_RUNTIME_ROOT"),
    )
    parser.add_argument(
        "--state-root",
        default=os.environ.get("BROKER_STATE_ROOT"),
    )
    parser.add_argument(
        "--receipts-root",
        default=os.environ.get("BROKER_CHECKPOINT_RECEIPT_ROOT"),
    )
    parser.add_argument(
        "--app-uid",
        type=_numeric_id,
        default=_environment_int("OPENLOOP_APP_UID", 1000),
    )
    parser.add_argument(
        "--broker-uid",
        type=_numeric_id,
        default=_environment_int("OPENLOOP_BROKER_UID", 10002),
    )
    parser.add_argument(
        "--data-gid",
        type=_numeric_id,
        default=_environment_int("OPENLOOP_DATA_GID", 10777),
    )
    return parser


def _selected_path(value: str | None, root: Path | None, child: str) -> Path:
    if value:
        path = Path(value)
    elif root is not None:
        path = root / child
    else:
        raise ProvisioningProblem(
            f"{child} path is required (set an explicit path or OPENLOOP_BROKER_ROOT)"
        )
    if not path.is_absolute():
        raise ProvisioningProblem(f"{child} path must be absolute")
    if any(part == ".." for part in path.parts):
        raise ProvisioningProblem(f"{child} path must not contain '..'")
    return path


def _specs(args: argparse.Namespace) -> tuple[_DirectorySpec, ...]:
    root = Path(args.root) if args.root else None
    if root is not None and not root.is_absolute():
        raise ProvisioningProblem("OPENLOOP_BROKER_ROOT must be absolute")
    specs = (
        _DirectorySpec(
            _selected_path(args.control_socket_dir, root, "control"),
            args.broker_uid,
            args.data_gid,
            0o750,
        ),
        _DirectorySpec(
            _selected_path(args.ingress_root, root, "ingress"),
            args.app_uid,
            args.data_gid,
            0o2750,
        ),
        _DirectorySpec(
            _selected_path(args.runtime_root, root, "runtime"),
            args.broker_uid,
            args.data_gid,
            0o750,
        ),
        _DirectorySpec(
            _selected_path(args.state_root, root, "state"),
            args.broker_uid,
            args.data_gid,
            0o700,
        ),
        _DirectorySpec(
            _selected_path(args.receipts_root, root, "receipts"),
            args.app_uid,
            args.data_gid,
            0o2750,
        ),
    )
    paths = [spec.path for spec in specs]
    if any(path == Path(os.sep) for path in paths):
        raise ProvisioningProblem("a provisioned directory cannot be the filesystem root")
    if len(set(paths)) != len(paths):
        raise ProvisioningProblem("the five provisioned directories must be distinct")
    return specs


def _open_directory(path: Path) -> int:
    """Open/create ``path`` without following any path-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in path.parts[1:]:
            if component in {"", "."}:
                continue
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, 0o700, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _provision(spec: _DirectorySpec) -> None:
    try:
        descriptor = _open_directory(spec.path)
    except (NotADirectoryError, OSError) as error:
        raise ProvisioningProblem(f"directory path rejected: {spec.path}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ProvisioningProblem(f"directory path rejected: {spec.path}")
        if info.st_uid != spec.uid or info.st_gid != spec.gid:
            os.fchown(descriptor, spec.uid, spec.gid)
            info = os.fstat(descriptor)
        if stat.S_IMODE(info.st_mode) != spec.mode:
            os.fchmod(descriptor, spec.mode)
    except ProvisioningProblem:
        raise
    except OSError as error:
        raise ProvisioningProblem(f"could not provision directory: {spec.path}") from error
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    """Create/repair the ownership matrix; return nonzero on any unsafe path."""
    try:
        args = _parser().parse_args(argv)
        for spec in _specs(args):
            _provision(spec)
    except (argparse.ArgumentTypeError, ProvisioningProblem) as error:
        print(f"broker provisioning failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
