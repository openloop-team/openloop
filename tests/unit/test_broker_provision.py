"""Exact, idempotent filesystem provisioning for the split broker services."""

from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from openloop.broker_provision import main


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _arguments(root: Path) -> list[str]:
    return [
        "--root",
        str(root),
        "--app-uid",
        str(os.getuid()),
        "--broker-uid",
        str(os.getuid()),
        "--data-gid",
        str(os.getgid()),
    ]


def test_provisions_exact_matrix_idempotently_and_corrects_drift(tmp_path):
    root = tmp_path / "broker"
    root.mkdir()
    arguments = _arguments(root)

    assert main(arguments) == 0

    expected = {
        "control": 0o750,
        "ingress": 0o2750,
        "runtime": 0o750,
        "state": 0o700,
        "receipts": 0o2750,
    }
    first = {}
    for name, mode in expected.items():
        path = root / name
        info = path.stat()
        assert stat.S_ISDIR(info.st_mode)
        assert info.st_uid == os.getuid()
        assert info.st_gid == os.getgid()
        assert _mode(path) == mode
        first[name] = (info.st_ino, info.st_ctime_ns, info.st_mtime_ns)

    assert main(arguments) == 0
    for name in expected:
        info = (root / name).stat()
        assert (info.st_ino, info.st_ctime_ns, info.st_mtime_ns) == first[name]

    (root / "ingress").chmod(0o777)
    (root / "state").chmod(0o755)
    assert main(arguments) == 0
    assert _mode(root / "ingress") == 0o2750
    assert _mode(root / "state") == 0o700


def test_explicit_path_environment_is_supported(tmp_path, monkeypatch):
    paths = {
        "BROKER_CONTROL_SOCKET_DIR": tmp_path / "socket",
        "BROKER_INGRESS_ROOT": tmp_path / "in",
        "BROKER_RUNTIME_ROOT": tmp_path / "run",
        "BROKER_STATE_ROOT": tmp_path / "private",
        "BROKER_CHECKPOINT_RECEIPT_ROOT": tmp_path / "receipts-only",
    }
    for name, path in paths.items():
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("OPENLOOP_APP_UID", str(os.getuid()))
    monkeypatch.setenv("OPENLOOP_BROKER_UID", str(os.getuid()))
    monkeypatch.setenv("OPENLOOP_DATA_GID", str(os.getgid()))

    assert main([]) == 0
    assert _mode(paths["BROKER_CONTROL_SOCKET_DIR"]) == 0o750
    assert _mode(paths["BROKER_INGRESS_ROOT"]) == 0o2750
    assert _mode(paths["BROKER_RUNTIME_ROOT"]) == 0o750
    assert _mode(paths["BROKER_STATE_ROOT"]) == 0o700
    assert _mode(paths["BROKER_CHECKPOINT_RECEIPT_ROOT"]) == 0o2750


@pytest.mark.parametrize("bad_kind", ["file", "symlink"])
def test_refuses_non_directory_or_symlink_without_deleting_it(tmp_path, bad_kind):
    root = tmp_path / "broker"
    root.mkdir()
    target = root / "control"
    if bad_kind == "file":
        target.write_text("keep me")
    else:
        destination = tmp_path / "destination"
        destination.mkdir()
        target.symlink_to(destination, target_is_directory=True)

    assert main(_arguments(root)) == 1
    assert target.exists()
    if bad_kind == "file":
        assert target.read_text() == "keep me"
    else:
        assert target.is_symlink()


def test_refuses_symlink_in_ancestor_chain(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    assert main(_arguments(linked)) == 1
    assert list(real.iterdir()) == []


def test_refuses_conflicting_duplicate_paths(tmp_path):
    root = tmp_path / "broker"
    root.mkdir()

    assert main([*_arguments(root), "--state-root", str(root / "runtime")]) == 1
    assert list(root.iterdir()) == []


def test_reports_chown_failure_without_deleting_the_directory(tmp_path, monkeypatch):
    root = tmp_path / "broker"
    root.mkdir()

    def denied(*_args):
        raise PermissionError("simulated foreign uid")

    monkeypatch.setattr(os, "fchown", denied)
    arguments = _arguments(root)
    arguments[arguments.index("--broker-uid") + 1] = str(os.getuid() + 1)
    assert main(arguments) == 1
    assert (root / "control").is_dir()
