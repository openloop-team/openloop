"""Portable, symlink-free, budget-aware roots for Unix-socket tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openloop.broker_rpc.server import MAX_UNIX_SOCKET_PATH_BYTES
import tests.support.socket_paths as socket_paths
from tests.support.socket_paths import (
    LONGEST_BROKER_TEST_SOCKET_SUFFIX,
    ShortSocketPathProblem,
    TEST_SOCKET_TMPDIR_ENV,
    create_short_socket_root,
    select_short_socket_parent,
)


def test_selector_resolves_synthetic_symlink_and_real_directory(
    monkeypatch, tmp_path
):
    real = tmp_path / "private-tmp"
    real.mkdir()
    alias = tmp_path / "tmp"
    alias.symlink_to(real, target_is_directory=True)
    suffix = Path("control.sock")
    # The synthetic pytest root is intentionally long. This test isolates
    # realpath behavior; budget behavior has its own test below.
    monkeypatch.setattr(socket_paths, "MAX_UNIX_SOCKET_PATH_BYTES", 4096)

    assert select_short_socket_parent(
        candidates=(alias,), required_suffix=suffix
    ) == real.resolve()
    assert select_short_socket_parent(
        candidates=(real,), required_suffix=suffix
    ) == real.resolve()


def test_explicit_environment_override_is_resolved(monkeypatch, tmp_path):
    real = tmp_path / "private-tmp"
    real.mkdir()
    alias = tmp_path / "tmp"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv(TEST_SOCKET_TMPDIR_ENV, os.fspath(alias))
    monkeypatch.setattr(socket_paths, "MAX_UNIX_SOCKET_PATH_BYTES", 4096)

    selected = select_short_socket_parent(required_suffix=Path("control.sock"))

    assert selected == real.resolve()


def test_explicit_invalid_override_never_falls_back(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(ShortSocketPathProblem, match="does not exist"):
        select_short_socket_parent(
            override=missing,
            candidates=(Path("/tmp"),),
            required_suffix=Path("control.sock"),
        )


def test_selector_rejects_a_candidate_that_cannot_fit_the_server_budget(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    too_long = Path("x" * MAX_UNIX_SOCKET_PATH_BYTES)

    with pytest.raises(ShortSocketPathProblem, match="no writable"):
        select_short_socket_parent(
            candidates=(candidate,),
            required_suffix=too_long,
        )


def test_created_root_is_private_resolved_and_within_shared_budget():
    root = create_short_socket_root()
    try:
        assert root == root.resolve()
        assert root.stat().st_mode & 0o777 == 0o700
        deepest_socket = root / LONGEST_BROKER_TEST_SOCKET_SUFFIX
        assert len(os.fsencode(deepest_socket)) <= MAX_UNIX_SOCKET_PATH_BYTES
    finally:
        root.rmdir()


def test_shared_fixture_returns_a_private_resolved_root(short_socket_root):
    assert short_socket_root == short_socket_root.resolve()
    assert short_socket_root.stat().st_mode & 0o777 == 0o700
