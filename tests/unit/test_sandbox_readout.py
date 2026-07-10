"""Unit tests for read-out containment (Phase 0 lock 4 — no docker needed).

``outputs/`` is a directory hostile code controlled; reading it on the host is
the exfiltration gate. Every hostile entry the lock names is planted for real:
symlink, FIFO, hardlink, nested path — each must be refused, the FIFO without
blocking the open.
"""

import os

import pytest

from openloop.sandbox import ReadOutViolation, read_contained


@pytest.fixture()
def outputs(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    return root


def test_reads_a_plain_report(outputs):
    (outputs / "report.md").write_text("# findings\n")
    data, truncated = read_contained(outputs, "report.md", max_bytes=1024)
    assert data == b"# findings\n"
    assert truncated is False


def test_oversized_report_is_truncated_with_the_flag(outputs):
    (outputs / "report.md").write_bytes(b"x" * 1000)
    data, truncated = read_contained(outputs, "report.md", max_bytes=10)
    assert data == b"x" * 10
    assert truncated is True


def test_symlink_to_host_file_is_refused(outputs, tmp_path):
    """The classic kill: outputs/report.md -> an absolute host path. A naive
    open follows it out of the sandbox boundary."""
    secret = tmp_path / "host-secret"
    secret.write_text("hostage")
    (outputs / "report.md").symlink_to(secret)
    with pytest.raises(ReadOutViolation, match="symlink"):
        read_contained(outputs, "report.md", max_bytes=1024)


def test_fifo_is_refused_without_blocking(outputs):
    """O_NOFOLLOW does NOT stop a FIFO open from blocking — O_NONBLOCK does.
    If this test hangs, the flag is gone."""
    os.mkfifo(outputs / "report.md")
    with pytest.raises(ReadOutViolation, match="not a regular file"):
        read_contained(outputs, "report.md", max_bytes=1024)


def test_hardlink_is_refused(outputs, tmp_path):
    """st_nlink > 1: a hardlink planted in outputs/ can alias inputs/ —
    within the provisioned blast radius, refused as defense-in-depth."""
    original = tmp_path / "aliased"
    original.write_text("data")
    os.link(original, outputs / "report.md")
    with pytest.raises(ReadOutViolation, match="links"):
        read_contained(outputs, "report.md", max_bytes=1024)


def test_directory_entry_is_refused(outputs):
    (outputs / "report.md").mkdir()
    with pytest.raises(ReadOutViolation, match="not a regular file"):
        read_contained(outputs, "report.md", max_bytes=1024)


@pytest.mark.parametrize("name", ["a/b", "../escape", ".", "..", "", "x\x00y"])
def test_non_bare_filenames_are_refused(outputs, name):
    """Filenames are hostile; only bare single components are accepted —
    nested paths wait for openat2(RESOLVE_BENEATH)."""
    with pytest.raises(ReadOutViolation, match="bare filename"):
        read_contained(outputs, name, max_bytes=1024)


def test_missing_report_is_an_outcome_not_an_attack(outputs):
    with pytest.raises(FileNotFoundError):
        read_contained(outputs, "report.md", max_bytes=1024)


def test_symlinked_root_is_refused(outputs, tmp_path):
    """The root itself must be a real directory — a symlinked stand-in fails
    the O_NOFOLLOW dirfd open."""
    alias = tmp_path / "alias"
    alias.symlink_to(outputs)
    with pytest.raises(OSError):
        read_contained(alias, "report.md", max_bytes=1024)
