import errno
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from openloop.broker_control import workspace_ingress as wi_module
from openloop.broker_control.workspace_ingress import (
    LocalWorkspaceIngress,
    WorkspaceIngressProblem,
)
from openloop.broker_runtime.contract import GenerationRuntimeIdentity


def _identity(job_id, generation=1):
    return GenerationRuntimeIdentity(
        operation_id=uuid4(),
        job_id=job_id,
        generation=generation,
        deadline=(datetime.now(UTC) + timedelta(minutes=5)).replace(microsecond=0),
    )


def _source_tree(base):
    source = base / "source"
    source.mkdir()
    (source / "sub").mkdir()
    (source / "sub" / "data.txt").write_text("payload")
    executable = source / "run.sh"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    (source / "link").symlink_to("run.sh")
    return source


def test_stage_materialize_and_replay_preserve_checkout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    executable = source / "script.sh"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    (source / "link").symlink_to("script.sh")
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)
    job_id = uuid4()

    first = ingress.stage(job_id, 1, source)
    assert ingress.stage(job_id, 1, source) == first
    destination = tmp_path / "workspace"
    destination.mkdir(mode=0o700)
    identity = _identity(job_id)
    ingress.materialize(identity, destination)
    ingress.materialize(identity, destination)

    assert (destination / ".git" / "HEAD").read_text() == "ref: refs/heads/main\n"
    assert (destination / "script.sh").stat().st_mode & 0o111
    assert (destination / "link").is_symlink()
    assert (destination / "link").readlink() == Path("script.sh")


def test_conflicting_stage_and_operation_fail_closed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_text("one")
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)
    job_id = uuid4()
    ingress.stage(job_id, 1, source)
    (source / "file").write_text("two")
    with pytest.raises(WorkspaceIngressProblem):
        ingress.stage(job_id, 1, source)

    destination = tmp_path / "workspace"
    destination.mkdir(mode=0o700)
    ingress.materialize(_identity(job_id), destination)
    with pytest.raises(WorkspaceIngressProblem):
        ingress.materialize(_identity(job_id), destination)


def test_bounds_special_files_and_recursive_root_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "large").write_bytes(b"x" * 5)
    root = source / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root, max_bytes=4)
    with pytest.raises(WorkspaceIngressProblem):
        ingress.stage(uuid4(), 1, source)


def test_discard_removes_only_one_generation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_text("content")
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)
    job_id = uuid4()
    ingress.stage(job_id, 1, source)
    ingress.stage(job_id, 2, source)

    ingress.discard(_identity(job_id, 1))

    assert not (root / str(job_id) / "1").exists()
    assert (root / str(job_id) / "2").exists()

    ingress.discard(_identity(job_id, 2))

    assert not (root / str(job_id)).exists()


# --- Phase C: group handoff, fd-anchored traversal, markers, prune ---------


def test_stage_shared_gid_sets_group_modes(tmp_path):
    source = _source_tree(tmp_path)
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    gid = os.getgid()
    ingress = LocalWorkspaceIngress(root, shared_gid=gid)
    job_id = uuid4()
    ingress.stage(job_id, 1, source)

    gen_root = root / str(job_id) / "1"
    tree = gen_root / "tree"
    manifest = gen_root / "manifest.json"
    # The final generation root is exactly setgid group-mode + the shared gid.
    assert stat.S_IMODE(gen_root.lstat().st_mode) == 0o2750
    assert gen_root.lstat().st_gid == gid
    assert stat.S_IMODE(tree.lstat().st_mode) == 0o2750
    assert stat.S_IMODE((tree / "sub").lstat().st_mode) == 0o2750
    assert (tree / "sub").lstat().st_gid == gid
    assert stat.S_IMODE((tree / "sub" / "data.txt").lstat().st_mode) == 0o640
    assert (tree / "sub" / "data.txt").lstat().st_gid == gid
    assert stat.S_IMODE((tree / "run.sh").lstat().st_mode) == 0o750
    assert stat.S_IMODE(manifest.lstat().st_mode) == 0o440
    assert manifest.lstat().st_gid == gid
    assert (tree / "link").is_symlink()


def test_shared_root_symlink_is_rejected_without_mutating_target(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    before = victim.stat()
    root = tmp_path / "ingress"
    root.symlink_to(victim, target_is_directory=True)

    with pytest.raises(WorkspaceIngressProblem):
        LocalWorkspaceIngress(root, shared_gid=os.getgid())

    after = victim.stat()
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert after.st_gid == before.st_gid


def test_stage_rejects_symlinked_job_without_mutating_target(tmp_path):
    source = _source_tree(tmp_path)
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root, shared_gid=os.getgid())
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    before = victim.stat()
    job_id = uuid4()
    (root / str(job_id)).symlink_to(victim, target_is_directory=True)

    with pytest.raises(WorkspaceIngressProblem):
        ingress.stage(job_id, 1, source)

    after = victim.stat()
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert after.st_gid == before.st_gid


def _broker_side(root, markers):
    return LocalWorkspaceIngress(
        root,
        shared_gid=os.getgid(),
        expected_stage_uid=os.getuid(),
        marker_root=markers,
    )


def test_materialize_with_expected_uid_and_marker_root(tmp_path):
    source = _source_tree(tmp_path)
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    markers = tmp_path / "markers"
    stage_ingress = LocalWorkspaceIngress(root, shared_gid=os.getgid())
    job_id = uuid4()
    stage_ingress.stage(job_id, 1, source)

    broker = _broker_side(root, markers)
    destination = tmp_path / "workspace"
    destination.mkdir(mode=0o700)
    identity = _identity(job_id)
    broker.materialize(identity, destination)

    assert (destination / "sub" / "data.txt").read_text() == "payload"
    assert (destination / "link").is_symlink()
    # The consumed marker lands under marker_root, never inside the staged tree.
    marker = markers / str(job_id) / "1" / "consumed-operation"
    assert marker.read_text() == str(identity.operation_id)
    assert not (root / str(job_id) / "1" / "consumed-operation").exists()

    # Replay with the same operation id → no-op.
    broker.materialize(identity, destination)
    # A different operation id → raises.
    with pytest.raises(WorkspaceIngressProblem):
        broker.materialize(_identity(job_id), destination)


def test_marker_root_discard_is_marker_only(tmp_path):
    source = _source_tree(tmp_path)
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    markers = tmp_path / "markers"
    stage_ingress = LocalWorkspaceIngress(root, shared_gid=os.getgid())
    job_id = uuid4()
    stage_ingress.stage(job_id, 1, source)

    broker = _broker_side(root, markers)
    broker.discard(_identity(job_id, 1))

    # The app-owned tree survives; only a discard marker is written.
    assert (root / str(job_id) / "1" / "tree").exists()
    assert (markers / str(job_id) / "1" / "discarded").exists()


def test_marker_first_replay_survives_pruned_tree(tmp_path):
    source = _source_tree(tmp_path)
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    markers = tmp_path / "markers"
    stage_ingress = LocalWorkspaceIngress(root, shared_gid=os.getgid())
    job_id = uuid4()
    stage_ingress.stage(job_id, 1, source)

    broker = _broker_side(root, markers)
    destination = tmp_path / "workspace"
    destination.mkdir(mode=0o700)
    identity = _identity(job_id)
    broker.materialize(identity, destination)

    # The producer prunes the staged tree after launch.
    stage_ingress.prune(job_id, 1)
    assert not (root / str(job_id)).exists()

    # Same operation id → marker short-circuits with no tree needed.
    broker.materialize(identity, destination)
    # A different operation id → raises (consumed by another operation).
    with pytest.raises(WorkspaceIngressProblem):
        broker.materialize(_identity(job_id), destination)


def test_marker_root_requires_tree_when_unconsumed(tmp_path):
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    LocalWorkspaceIngress(root, shared_gid=os.getgid())
    markers = tmp_path / "markers"
    broker = _broker_side(root, markers)
    destination = tmp_path / "workspace"
    destination.mkdir(mode=0o700)
    # No marker and no staged tree → raise.
    with pytest.raises(WorkspaceIngressProblem):
        broker.materialize(_identity(uuid4()), destination)


def test_broker_rejects_group_writable_stage_root(tmp_path):
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    LocalWorkspaceIngress(root, shared_gid=os.getgid())
    root.chmod(0o2770)

    with pytest.raises(WorkspaceIngressProblem):
        _broker_side(root, tmp_path / "markers")


@pytest.mark.parametrize(
    ("relative", "mode"),
    [
        (Path("manifest.json"), 0o640),
        (Path("tree/sub"), 0o2770),
        (Path("tree/sub/data.txt"), 0o660),
    ],
)
def test_broker_rejects_writable_staged_entries(tmp_path, relative, mode):
    source = _source_tree(tmp_path)
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    stage_ingress = LocalWorkspaceIngress(root, shared_gid=os.getgid())
    job_id = uuid4()
    stage_ingress.stage(job_id, 1, source)
    generation_root = root / str(job_id) / "1"
    (generation_root / relative).chmod(mode)

    broker = _broker_side(root, tmp_path / "markers")
    destination = tmp_path / "workspace"
    destination.mkdir(mode=0o700)

    with pytest.raises(WorkspaceIngressProblem):
        broker.materialize(_identity(job_id), destination)


def test_broker_rejects_symlinked_job_ancestor(tmp_path):
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    LocalWorkspaceIngress(root, shared_gid=os.getgid())
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    job_id = uuid4()
    (root / str(job_id)).symlink_to(victim, target_is_directory=True)

    broker = _broker_side(root, tmp_path / "markers")
    destination = tmp_path / "workspace"
    destination.mkdir(mode=0o700)

    with pytest.raises(WorkspaceIngressProblem):
        broker.materialize(_identity(job_id), destination)


def test_prune_and_prune_stale(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_text("x")
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)

    job_id = uuid4()
    ingress.stage(job_id, 1, source)
    ingress.prune(job_id, 1)
    assert not (root / str(job_id)).exists()

    stale = uuid4()
    ingress.stage(stale, 1, source)
    assert ingress.prune_stale(max_age_seconds=0) == 1
    assert not (root / str(stale)).exists()

    fresh = uuid4()
    ingress.stage(fresh, 1, source)
    assert ingress.prune_stale(max_age_seconds=3600) == 0
    assert (root / str(fresh) / "1").exists()


def test_prune_rejects_symlinked_job_without_deleting_target(tmp_path):
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)
    victim = tmp_path / "victim"
    generation = victim / "1"
    generation.mkdir(parents=True)
    (generation / "keep.txt").write_text("keep")
    job_id = uuid4()
    (root / str(job_id)).symlink_to(victim, target_is_directory=True)

    with pytest.raises(WorkspaceIngressProblem):
        ingress.prune(job_id, 1)

    assert (generation / "keep.txt").read_text() == "keep"


def test_prune_stale_skips_symlinked_job_without_deleting_target(tmp_path):
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)
    victim = tmp_path / "victim"
    generation = victim / "1"
    generation.mkdir(parents=True)
    (generation / "manifest.json").write_text("{}")
    (generation / "keep.txt").write_text("keep")
    (root / str(uuid4())).symlink_to(victim, target_is_directory=True)

    assert ingress.prune_stale(max_age_seconds=0) == 0
    assert (generation / "keep.txt").read_text() == "keep"


def _delegating_open(real_open, name, *, divert=None):
    """Patch os.open to sabotage a single dir_fd-relative open of *name*."""

    def patched(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None and path == name:
            result = divert(path, flags, mode)
            if result is not None:
                return result
        if dir_fd is not None:
            return real_open(path, flags, mode, dir_fd=dir_fd)
        return real_open(path, flags, mode)

    return patched


def test_traversal_preserves_symlink_but_rejects_directory_swap(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "realdir").mkdir()
    (source / "realdir" / "inner").write_text("x")
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)

    real_open = os.open

    def divert(path, flags, mode):
        if flags & getattr(os, "O_DIRECTORY", 0):
            raise OSError(errno.ELOOP, "directory became a symlink")
        return None

    monkeypatch.setattr(
        wi_module.os, "open", _delegating_open(real_open, "realdir", divert=divert)
    )
    with pytest.raises(WorkspaceIngressProblem):
        ingress.stage(uuid4(), 1, source)


def test_traversal_rejects_regular_file_swapped_to_symlink(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "victim.bin").write_text("data")
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)

    real_open = os.open

    def divert(path, flags, mode):
        if not (flags & os.O_CREAT):
            raise OSError(errno.ELOOP, "file became a symlink")
        return None

    monkeypatch.setattr(
        wi_module.os, "open", _delegating_open(real_open, "victim.bin", divert=divert)
    )
    with pytest.raises(WorkspaceIngressProblem):
        ingress.stage(uuid4(), 1, source)


def test_traversal_rejects_inode_swap_between_scan_and_open(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "victim.bin").write_text("original")
    decoy = tmp_path / "decoy.bin"
    decoy.write_text("a different inode entirely")
    root = tmp_path / "ingress"
    root.mkdir(mode=0o700)
    ingress = LocalWorkspaceIngress(root)

    real_open = os.open

    def divert(path, flags, mode):
        if not (flags & os.O_CREAT):
            # Substitute a different real file → different (st_dev, st_ino).
            return real_open(str(decoy), flags & ~getattr(os, "O_NOFOLLOW", 0))
        return None

    monkeypatch.setattr(
        wi_module.os, "open", _delegating_open(real_open, "victim.bin", divert=divert)
    )
    with pytest.raises(WorkspaceIngressProblem):
        ingress.stage(uuid4(), 1, source)
