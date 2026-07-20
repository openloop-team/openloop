from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

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
