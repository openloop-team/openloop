import os
import stat
from uuid import UUID

import pytest

from openloop.broker_control.durable import (
    LocalDurableStateAdapter,
    LocalDurableStateProblem,
)
from openloop.broker_control.development import (
    local_durable_adapter_for_docker,
    validate_local_durable_binding,
)
from openloop.broker_runtime import DockerRuntimeConfig


JOB_ID = UUID("00000000-0000-4000-8000-000000000401")
DIGEST = "d" * 64


def _trusted_root(tmp_path):
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _adapter(tmp_path):
    return LocalDurableStateAdapter(
        state_root=_trusted_root(tmp_path),
        uid=os.getuid(),
        gid=os.getgid(),
    )


async def test_local_durable_descriptor_is_opaque_and_ensure_preserves_state(tmp_path):
    adapter = _adapter(tmp_path)
    descriptor = adapter.describe(JOB_ID, "runtime-v1", DIGEST)

    assert descriptor.durable_state_ref == f"local-openhands:v1:{JOB_ID}"
    rendered = repr(descriptor)
    assert descriptor.durable_state_ref not in rendered
    assert descriptor.durable_digest not in rendered
    assert str(adapter.binding.state_root) not in repr(adapter.binding)

    await adapter.ensure(descriptor)
    state = adapter.binding.state_root / str(JOB_ID) / "agent-server"
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    (state / "conversation.json").write_text("durable")

    await adapter.ensure(descriptor)
    assert (state / "conversation.json").read_text() == "durable"


async def test_local_durable_rejects_symlinked_job_or_state_directory(tmp_path):
    adapter = _adapter(tmp_path)
    descriptor = adapter.describe(JOB_ID, "runtime-v1", DIGEST)
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    (adapter.binding.state_root / str(JOB_ID)).symlink_to(
        target,
        target_is_directory=True,
    )

    with pytest.raises(LocalDurableStateProblem):
        await adapter.ensure(descriptor)
    assert list(target.iterdir()) == []

    job_link = adapter.binding.state_root / str(JOB_ID)
    job_link.unlink()
    job_link.mkdir(mode=0o700)
    (job_link / "agent-server").symlink_to(target, target_is_directory=True)
    with pytest.raises(LocalDurableStateProblem):
        await adapter.ensure(descriptor)
    assert list(target.iterdir()) == []


def test_local_durable_rejects_unsafe_or_symlinked_root(tmp_path):
    root = _trusted_root(tmp_path)
    root.chmod(0o755)
    with pytest.raises(LocalDurableStateProblem):
        LocalDurableStateAdapter(
            state_root=root,
            uid=os.getuid(),
            gid=os.getgid(),
        )

    root.chmod(0o700)
    link = tmp_path / "state-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(LocalDurableStateProblem):
        LocalDurableStateAdapter(
            state_root=link,
            uid=os.getuid(),
            gid=os.getgid(),
        )


def test_development_docker_binding_uses_one_config_and_rejects_mismatch(tmp_path):
    runtime = tmp_path / "runtime"
    state = tmp_path / "state"
    other = tmp_path / "other"
    for path in (runtime, state, other):
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    config = DockerRuntimeConfig(
        runtime,
        state,
        platform="linux/arm64",
        uid=os.getuid(),
        gid=os.getgid(),
    )
    adapter = local_durable_adapter_for_docker(config)
    validate_local_durable_binding(adapter, config)

    mismatched = LocalDurableStateAdapter(
        state_root=other,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    with pytest.raises(LocalDurableStateProblem):
        validate_local_durable_binding(mismatched, config)
