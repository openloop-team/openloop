import os
import shutil
import stat
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from openloop.broker_runtime import (
    OpenHandsGenerationSpec,
    RuntimeIdentityConflict,
    RuntimeUnavailable,
)
from openloop.broker_runtime.docker_policy import (
    DockerGenerationPolicy,
    DockerRuntimeConfig,
)
from openloop.broker_runtime.filesystem import (
    generation_filesystem_observation,
    prepare_generation_filesystem,
    release_generation_filesystem,
)


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
JOB_ID = UUID("22222222-2222-4222-8222-222222222222")


@pytest.fixture
def short_root():
    # UDS paths have a 100-byte profile budget; pytest's macOS temp path is
    # intentionally much longer than any supported broker runtime root.
    root = Path(tempfile.mkdtemp(prefix="olrt-", dir="/private/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _spec():
    return OpenHandsGenerationSpec(
        operation_id=UUID("11111111-1111-4111-8111-111111111111"),
        job_id=JOB_ID,
        conversation_id=UUID("33333333-3333-4333-8333-333333333333"),
        generation=1,
        deadline=NOW + timedelta(minutes=5),
        relay_capability="r" * 43,
        session_api_key="s" * 43,
        conversation_secret="c" * 43,
    )


def _policy(tmp_path: Path):
    runtime = tmp_path / "r"
    state = tmp_path / "s"
    runtime.mkdir(mode=0o700)
    state.mkdir(mode=0o700)
    config = DockerRuntimeConfig(
        runtime,
        state,
        platform="linux/arm64",
        uid=os.getuid(),
        gid=os.getgid(),
    )
    return DockerGenerationPolicy.build(config, _spec())


def test_prepare_installs_exact_artifacts_and_replays(short_root):
    policy = _policy(short_root)
    prepare_generation_filesystem(
        policy.paths, policy.compiled_relay, uid=os.getuid()
    )
    prepare_generation_filesystem(
        policy.paths, policy.compiled_relay, uid=os.getuid()
    )

    assert frozenset(path.name for path in policy.paths.root.iterdir()) == {
        "relay",
        "socket",
        "workspace",
    }
    assert (policy.paths.artifacts / "haproxy.cfg").read_bytes() == (
        policy.compiled_relay.haproxy_config
    )
    assert (policy.paths.artifacts / "relay-capability").read_bytes() == (
        policy.compiled_relay.capability_file.payload
    )
    for path in policy.paths.artifacts.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert generation_filesystem_observation(
        policy.paths, uid=os.getuid()
    ) == (True, True)


def test_prepare_rejects_missing_or_unsafe_configured_root(short_root):
    runtime = short_root / "missing"
    state = short_root / "state"
    state.mkdir(mode=0o700)
    config = DockerRuntimeConfig(
        runtime,
        state,
        platform="linux/arm64",
        uid=os.getuid(),
        gid=os.getgid(),
    )
    policy = DockerGenerationPolicy.build(config, _spec())
    with pytest.raises(RuntimeUnavailable, match="runtime root.*does not exist"):
        prepare_generation_filesystem(
            policy.paths, policy.compiled_relay, uid=os.getuid()
        )


def test_prepare_rejects_symlinked_generation_component(short_root):
    policy = _policy(short_root)
    job = policy.paths.root.parent
    job.mkdir(mode=0o700)
    target = short_root / "target"
    target.mkdir(mode=0o700)
    policy.paths.root.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeIdentityConflict, match="safe directory"):
        prepare_generation_filesystem(
            policy.paths, policy.compiled_relay, uid=os.getuid()
        )


def test_prepare_rejects_existing_artifact_content_or_extra_entry(short_root):
    policy = _policy(short_root)
    prepare_generation_filesystem(
        policy.paths, policy.compiled_relay, uid=os.getuid()
    )
    capability = policy.paths.artifacts / "relay-capability"
    capability.chmod(0o600)
    capability.write_text("z" * 43 + "\n")
    capability.chmod(0o400)

    with pytest.raises(RuntimeIdentityConflict, match="content does not match"):
        prepare_generation_filesystem(
            policy.paths, policy.compiled_relay, uid=os.getuid()
        )


def test_release_removes_generation_but_preserves_durable_state(short_root):
    policy = _policy(short_root)
    prepare_generation_filesystem(
        policy.paths, policy.compiled_relay, uid=os.getuid()
    )
    (policy.paths.workspace / "generated.txt").write_text("data")
    (policy.paths.state / "conversation.json").write_text("durable")

    release_generation_filesystem(policy.paths, uid=os.getuid())
    release_generation_filesystem(policy.paths, uid=os.getuid())

    assert not policy.paths.root.exists()
    assert (policy.paths.state / "conversation.json").read_text() == "durable"


def test_release_refuses_unknown_generation_entry(short_root):
    policy = _policy(short_root)
    prepare_generation_filesystem(
        policy.paths, policy.compiled_relay, uid=os.getuid()
    )
    (policy.paths.root / "foreign").write_text("do not delete")

    with pytest.raises(RuntimeIdentityConflict, match="unknown root entry"):
        release_generation_filesystem(policy.paths, uid=os.getuid())
    assert (policy.paths.root / "foreign").read_text() == "do not delete"
