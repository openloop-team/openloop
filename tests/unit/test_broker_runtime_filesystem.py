import os
import socket
import stat
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
    harden_relay_socket,
    install_checkpoint_relay_config,
    prepare_generation_filesystem,
    relay_artifact_mode,
    release_generation_filesystem,
)
from openloop.tools.openhands_relay import RelayMode


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
JOB_ID = UUID("22222222-2222-4222-8222-222222222222")


@pytest.fixture
def short_root(short_socket_root):
    return short_socket_root


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


def _policy(tmp_path: Path, *, shared_gid: int | None = None):
    runtime = tmp_path / "r"
    state = tmp_path / "s"
    runtime.mkdir(mode=0o750 if shared_gid is not None else 0o700)
    if shared_gid is not None:
        os.chown(runtime, -1, shared_gid)
        runtime.chmod(0o750)
    state.mkdir(mode=0o700)
    config = DockerRuntimeConfig(
        runtime,
        state,
        platform="linux/arm64",
        uid=os.getuid(),
        gid=os.getgid(),
        shared_gid=shared_gid,
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


def test_harden_relay_socket_validates_identity_and_sets_owner_only_mode(short_root):
    policy = _policy(short_root)
    prepare_generation_filesystem(
        policy.paths, policy.compiled_relay, uid=os.getuid()
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(os.fspath(policy.paths.host_socket))

        harden_relay_socket(policy.paths, uid=os.getuid())

        assert stat.S_IMODE(policy.paths.host_socket.stat().st_mode) == 0o600
    finally:
        listener.close()


def test_prepare_shared_relay_chain_keeps_private_leaves_owner_only(short_root):
    policy = _policy(short_root, shared_gid=os.getgid())

    prepare_generation_filesystem(
        policy.paths,
        policy.compiled_relay,
        uid=os.getuid(),
        shared_gid=policy.config.shared_gid,
    )

    shared_chain = (
        policy.config.runtime_root,
        policy.paths.root.parent,
        policy.paths.root,
        policy.paths.socket,
    )
    for path in shared_chain:
        info = path.stat()
        assert stat.S_IMODE(info.st_mode) == 0o750
        assert info.st_gid == os.getgid()

    private_leaves = (
        policy.paths.artifacts,
        policy.paths.workspace,
        policy.config.state_root,
        policy.paths.state.parent,
        policy.paths.state,
    )
    for path in private_leaves:
        assert stat.S_IMODE(path.stat().st_mode) == 0o700


@pytest.mark.parametrize("component", ["runtime", "job", "generation", "socket"])
def test_prepare_shared_relay_chain_rejects_owner_only_component(
    short_root, component
):
    policy = _policy(short_root, shared_gid=os.getgid())
    prepare_generation_filesystem(
        policy.paths,
        policy.compiled_relay,
        uid=os.getuid(),
        shared_gid=policy.config.shared_gid,
    )
    selected = {
        "runtime": policy.config.runtime_root,
        "job": policy.paths.root.parent,
        "generation": policy.paths.root,
        "socket": policy.paths.socket,
    }[component]
    selected.chmod(0o700)

    with pytest.raises(RuntimeIdentityConflict, match="mode does not match"):
        prepare_generation_filesystem(
            policy.paths,
            policy.compiled_relay,
            uid=os.getuid(),
            shared_gid=policy.config.shared_gid,
        )


def test_prepare_shared_relay_chain_rejects_wrong_gid(short_root):
    policy = _policy(short_root, shared_gid=os.getgid())

    with pytest.raises(RuntimeIdentityConflict, match="group does not match"):
        prepare_generation_filesystem(
            policy.paths,
            policy.compiled_relay,
            uid=os.getuid(),
            shared_gid=os.getgid() + 1,
        )


def test_prepare_owner_only_rejects_group_traversable_runtime_root(short_root):
    policy = _policy(short_root)
    policy.config.runtime_root.chmod(0o750)

    with pytest.raises(RuntimeIdentityConflict, match="mode does not match"):
        prepare_generation_filesystem(
            policy.paths, policy.compiled_relay, uid=os.getuid()
        )


def test_harden_relay_socket_sets_shared_group_and_mode(short_root):
    policy = _policy(short_root, shared_gid=os.getgid())
    prepare_generation_filesystem(
        policy.paths,
        policy.compiled_relay,
        uid=os.getuid(),
        shared_gid=policy.config.shared_gid,
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(os.fspath(policy.paths.host_socket))

        harden_relay_socket(
            policy.paths,
            uid=os.getuid(),
            shared_gid=policy.config.shared_gid,
        )

        info = policy.paths.host_socket.stat()
        assert stat.S_IMODE(info.st_mode) == 0o660
        assert info.st_gid == os.getgid()
    finally:
        listener.close()


def test_checkpoint_transition_replaces_only_config_and_replays(short_root):
    running = _policy(short_root)
    checkpoint = DockerGenerationPolicy.build(
        running.config, _spec(), mode=RelayMode.CHECKPOINT
    )
    prepare_generation_filesystem(
        running.paths, running.compiled_relay, uid=os.getuid()
    )
    capability_before = (
        running.paths.artifacts / "relay-capability"
    ).read_bytes()

    install_checkpoint_relay_config(
        running.paths,
        running.compiled_relay,
        checkpoint.compiled_relay,
        uid=os.getuid(),
    )
    install_checkpoint_relay_config(
        running.paths,
        running.compiled_relay,
        checkpoint.compiled_relay,
        uid=os.getuid(),
    )

    assert relay_artifact_mode(
        running.paths,
        running.compiled_relay,
        checkpoint.compiled_relay,
        uid=os.getuid(),
    ) is RelayMode.CHECKPOINT
    assert (running.paths.artifacts / "haproxy.cfg").read_bytes() == (
        checkpoint.compiled_relay.haproxy_config
    )
    assert (
        running.paths.artifacts / "relay-capability"
    ).read_bytes() == capability_before
    assert stat.S_IMODE(
        (running.paths.artifacts / "haproxy.cfg").stat().st_mode
    ) == 0o400


def test_checkpoint_transition_recovers_owned_partial_temp_file(short_root):
    running = _policy(short_root)
    checkpoint = DockerGenerationPolicy.build(
        running.config, _spec(), mode=RelayMode.CHECKPOINT
    )
    prepare_generation_filesystem(
        running.paths, running.compiled_relay, uid=os.getuid()
    )
    temporary = running.paths.artifacts / ".haproxy.cfg.checkpoint"
    temporary.write_bytes(b"interrupted")
    temporary.chmod(0o600)

    install_checkpoint_relay_config(
        running.paths,
        running.compiled_relay,
        checkpoint.compiled_relay,
        uid=os.getuid(),
    )

    assert not temporary.exists()
    assert relay_artifact_mode(
        running.paths,
        running.compiled_relay,
        checkpoint.compiled_relay,
        uid=os.getuid(),
    ) is RelayMode.CHECKPOINT


def test_checkpoint_transition_rejects_unsafe_temp_artifact(short_root):
    running = _policy(short_root)
    checkpoint = DockerGenerationPolicy.build(
        running.config, _spec(), mode=RelayMode.CHECKPOINT
    )
    prepare_generation_filesystem(
        running.paths, running.compiled_relay, uid=os.getuid()
    )
    temporary = running.paths.artifacts / ".haproxy.cfg.checkpoint"
    temporary.symlink_to(running.paths.artifacts / "haproxy.cfg")

    with pytest.raises(RuntimeIdentityConflict):
        install_checkpoint_relay_config(
            running.paths,
            running.compiled_relay,
            checkpoint.compiled_relay,
            uid=os.getuid(),
        )


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
