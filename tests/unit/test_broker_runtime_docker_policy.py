from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from openloop.broker_runtime import OpenHandsGenerationSpec, RuntimeUnavailable
from openloop.broker_runtime.docker_policy import (
    AGENT_COMMAND,
    AGENT_MEMORY_BYTES,
    DockerGenerationPolicy,
    DockerRuntimeConfig,
    LABEL_DEADLINE,
    LABEL_GENERATION,
    LABEL_JOB,
    LABEL_OPERATION,
    LABEL_PROFILE,
    LABEL_ROLE,
    LABEL_SCHEMA,
    RELAY_COMMAND,
    RUNTIME_PROFILE,
    RUNTIME_SCHEMA,
    image_contract_commands,
)
from openloop.tools.openhands_docker import runtime_server_image
from openloop.tools.openhands_relay import (
    CONTAINER_RELAY_CAPABILITY_FILE,
    CONTAINER_RELAY_CONFIG_FILE,
    RelayMode,
)


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")
JOB_ID = UUID("22222222-2222-4222-8222-222222222222")
CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")
RELAY_CAPABILITY = "r" * 43
SESSION_KEY = "s" * 43
CONVERSATION_SECRET = "c" * 43


def _spec(**changes):
    values = {
        "operation_id": OPERATION_ID,
        "job_id": JOB_ID,
        "conversation_id": CONVERSATION_ID,
        "generation": 7,
        "deadline": NOW + timedelta(minutes=5),
        "relay_capability": RELAY_CAPABILITY,
        "session_api_key": SESSION_KEY,
        "conversation_secret": CONVERSATION_SECRET,
    }
    values.update(changes)
    return OpenHandsGenerationSpec(**values)


def _config(tmp_path: Path, **changes):
    values = {
        # Docker UDS paths have a tight kernel byte limit. Keep pure command
        # tests independent of pytest's intentionally long temporary root.
        "runtime_root": Path("/tmp/olrt"),
        "state_root": Path("/tmp/olst"),
        "platform": "linux/arm64",
        "uid": 1234,
        "gid": 1235,
    }
    values.update(changes)
    return DockerRuntimeConfig(**values)


def _policy(tmp_path: Path):
    return DockerGenerationPolicy.build(_config(tmp_path), _spec())


def _label_values(argv):
    return {
        argv[index + 1].split("=", 1)[0]: argv[index + 1].split("=", 1)[1]
        for index, value in enumerate(argv)
        if value == "--label"
    }


def test_config_requires_disjoint_trusted_absolute_roots(tmp_path):
    with pytest.raises(ValueError, match="absolute Path"):
        DockerRuntimeConfig(Path("relative"), tmp_path / "state")
    with pytest.raises(ValueError, match="disjoint"):
        DockerRuntimeConfig(tmp_path, tmp_path / "nested")
    with pytest.raises(ValueError, match="unsupported character"):
        DockerRuntimeConfig(tmp_path / "run,time", tmp_path / "state")


@pytest.mark.parametrize("shared_gid", [True, -1, 2**31])
def test_config_rejects_invalid_shared_gid(tmp_path, shared_gid):
    with pytest.raises(ValueError, match="shared_gid is out of range"):
        _config(tmp_path, shared_gid=shared_gid)


def test_config_accepts_numeric_shared_gid(tmp_path):
    assert _config(tmp_path, shared_gid=4321).shared_gid == 4321


def test_policy_derives_names_paths_and_fixed_running_relay(tmp_path):
    policy = _policy(tmp_path)
    stem = "ol-oh-22222222222242228222222222222222-g7"
    assert policy.names.network == f"{stem}-net"
    assert policy.names.agent == f"{stem}-agent"
    assert policy.names.relay == f"{stem}-relay"
    assert policy.paths.root == (
        Path("/tmp/olrt").resolve() / str(JOB_ID) / "7"
    )
    assert policy.paths.state == (
        Path("/tmp/olst").resolve() / str(JOB_ID) / "agent-server"
    )
    assert policy.compiled_relay.endpoint.mode.value == "running"
    rendered = repr(policy)
    for secret in (RELAY_CAPABILITY, SESSION_KEY, CONVERSATION_SECRET):
        assert secret not in rendered


def test_policy_can_compile_only_the_fixed_checkpoint_relay_variant(tmp_path):
    policy = DockerGenerationPolicy.build(
        _config(tmp_path), _spec(), mode=RelayMode.CHECKPOINT
    )

    assert policy.compiled_relay.endpoint.mode is RelayMode.CHECKPOINT
    assert b"path_archive" in policy.compiled_relay.haproxy_config
    assert b"path_websocket" in policy.compiled_relay.haproxy_config


def test_policy_refuses_host_socket_over_uds_budget(tmp_path):
    long_root = tmp_path / ("x" * 80)
    with pytest.raises(RuntimeUnavailable, match="UDS budget"):
        DockerGenerationPolicy.build(
            _config(tmp_path, runtime_root=long_root), _spec()
        )


def test_network_command_carries_complete_immutable_identity(tmp_path):
    policy = _policy(tmp_path)
    command = policy.network_create()
    labels = _label_values(command.argv)
    assert command.argv[:4] == ("docker", "network", "create", "--driver")
    assert labels == {
        LABEL_SCHEMA: RUNTIME_SCHEMA,
        LABEL_PROFILE: RUNTIME_PROFILE,
        LABEL_OPERATION: str(OPERATION_ID),
        LABEL_JOB: str(JOB_ID),
        LABEL_GENERATION: "7",
        LABEL_ROLE: "network",
        LABEL_DEADLINE: "1784376300",
    }
    assert command.argv[-1] == policy.names.network


def test_agent_command_is_fixed_hardened_secret_free_and_self_expiring(tmp_path):
    policy = _policy(tmp_path)
    command = policy.agent_create()
    argv = command.argv
    labels = _label_values(argv)
    assert labels[LABEL_ROLE] == "agent"
    assert "--init" not in argv
    assert "-p" not in argv and "--publish" not in argv
    assert argv[argv.index("--entrypoint") + 1] == "/bin/sh"
    script = argv[argv.index("-c") + 1]
    assert "date +%s" in script
    assert "exec timeout" in script
    assert str(_spec().identity.deadline_epoch) in argv
    assert tuple(argv[-len(AGENT_COMMAND) :]) == AGENT_COMMAND
    assert argv[argv.index("--memory") + 1] == str(AGENT_MEMORY_BYTES)
    assert argv[argv.index("--memory-swap") + 1] == str(AGENT_MEMORY_BYTES)
    assert "--read-only" in argv
    assert "--cap-drop" in argv
    assert "no-new-privileges" in argv
    assert "--network-alias" in argv
    assert argv[argv.index("--network-alias") + 1] == "agent"
    assert runtime_server_image(
        policy.config.agent_image, policy.config.platform
    ) in argv
    assert dict(command.environment) == {
        "OH_SESSION_API_KEYS_0": SESSION_KEY,
        "OH_SECRET_KEY": CONVERSATION_SECRET,
    }
    assert "OH_SESSION_API_KEYS_0" in argv
    assert "OH_SECRET_KEY" in argv
    rendered_argv = " ".join(argv)
    for secret in (RELAY_CAPABILITY, SESSION_KEY, CONVERSATION_SECRET):
        assert secret not in rendered_argv
        assert secret not in repr(command)


def test_relay_command_realizes_compiled_policy_and_mounts_secret_read_only(tmp_path):
    policy = _policy(tmp_path)
    command = policy.relay_create()
    argv = command.argv
    labels = _label_values(argv)
    runtime = policy.compiled_relay.runtime
    assert labels[LABEL_ROLE] == "relay"
    assert runtime.image in argv
    assert argv[argv.index("--platform") + 1] == policy.config.platform
    assert "--read-only" in argv
    assert argv[argv.index("--memory") + 1] == str(runtime.memory_bytes)
    assert argv[argv.index("--pids-limit") + 1] == str(runtime.pids_limit)
    assert "--init" not in argv
    assert "-p" not in argv and "--publish" not in argv
    assert tuple(argv[-len(RELAY_COMMAND) :]) == RELAY_COMMAND
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert any("dst=/run/openloop/config,readonly" in mount for mount in mounts)
    assert any("dst=/run/openloop/secrets,readonly" in mount for mount in mounts)
    assert any(
        f"dst={policy.compiled_relay.endpoint.socket_path.parent}" in mount
        and "readonly" not in mount
        for mount in mounts
    )
    assert CONTAINER_RELAY_CONFIG_FILE in argv
    assert CONTAINER_RELAY_CAPABILITY_FILE not in " ".join(argv)
    for secret in (RELAY_CAPABILITY, SESSION_KEY, CONVERSATION_SECRET):
        assert secret not in " ".join(argv)


def test_start_refuses_foreign_name(tmp_path):
    policy = _policy(tmp_path)
    object_id = "a" * 64
    assert policy.start(policy.names.agent, object_id).argv == (
        "docker", "start", object_id
    )
    with pytest.raises(ValueError, match="foreign"):
        policy.start("someone-elses-container", object_id)
    with pytest.raises(ValueError, match="invalid"):
        policy.start(policy.names.agent, "not-an-object-id")


def test_image_contract_probe_is_fixed_and_non_networked(tmp_path):
    config = _config(tmp_path)
    commands = image_contract_commands(config)
    assert commands[0].argv[:2] == ("docker", "version")
    assert commands[1].argv[:3] == ("docker", "image", "inspect")
    assert commands[2].argv[:3] == ("docker", "image", "inspect")
    for command in commands[3:]:
        assert command.argv[:2] == ("docker", "run")
        assert command.argv[command.argv.index("--platform") + 1] == config.platform
        assert command.argv[command.argv.index("--network") + 1] == "none"
        assert "--read-only" in command.argv
        assert "--cap-drop" in command.argv
        assert "command -v date" in command.argv[-1]
        assert "command -v timeout" in command.argv[-1]
