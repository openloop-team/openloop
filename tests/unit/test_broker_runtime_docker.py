import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from openloop.broker_runtime import (
    CommandExecution,
    DockerOpenHandsRuntimeDriver,
    DockerRuntimeConfig,
    OpenHandsGenerationSpec,
    RuntimeExpired,
    RuntimeHealthFailure,
    RuntimeIdentityConflict,
    RuntimeResourceState,
    RuntimeUnavailable,
)
from openloop.broker_runtime.docker_policy import (
    AGENT_CPUS,
    AGENT_MEMORY_BYTES,
    AGENT_PIDS_LIMIT,
    EXPECTED_AGENT_ENTRYPOINT,
    EXPECTED_RELAY_ENTRYPOINT,
    LABEL_ROLE,
    derive_generation_names,
)
from openloop.tools.openhands_relay import RelayMode


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")
JOB_ID = UUID("22222222-2222-4222-8222-222222222222")
CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")


@pytest.fixture
def short_root():
    root = Path(tempfile.mkdtemp(prefix="olrd-", dir="/private/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _spec(**changes):
    values = {
        "operation_id": OPERATION_ID,
        "job_id": JOB_ID,
        "conversation_id": CONVERSATION_ID,
        "generation": 1,
        "deadline": NOW + timedelta(minutes=5),
        "relay_capability": "r" * 43,
        "session_api_key": "s" * 43,
        "conversation_secret": "c" * 43,
    }
    values.update(changes)
    return OpenHandsGenerationSpec(**values)


def _config(short_root):
    runtime = short_root / "r"
    state = short_root / "s"
    runtime.mkdir(mode=0o700)
    state.mkdir(mode=0o700)
    return DockerRuntimeConfig(
        runtime,
        state,
        platform="linux/arm64",
        uid=os.getuid(),
        gid=os.getgid(),
    )


def _labels(argv):
    return {
        argv[index + 1].split("=", 1)[0]: argv[index + 1].split("=", 1)[1]
        for index, value in enumerate(argv)
        if value == "--label"
    }


class FakeDocker:
    def __init__(self, config):
        self.config = config
        self.calls = []
        self.networks = {}
        self.containers = {}
        self.agent_entrypoint = EXPECTED_AGENT_ENTRYPOINT
        self.relay_entrypoint = EXPECTED_RELAY_ENTRYPOINT
        self.agent_platform = config.platform
        self.relay_platform = config.platform
        self.create_races = set()
        self.network_create_race = False
        self.fail_remove_roles = set()

    def _container_name(self, identifier):
        if identifier in self.containers:
            return identifier
        for name, document in self.containers.items():
            if document["Id"] == identifier:
                return name
        return None

    def _network_name(self, identifier):
        if identifier in self.networks:
            return identifier
        for name, document in self.networks.items():
            if document["Id"] == identifier:
                return name
        return None

    @staticmethod
    def _mounts(argv):
        mounts = []
        for index, value in enumerate(argv):
            if value != "--mount":
                continue
            fields = {}
            for item in argv[index + 1].split(","):
                if "=" in item:
                    name, field_value = item.split("=", 1)
                    fields[name] = field_value
                else:
                    fields[item] = True
            mounts.append(
                {
                    "Type": fields["type"],
                    "Source": fields["src"],
                    "Destination": fields["dst"],
                    "RW": "readonly" not in fields,
                }
            )
        return mounts

    @staticmethod
    def _environment(command):
        values = dict(command.environment)
        for index, value in enumerate(command.argv):
            if value != "--env":
                continue
            setting = command.argv[index + 1]
            if "=" in setting:
                name, contents = setting.split("=", 1)
                values[name] = contents
        return [f"{key}={value}" for key, value in values.items()]

    @staticmethod
    def _tmpfs(argv):
        values = {}
        for index, value in enumerate(argv):
            if value != "--tmpfs":
                continue
            path, options = argv[index + 1].split(":", 1)
            values[path] = options
        return values

    def _container_document(self, command):
        argv = command.argv
        name = argv[argv.index("--name") + 1]
        labels = _labels(argv)
        role = labels[LABEL_ROLE]
        image = (
            self.config.resolved_agent_image
            if role == "agent"
            else self.config.relay_image
        )
        image_index = argv.index(image)
        memory = (
            AGENT_MEMORY_BYTES
            if role == "agent"
            else 64 * 1024 * 1024
        )
        pids = AGENT_PIDS_LIMIT if role == "agent" else 64
        return {
            "Id": ("b" if role == "agent" else "c") * 64,
            "Name": f"/{name}",
            "Platform": "linux",
            "Config": {
                "Labels": labels,
                "Image": image,
                "User": f"{self.config.uid}:{self.config.gid}",
                "Entrypoint": ["/bin/sh"],
                "Cmd": list(argv[image_index + 1 :]),
                # Secrets deliberately exist in Docker configuration but this
                # raw document never crosses the driver observation boundary.
                "Env": self._environment(command),
            },
            "HostConfig": {
                "NetworkMode": argv[argv.index("--network") + 1],
                "ReadonlyRootfs": "--read-only" in argv,
                "CapDrop": [argv[argv.index("--cap-drop") + 1]],
                "CapAdd": None,
                "SecurityOpt": ["no-new-privileges:true"],
                "PortBindings": None,
                "PublishAllPorts": False,
                "Privileged": False,
                "AutoRemove": False,
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                "Devices": [],
                "DeviceRequests": None,
                "PidMode": "",
                "IpcMode": "private",
                "UTSMode": "",
                "Tmpfs": self._tmpfs(argv),
                "Memory": memory,
                "MemorySwap": memory,
                "PidsLimit": pids,
                "NanoCpus": int(AGENT_CPUS * 1_000_000_000) if role == "agent" else 0,
            },
            "Mounts": self._mounts(argv),
            "State": {"Status": "created"},
            "NetworkSettings": {
                "Ports": {},
                "Networks": {
                    argv[argv.index("--network") + 1]: {
                        "Aliases": [name, "agent"] if role == "agent" else None
                    }
                },
            },
        }

    async def __call__(self, command):
        self.calls.append(command)
        argv = command.argv
        if argv[1] == "version":
            return CommandExecution(0, "27.0\n")
        if argv[1:3] == ("image", "inspect"):
            entrypoint = (
                self.agent_entrypoint
                if argv[3] == self.config.resolved_agent_image
                else self.relay_entrypoint
            )
            platform = (
                self.agent_platform
                if argv[3] == self.config.resolved_agent_image
                else self.relay_platform
            )
            return CommandExecution(0, f"{json.dumps(entrypoint)}|{platform}\n")
        if argv[1] == "run":
            return CommandExecution(0)
        if argv[1:3] == ("network", "inspect"):
            document = self.networks.get(argv[3])
            if document is None:
                return CommandExecution(1, stderr="Error: No such network")
            return CommandExecution(0, json.dumps(document))
        if argv[1:3] == ("network", "ls"):
            return CommandExecution(0, "\n".join(self.networks) + "\n")
        if argv[1:3] == ("network", "create"):
            name = argv[-1]
            self.networks[name] = {
                "Id": "a" * 64,
                "Name": name,
                "Scope": "local",
                "Driver": "bridge",
                "EnableIPv6": False,
                "Internal": False,
                "Attachable": False,
                "Ingress": False,
                "ConfigOnly": False,
                "Options": {},
                "Containers": {},
                "Labels": _labels(argv),
            }
            if self.network_create_race:
                return CommandExecution(1, stderr="network create reply lost")
            return CommandExecution(0, "network-id\n")
        if argv[1:3] == ("network", "rm"):
            name = self._network_name(argv[3])
            if name is not None:
                self.networks.pop(name)
            return CommandExecution(0, argv[3] + "\n")
        if argv[1:3] == ("container", "inspect"):
            document = self.containers.get(argv[3])
            if document is None:
                return CommandExecution(1, stderr="Error: No such container")
            return CommandExecution(0, json.dumps(document))
        if argv[1:3] == ("container", "ls"):
            return CommandExecution(0, "\n".join(self.containers) + "\n")
        if argv[1] == "create":
            document = self._container_document(command)
            self.containers[document["Name"].lstrip("/")] = document
            if document["Config"]["Labels"][LABEL_ROLE] in self.create_races:
                return CommandExecution(1, stderr="name already in use")
            return CommandExecution(0, "container-id\n")
        if argv[1] == "start":
            name = self._container_name(argv[2])
            if name is None:
                return CommandExecution(1, stderr="Error: No such container")
            self.containers[name]["State"]["Status"] = "running"
            return CommandExecution(0, argv[2] + "\n")
        if argv[1] == "stop":
            name = self._container_name(argv[-1])
            if name is None:
                return CommandExecution(1, stderr="Error: No such container")
            self.containers[name]["State"]["Status"] = "exited"
            return CommandExecution(0, name + "\n")
        if argv[1] == "rm":
            name = self._container_name(argv[2])
            document = self.containers.get(name) if name is not None else None
            if (
                document is not None
                and document["Config"]["Labels"][LABEL_ROLE]
                in self.fail_remove_roles
            ):
                return CommandExecution(1, stderr="injected remove failure")
            if name is not None:
                self.containers.pop(name)
            return CommandExecution(0, argv[2] + "\n")
        raise AssertionError(f"unexpected Docker command: {argv!r}")


class Health:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    async def __call__(self, policy):
        self.calls.append(policy)
        if self.failure is not None:
            raise self.failure


def _driver(
    short_root,
    *,
    health=None,
    clock=lambda: NOW,
    socket_hardener=lambda _paths, _uid: None,
):
    config = _config(short_root)
    docker = FakeDocker(config)
    checker = health or Health()
    driver = DockerOpenHandsRuntimeDriver(
        config,
        runner=docker,
        clock=clock,
        health_checker=checker,
        socket_hardener=socket_hardener,
    )
    return driver, docker, checker


def test_command_execution_repr_never_renders_retained_output():
    execution = CommandExecution(
        1,
        stdout="sensitive stdout",
        stderr="sensitive stderr",
    )

    rendered = repr(execution)

    assert "sensitive" not in rendered
    assert "stdout=<bounded text" in rendered


def test_describe_endpoint_is_pure_and_uses_host_socket(short_root):
    driver, docker, health = _driver(short_root)

    endpoint = driver.describe_endpoint(_spec())

    assert driver.maximum_lifetime_seconds == 86_400
    assert endpoint.socket_path == (
        driver.config.runtime_root
        / str(JOB_ID)
        / "1"
        / "socket"
        / "agent.sock"
    )
    assert docker.calls == []
    assert health.calls == []
    assert not endpoint.socket_path.exists()


async def test_ensure_creates_agent_then_relay_and_returns_redacted_result(short_root):
    driver, docker, health = _driver(short_root)
    endpoint = driver.describe_endpoint(_spec())
    result = await driver.ensure(_spec())

    names = derive_generation_names(_spec().identity)
    create_roles = [
        _labels(command.argv)[LABEL_ROLE]
        for command in docker.calls
        if command.argv[1] == "create"
    ]
    assert create_roles == ["agent", "relay"]
    assert docker.containers[names.agent]["State"]["Status"] == "running"
    assert docker.containers[names.relay]["State"]["Status"] == "running"
    assert health.calls
    assert result.observation.complete
    assert result.handle == _spec().identity.opaque_handle
    assert result.endpoint == endpoint
    assert driver._resource_locks == {}
    for secret in ("r" * 43, "s" * 43, "c" * 43):
        assert secret not in repr(result)


async def test_ensure_replays_without_duplicate_resources(short_root):
    driver, docker, health = _driver(short_root)
    await driver.ensure(_spec())
    first_creates = sum(command.argv[1] == "create" for command in docker.calls)

    await driver.ensure(_spec())

    assert sum(command.argv[1] == "create" for command in docker.calls) == first_creates
    assert len(health.calls) == 2


async def test_quiesce_replaces_only_relay_and_replays_checkpoint_mode(short_root):
    driver, docker, health = _driver(short_root)
    await driver.ensure(_spec())
    names = derive_generation_names(_spec().identity)
    agent_before = docker.containers[names.agent]

    first = await driver.quiesce(_spec())
    relay_creates = sum(
        command.argv[1] == "create"
        and _labels(command.argv)[LABEL_ROLE] == "relay"
        for command in docker.calls
    )
    second = await driver.quiesce(_spec())

    assert first.endpoint.mode is RelayMode.CHECKPOINT
    assert second.endpoint == first.endpoint
    assert first.observation.complete
    assert docker.containers[names.agent] is agent_before
    assert docker.containers[names.agent]["State"]["Status"] == "running"
    assert docker.containers[names.relay]["State"]["Status"] == "running"
    assert relay_creates == 2
    assert sum(
        command.argv[1] == "create"
        and _labels(command.argv)[LABEL_ROLE] == "relay"
        for command in docker.calls
    ) == relay_creates
    assert health.calls[-1].compiled_relay.endpoint.mode is RelayMode.CHECKPOINT


async def test_concurrent_ensure_serializes_one_generation_and_reclaims_lock(
    short_root,
):
    driver, docker, _ = _driver(short_root)

    first, second = await asyncio.gather(
        driver.ensure(_spec()),
        driver.ensure(_spec()),
    )

    assert first == second
    assert sum(command.argv[1] == "create" for command in docker.calls) == 2
    assert driver._resource_locks == {}


async def test_create_reply_loss_adopts_exact_resource_without_duplication(
    short_root,
):
    driver, docker, _ = _driver(short_root)
    docker.network_create_race = True
    docker.create_races = {"agent", "relay"}

    result = await driver.ensure(_spec())

    assert result.observation.complete
    assert len(docker.containers) == 2
    create_roles = [
        _labels(command.argv)[LABEL_ROLE]
        for command in docker.calls
        if command.argv[1] == "create"
    ]
    assert create_roles == ["agent", "relay"]
    assert sum(
        command.argv[1:3] == ("network", "create")
        for command in docker.calls
    ) == 1


async def test_inspect_returns_structured_state_without_docker_environment(short_root):
    driver, docker, _ = _driver(short_root)
    await driver.ensure(_spec())

    observation = await driver.inspect(_spec().identity)

    assert observation.network is RuntimeResourceState.CREATED
    assert observation.agent is RuntimeResourceState.RUNNING
    assert observation.relay is RuntimeResourceState.RUNNING
    rendered = repr(observation)
    assert "OH_SESSION_API_KEYS_0" not in rendered
    assert "s" * 43 not in rendered


async def test_release_invalidates_relay_first_and_preserves_state(short_root):
    driver, docker, _ = _driver(short_root)
    await driver.ensure(_spec())
    policy = driver._health_checker.calls[0]
    (policy.paths.state / "conversation.json").write_text("durable")
    call_start = len(docker.calls)

    first = await driver.release(_spec().identity)
    second = await driver.release(_spec().identity)

    mutations = [
        command.argv
        for command in docker.calls[call_start:]
        if command.argv[1] in ("stop", "rm") or command.argv[1:3] == ("network", "rm")
    ]
    agent_id = "b" * 64
    relay_id = "c" * 64
    network_id = "a" * 64
    assert mutations[:5] == [
        ("docker", "stop", "--time", "10", relay_id),
        ("docker", "rm", relay_id),
        ("docker", "stop", "--time", "10", agent_id),
        ("docker", "rm", agent_id),
        ("docker", "network", "rm", network_id),
    ]
    assert first == second
    assert (policy.paths.state / "conversation.json").read_text() == "durable"
    assert not policy.paths.root.exists()


async def test_foreign_resource_identity_fails_without_deletion(short_root):
    driver, docker, _ = _driver(short_root)
    names = derive_generation_names(_spec().identity)
    docker.networks[names.network] = {
        "Name": names.network,
        "Driver": "bridge",
        "Labels": {LABEL_ROLE: "network"},
    }

    with pytest.raises(RuntimeIdentityConflict, match="network runtime identity"):
        await driver.ensure(_spec())

    assert names.network in docker.networks
    assert not any(command.argv[1:3] == ("network", "rm") for command in docker.calls)


async def test_health_failure_cleans_exact_generation_and_redacts_secrets(short_root):
    failure = RuntimeHealthFailure("fixed relay health gate failed")
    driver, docker, health = _driver(short_root, health=Health(failure))

    with pytest.raises(RuntimeHealthFailure, match="fixed relay health gate failed"):
        await driver.ensure(_spec())

    assert health.calls
    assert docker.containers == {}
    assert docker.networks == {}
    assert not health.calls[0].paths.root.exists()


async def test_primary_failure_preserves_cleanup_failure_diagnostic(short_root):
    failure = RuntimeHealthFailure("fixed relay health gate failed")
    driver, docker, _ = _driver(short_root, health=Health(failure))
    docker.fail_remove_roles = {"relay"}

    with pytest.raises(RuntimeHealthFailure) as captured:
        await driver.ensure(_spec())

    notes = captured.value.__notes__
    assert any("cleanup also failed" in note for note in notes)
    assert "r" * 43 not in repr(notes)


async def test_expired_or_overlong_generation_never_touches_docker(short_root):
    driver, docker, _ = _driver(short_root)
    with pytest.raises(RuntimeExpired, match="elapsed"):
        await driver.ensure(_spec(deadline=NOW))
    with pytest.raises(RuntimeExpired, match="maximum"):
        await driver.ensure(_spec(deadline=NOW + timedelta(days=2)))
    assert docker.calls == []


async def test_probe_fails_closed_on_image_entrypoint_drift(short_root):
    driver, docker, _ = _driver(short_root)
    docker.agent_entrypoint = ("changed",)

    with pytest.raises(RuntimeUnavailable, match="agent image entrypoint"):
        await driver.probe()


async def test_probe_fails_closed_on_image_platform_drift(short_root):
    driver, docker, _ = _driver(short_root)
    docker.relay_platform = "linux/amd64"

    with pytest.raises(RuntimeUnavailable, match="relay image platform"):
        await driver.probe()


async def test_runner_exceptions_and_oversized_output_are_typed_and_bounded(
    short_root,
):
    driver, _, _ = _driver(short_root)

    async def exploding_runner(command):
        raise ValueError("sensitive runner detail")

    driver._runner = exploding_runner
    with pytest.raises(RuntimeUnavailable, match="command runner failed") as captured:
        await driver.probe()
    assert "sensitive runner detail" not in str(captured.value)

    async def oversized_runner(command):
        return CommandExecution(0, stdout="x" * (300 * 1024))

    driver._runner = oversized_runner
    with pytest.raises(RuntimeUnavailable, match="fixed bound"):
        await driver.probe()


async def test_probe_failure_releases_an_existing_exact_generation(short_root):
    driver, docker, _ = _driver(short_root)
    await driver.ensure(_spec())
    policy = driver._health_checker.calls[0]
    driver._probe_complete = False
    docker.agent_entrypoint = ("changed",)

    with pytest.raises(RuntimeUnavailable, match="agent image entrypoint"):
        await driver.ensure(_spec())

    assert docker.containers == {}
    assert docker.networks == {}
    assert not policy.paths.root.exists()


@pytest.mark.parametrize(
    "drift",
    ["secret", "tmpfs", "extra_network", "privileged", "restart"],
)
async def test_container_configuration_drift_is_never_adopted_or_deleted(
    short_root,
    drift,
):
    driver, docker, _ = _driver(short_root)
    await driver.ensure(_spec())
    names = derive_generation_names(_spec().identity)
    agent = docker.containers[names.agent]
    if drift == "secret":
        agent["Config"]["Env"] = [
            "OH_SECRET_KEY=" + "z" * 43
            if value.startswith("OH_SECRET_KEY=")
            else value
            for value in agent["Config"]["Env"]
        ]
    elif drift == "tmpfs":
        agent["HostConfig"]["Tmpfs"]["/tmp"] = "rw,size=1m"
    elif drift == "extra_network":
        agent["NetworkSettings"]["Networks"]["foreign"] = {"Aliases": []}
    elif drift == "privileged":
        agent["HostConfig"]["Privileged"] = True
    else:
        agent["HostConfig"]["RestartPolicy"]["Name"] = "always"
    call_start = len(docker.calls)

    with pytest.raises(RuntimeIdentityConflict):
        await driver.ensure(_spec())

    mutations = [
        command
        for command in docker.calls[call_start:]
        if command.argv[1] in ("stop", "rm")
        or command.argv[1:3] == ("network", "rm")
    ]
    assert mutations == []
    assert names.agent in docker.containers


async def test_foreign_network_attachment_is_never_deleted(short_root):
    driver, docker, _ = _driver(short_root)
    await driver.ensure(_spec())
    names = derive_generation_names(_spec().identity)
    docker.networks[names.network]["Containers"]["foreign-id"] = {
        "Name": "foreign"
    }

    with pytest.raises(RuntimeIdentityConflict, match="foreign attachment"):
        await driver.ensure(_spec())

    assert names.network in docker.networks
    assert names.agent in docker.containers


async def test_early_container_exit_is_not_restarted(short_root):
    driver, docker, _ = _driver(short_root)
    await driver.ensure(_spec())
    names = derive_generation_names(_spec().identity)
    docker.containers[names.relay]["State"]["Status"] = "exited"

    with pytest.raises(RuntimeUnavailable, match="relay container exited"):
        await driver.ensure(_spec())

    relay_starts = [
        command
        for command in docker.calls
        if command.argv == ("docker", "start", "c" * 64)
    ]
    assert len(relay_starts) == 1


class MutableClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


async def test_expiry_sweep_releases_only_after_absolute_deadline_plus_grace(
    short_root,
):
    clock = MutableClock(NOW)
    driver, docker, _ = _driver(short_root, clock=clock)
    spec = _spec(deadline=NOW + timedelta(minutes=5))
    await driver.ensure(spec)

    clock.value = spec.deadline + timedelta(seconds=30)
    before = await driver.sweep_expired()
    assert before.released == ()
    assert docker.containers

    clock.value += timedelta(seconds=1)
    after = await driver.sweep_expired()
    assert after.released == (spec.identity,)
    assert after.failed == ()
    assert docker.containers == {}
    assert docker.networks == {}


async def test_expiry_sweep_ignores_malformed_or_foreign_labels(short_root):
    clock = MutableClock(NOW + timedelta(days=1))
    driver, docker, _ = _driver(short_root, clock=clock)
    docker.networks["foreign"] = {
        "Name": "foreign",
        "Driver": "bridge",
        "Labels": {
            "openloop.runtime.schema": "v1",
            "openloop.runtime.profile": "openhands",
            "openloop.runtime.role": "network",
            "openloop.runtime.deadline": "not-an-epoch",
        },
    }

    result = await driver.sweep_expired()

    assert result.released == ()
    assert result.failed == ()
    assert "foreign" in docker.networks
