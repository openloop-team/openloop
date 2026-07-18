"""Pure Docker command policy for one fixed OpenHands generation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from openloop.openhands.runtime_profile import (
    CONVERSATION_LEASE_TTL_SECONDS,
    DEFAULT_OPENHANDS_SERVER_IMAGE,
    SUPPORTED_DOCKER_PLATFORMS,
    native_docker_platform,
    require_immutable_server_image,
    runtime_server_image,
)
from openloop.tools.openhands_relay import (
    CONTAINER_RELAY_CAPABILITY_FILE,
    CONTAINER_RELAY_CONFIG_FILE,
    DEFAULT_HAPROXY_RELAY_IMAGE,
    CompiledOpenHandsRelay,
    RelayMode,
    compile_openhands_relay,
)

from .contract import (
    GenerationRuntimeIdentity,
    OpenHandsGenerationSpec,
    RuntimeUnavailable,
)


RUNTIME_SCHEMA = "v1"
RUNTIME_PROFILE = "openhands"
LABEL_SCHEMA = "openloop.runtime.schema"
LABEL_PROFILE = "openloop.runtime.profile"
LABEL_OPERATION = "openloop.runtime.operation"
LABEL_JOB = "openloop.runtime.job"
LABEL_GENERATION = "openloop.runtime.generation"
LABEL_ROLE = "openloop.runtime.role"
LABEL_DEADLINE = "openloop.runtime.deadline"

AGENT_MEMORY_BYTES = 4 * 1024 * 1024 * 1024
AGENT_PIDS_LIMIT = 512
AGENT_CPUS = 2.0
AGENT_TMPFS = "/tmp:rw,nosuid,nodev,size=512m"
AGENT_COMMAND = (
    "/usr/local/bin/openhands-agent-server",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
)
AGENT_FIXED_ENVIRONMENT = (
    ("OH_CONVERSATIONS_PATH", "/openhands-state/conversations"),
    ("OH_LEASE_TTL_SECONDS", CONVERSATION_LEASE_TTL_SECONDS),
    ("HOME", "/tmp"),
    ("GIT_CONFIG_COUNT", "1"),
    ("GIT_CONFIG_KEY_0", "safe.directory"),
    ("GIT_CONFIG_VALUE_0", "/workspace"),
)
AGENT_SECRET_ENVIRONMENT_NAMES = ("OH_SESSION_API_KEYS_0", "OH_SECRET_KEY")
RELAY_COMMAND = (
    "/usr/local/sbin/haproxy",
    "-W",
    "-db",
    "-f",
    CONTAINER_RELAY_CONFIG_FILE,
)
EXPECTED_AGENT_ENTRYPOINT = (
    "tini",
    "--",
    "/usr/local/bin/openhands-agent-server",
)
EXPECTED_RELAY_ENTRYPOINT = ("docker-entrypoint.sh",)
_DIGEST_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_DOCKER_OBJECT_ID = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_CONTRACT_FORMAT = "{{json .Config.Entrypoint}}|{{.Os}}/{{.Architecture}}"
_DEADLINE_WRAPPER = (
    'set -eu; deadline="$1"; kill_after="$2"; shift 2; '
    'now="$(date +%s)"; remaining="$((deadline - now))"; '
    '[ "$remaining" -gt 0 ] || exit 124; '
    'exec timeout -k "$kill_after" "$remaining" "$@"'
)


def _bounded_positive(name: str, value: object, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _trusted_root(name: str, value: object) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute Path")
    rendered = os.fspath(value)
    if "\0" in rendered or "," in rendered:
        raise ValueError(f"{name} contains an unsupported character")
    resolved = value.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"{name} cannot be a filesystem root")
    return resolved


@dataclass(frozen=True, slots=True)
class DockerRuntimeConfig:
    runtime_root: Path
    state_root: Path
    docker: str = "docker"
    agent_image: str = DEFAULT_OPENHANDS_SERVER_IMAGE
    relay_image: str = DEFAULT_HAPROXY_RELAY_IMAGE
    platform: str = ""
    uid: int = -1
    gid: int = -1
    maximum_lifetime_seconds: int = 86_400
    kill_after_seconds: int = 10
    reconciliation_grace_seconds: int = 30

    def __post_init__(self) -> None:
        runtime_root = _trusted_root("runtime_root", self.runtime_root)
        state_root = _trusted_root("state_root", self.state_root)
        if runtime_root == state_root or runtime_root.is_relative_to(
            state_root
        ) or state_root.is_relative_to(runtime_root):
            raise ValueError("runtime_root and state_root must be disjoint")
        object.__setattr__(self, "runtime_root", runtime_root)
        object.__setattr__(self, "state_root", state_root)
        if not isinstance(self.docker, str) or not self.docker or "\0" in self.docker:
            raise ValueError("docker executable is invalid")
        require_immutable_server_image(self.agent_image)
        if not isinstance(self.relay_image, str) or _DIGEST_IMAGE.fullmatch(
            self.relay_image
        ) is None:
            raise ValueError("HAProxy relay image must be pinned by sha256 digest")
        selected_platform = self.platform or native_docker_platform()
        if selected_platform not in SUPPORTED_DOCKER_PLATFORMS:
            raise ValueError("unsupported Docker runtime platform")
        object.__setattr__(self, "platform", selected_platform)
        if isinstance(self.uid, bool) or not isinstance(self.uid, int):
            raise ValueError("uid is out of range")
        if isinstance(self.gid, bool) or not isinstance(self.gid, int):
            raise ValueError("gid is out of range")
        if self.uid < -1 or self.gid < -1:
            raise ValueError("uid and gid must use -1 only as the default sentinel")
        selected_uid = os.getuid() if self.uid == -1 else self.uid
        selected_gid = os.getgid() if self.gid == -1 else self.gid
        if not 0 <= selected_uid <= 2**31 - 1:
            raise ValueError("uid is out of range")
        if not 0 <= selected_gid <= 2**31 - 1:
            raise ValueError("gid is out of range")
        object.__setattr__(self, "uid", selected_uid)
        object.__setattr__(self, "gid", selected_gid)
        _bounded_positive(
            "maximum_lifetime_seconds",
            self.maximum_lifetime_seconds,
            maximum=86_400,
        )
        _bounded_positive("kill_after_seconds", self.kill_after_seconds, maximum=300)
        _bounded_positive(
            "reconciliation_grace_seconds",
            self.reconciliation_grace_seconds,
            maximum=3_600,
        )

    @property
    def resolved_agent_image(self) -> str:
        return runtime_server_image(self.agent_image, self.platform)


@dataclass(frozen=True, slots=True)
class GenerationNames:
    network: str
    agent: str
    relay: str


@dataclass(frozen=True, slots=True)
class GenerationPaths:
    root: Path
    artifacts: Path
    socket: Path
    workspace: Path
    state: Path

    @property
    def host_socket(self) -> Path:
        return self.socket / "agent.sock"


def derive_generation_names(identity: GenerationRuntimeIdentity) -> GenerationNames:
    if not isinstance(identity, GenerationRuntimeIdentity):
        raise TypeError("identity must be a GenerationRuntimeIdentity")
    stem = f"ol-oh-{identity.job_id.hex}-g{identity.generation}"
    return GenerationNames(
        network=f"{stem}-net",
        agent=f"{stem}-agent",
        relay=f"{stem}-relay",
    )


def derive_generation_paths(
    config: DockerRuntimeConfig,
    identity: GenerationRuntimeIdentity,
) -> GenerationPaths:
    if not isinstance(config, DockerRuntimeConfig):
        raise TypeError("config must be a DockerRuntimeConfig")
    if not isinstance(identity, GenerationRuntimeIdentity):
        raise TypeError("identity must be a GenerationRuntimeIdentity")
    root = config.runtime_root / str(identity.job_id) / str(identity.generation)
    paths = GenerationPaths(
        root=root,
        artifacts=root / "relay",
        socket=root / "socket",
        workspace=root / "workspace",
        state=config.state_root / str(identity.job_id) / "agent-server",
    )
    if len(os.fspath(paths.host_socket).encode("utf-8")) > 100:
        raise RuntimeUnavailable("host relay socket path exceeds the UDS budget")
    return paths


def runtime_labels(
    identity: GenerationRuntimeIdentity,
    role: str,
) -> dict[str, str]:
    if not isinstance(identity, GenerationRuntimeIdentity):
        raise TypeError("identity must be a GenerationRuntimeIdentity")
    if role not in ("network", "agent", "relay"):
        raise ValueError("unsupported runtime resource role")
    return {
        LABEL_SCHEMA: RUNTIME_SCHEMA,
        LABEL_PROFILE: RUNTIME_PROFILE,
        LABEL_OPERATION: str(identity.operation_id),
        LABEL_JOB: str(identity.job_id),
        LABEL_GENERATION: str(identity.generation),
        LABEL_ROLE: role,
        LABEL_DEADLINE: str(identity.deadline_epoch),
    }


@dataclass(frozen=True, slots=True)
class DockerCommand:
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.argv or any(
            not isinstance(value, str) or "\0" in value for value in self.argv
        ):
            raise ValueError("Docker command argv is invalid")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or "=" in item[0]
            or "\0" in item[0]
            or not isinstance(item[1], str)
            or "\0" in item[1]
            for item in self.environment
        ):
            raise ValueError("Docker command environment is invalid")
        if len({name for name, _ in self.environment}) != len(self.environment):
            raise ValueError("Docker command environment names must be unique")
        if self.timeout_seconds <= 0:
            raise ValueError("Docker command timeout must be positive")

    def __repr__(self) -> str:
        return (
            f"DockerCommand(argv={self.argv!r}, "
            f"environment_names={tuple(name for name, _ in self.environment)!r}, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class DockerGenerationPolicy:
    config: DockerRuntimeConfig
    spec: OpenHandsGenerationSpec
    names: GenerationNames
    paths: GenerationPaths
    compiled_relay: CompiledOpenHandsRelay

    @classmethod
    def build(
        cls,
        config: DockerRuntimeConfig,
        spec: OpenHandsGenerationSpec,
        *,
        mode: RelayMode = RelayMode.RUNNING,
    ) -> "DockerGenerationPolicy":
        if not isinstance(config, DockerRuntimeConfig):
            raise TypeError("config must be a DockerRuntimeConfig")
        if not isinstance(spec, OpenHandsGenerationSpec):
            raise TypeError("spec must be an OpenHandsGenerationSpec")
        if not isinstance(mode, RelayMode):
            raise TypeError("mode must be RelayMode")
        names = derive_generation_names(spec.identity)
        paths = derive_generation_paths(config, spec.identity)
        compiled = compile_openhands_relay(
            job_id=spec.job_id,
            generation=spec.generation,
            conversation_id=spec.conversation_id,
            relay_capability=spec.relay_capability,
            session_api_key=spec.session_api_key,
            mode=mode,
        )
        return cls(config, spec, names, paths, compiled)

    def __repr__(self) -> str:
        return (
            "DockerGenerationPolicy("
            f"config={self.config!r}, identity={self.spec.identity!r}, "
            f"names={self.names!r}, paths={self.paths!r}, "
            "compiled_relay=<redacted>)"
        )

    def labels(self, role: str) -> tuple[str, ...]:
        values = runtime_labels(self.spec.identity, role)
        result: list[str] = []
        for name, value in values.items():
            result.extend(("--label", f"{name}={value}"))
        return tuple(result)

    def network_create(self) -> DockerCommand:
        return DockerCommand(
            (
                self.config.docker,
                "network",
                "create",
                "--driver",
                "bridge",
                *self.labels("network"),
                self.names.network,
            )
        )

    def _deadline_args(self, workload: tuple[str, ...]) -> tuple[str, ...]:
        return (
            "-c",
            _DEADLINE_WRAPPER,
            "openloop-deadline",
            str(self.spec.identity.deadline_epoch),
            str(self.config.kill_after_seconds),
            *workload,
        )

    def agent_create(self) -> DockerCommand:
        environment = (
            ("OH_SESSION_API_KEYS_0", self.spec.session_api_key),
            ("OH_SECRET_KEY", self.spec.conversation_secret),
        )
        argv = (
            self.config.docker,
            "create",
            "--platform",
            self.config.platform,
            "--name",
            self.names.agent,
            *self.labels("agent"),
            "--network",
            self.names.network,
            "--network-alias",
            "agent",
            "--user",
            f"{self.config.uid}:{self.config.gid}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            AGENT_TMPFS,
            "--memory",
            str(AGENT_MEMORY_BYTES),
            "--memory-swap",
            str(AGENT_MEMORY_BYTES),
            "--pids-limit",
            str(AGENT_PIDS_LIMIT),
            "--cpus",
            str(AGENT_CPUS),
            "--mount",
            f"type=bind,src={self.paths.workspace},dst=/workspace",
            "--mount",
            f"type=bind,src={self.paths.state},dst=/openhands-state",
            "--env",
            AGENT_SECRET_ENVIRONMENT_NAMES[0],
            "--env",
            AGENT_SECRET_ENVIRONMENT_NAMES[1],
            *(
                item
                for name, value in AGENT_FIXED_ENVIRONMENT
                for item in ("--env", f"{name}={value}")
            ),
            "--entrypoint",
            "/bin/sh",
            self.config.resolved_agent_image,
            *self._deadline_args(AGENT_COMMAND),
        )
        return DockerCommand(argv, environment=environment, timeout_seconds=60.0)

    def relay_create(self) -> DockerCommand:
        runtime = self.compiled_relay.runtime
        argv: tuple[str, ...] = (
            self.config.docker,
            "create",
            "--name",
            self.names.relay,
            "--platform",
            self.config.platform,
            *self.labels("relay"),
            "--network",
            self.names.network,
            "--user",
            f"{self.config.uid}:{self.config.gid}",
            "--cap-drop",
            *runtime.cap_drop,
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--memory",
            str(runtime.memory_bytes),
            "--memory-swap",
            str(runtime.memory_bytes),
            "--pids-limit",
            str(runtime.pids_limit),
        )
        for tmpfs in runtime.tmpfs:
            argv += ("--tmpfs", tmpfs)
        generation_target = self.compiled_relay.endpoint.socket_path.parent
        argv += (
            "--mount",
            f"type=bind,src={self.paths.artifacts},dst=/run/openloop/config,readonly",
            "--mount",
            f"type=bind,src={self.paths.artifacts},dst=/run/openloop/secrets,readonly",
            "--mount",
            f"type=bind,src={self.paths.socket},dst={generation_target}",
            "--entrypoint",
            "/bin/sh",
            self.config.relay_image,
            *self._deadline_args(RELAY_COMMAND),
        )
        return DockerCommand(argv, timeout_seconds=60.0)

    def expected_container_args(self, role: str) -> tuple[str, ...]:
        if role == "agent":
            return self._deadline_args(AGENT_COMMAND)
        if role == "relay":
            return self._deadline_args(RELAY_COMMAND)
        raise ValueError("unsupported runtime container role")

    def start(self, name: str, object_id: str) -> DockerCommand:
        if name not in (self.names.agent, self.names.relay):
            raise ValueError("cannot start a foreign runtime resource")
        if not isinstance(object_id, str) or _DOCKER_OBJECT_ID.fullmatch(
            object_id
        ) is None:
            raise ValueError("cannot start an invalid runtime resource")
        return DockerCommand(
            (self.config.docker, "start", object_id), timeout_seconds=60.0
        )


def image_contract_commands(config: DockerRuntimeConfig) -> tuple[DockerCommand, ...]:
    if not isinstance(config, DockerRuntimeConfig):
        raise TypeError("config must be a DockerRuntimeConfig")
    tools = "command -v date >/dev/null && command -v timeout >/dev/null"
    return (
        DockerCommand((config.docker, "version", "--format", "{{.Server.Version}}")),
        DockerCommand(
            (
                config.docker,
                "image",
                "inspect",
                config.resolved_agent_image,
                "--format",
                _IMAGE_CONTRACT_FORMAT,
            )
        ),
        DockerCommand(
            (
                config.docker,
                "image",
                "inspect",
                config.relay_image,
                "--format",
                _IMAGE_CONTRACT_FORMAT,
            )
        ),
        DockerCommand(
            (
                config.docker,
                "run",
                "--rm",
                "--platform",
                config.platform,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--entrypoint",
                "/bin/sh",
                config.resolved_agent_image,
                "-c",
                f"{tools} && test -x {AGENT_COMMAND[0]}",
            ),
            timeout_seconds=60.0,
        ),
        DockerCommand(
            (
                config.docker,
                "run",
                "--rm",
                "--platform",
                config.platform,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--entrypoint",
                "/bin/sh",
                config.relay_image,
                "-c",
                f"{tools} && test -x {RELAY_COMMAND[0]}",
            ),
            timeout_seconds=60.0,
        ),
    )


__all__ = [
    "AGENT_COMMAND",
    "AGENT_CPUS",
    "AGENT_FIXED_ENVIRONMENT",
    "AGENT_MEMORY_BYTES",
    "AGENT_PIDS_LIMIT",
    "AGENT_SECRET_ENVIRONMENT_NAMES",
    "AGENT_TMPFS",
    "DockerCommand",
    "DockerGenerationPolicy",
    "DockerRuntimeConfig",
    "EXPECTED_AGENT_ENTRYPOINT",
    "EXPECTED_RELAY_ENTRYPOINT",
    "GenerationNames",
    "GenerationPaths",
    "LABEL_DEADLINE",
    "LABEL_GENERATION",
    "LABEL_JOB",
    "LABEL_OPERATION",
    "LABEL_PROFILE",
    "LABEL_ROLE",
    "LABEL_SCHEMA",
    "RELAY_COMMAND",
    "RUNTIME_PROFILE",
    "RUNTIME_SCHEMA",
    "derive_generation_names",
    "derive_generation_paths",
    "image_contract_commands",
    "runtime_labels",
]
