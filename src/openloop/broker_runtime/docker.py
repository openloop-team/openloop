"""Privileged, idempotent Docker driver for fixed OpenHands generations."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

import httpx

from openloop.tools.openhands_relay import (
    AGENT_SESSION_HEADER,
    RELAY_CAPABILITY_HEADER,
    RelayClientEndpoint,
    RelayRuntimePolicy,
)

from .contract import (
    EnsuredGeneration,
    GenerationObservation,
    GenerationRuntimeIdentity,
    OpenHandsGenerationSpec,
    ReleaseObservation,
    RuntimeDriver,
    RuntimeDriverError,
    RuntimeExpired,
    RuntimeHealthFailure,
    RuntimeIdentityConflict,
    RuntimeResourceState,
    RuntimeUnavailable,
)
from .docker_policy import (
    AGENT_CPUS,
    AGENT_FIXED_ENVIRONMENT,
    AGENT_MEMORY_BYTES,
    AGENT_PIDS_LIMIT,
    AGENT_SECRET_ENVIRONMENT_NAMES,
    AGENT_TMPFS,
    EXPECTED_AGENT_ENTRYPOINT,
    EXPECTED_RELAY_ENTRYPOINT,
    DockerCommand,
    DockerGenerationPolicy,
    DockerRuntimeConfig,
    GenerationNames,
    GenerationPaths,
    derive_generation_names,
    derive_generation_paths,
    image_contract_commands,
    runtime_labels,
)
from .filesystem import (
    generation_filesystem_observation,
    prepare_generation_filesystem,
    release_generation_filesystem,
)


_OUTPUT_LIMIT = 256 * 1024
_DOCKER_OBJECT_ID = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_VALUE = re.compile(r"[A-Za-z0-9_-]{32,256}\Z")
_TMPFS_SIZE = re.compile(r"([0-9]+)([kmgt]?)\Z", re.IGNORECASE)
_NOT_FOUND_MARKERS = (
    "no such container",
    "no such object",
    "no such network",
    "not found",
)


@dataclass(frozen=True, slots=True, repr=False)
class CommandExecution:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise TypeError("returncode must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("command output must be text")

    def __repr__(self) -> str:
        return (
            f"CommandExecution(returncode={self.returncode}, "
            f"stdout=<bounded text {len(self.stdout)} chars>, "
            f"stderr=<bounded text {len(self.stderr)} chars>)"
        )


@dataclass(frozen=True, slots=True)
class ExpirySweepObservation:
    released: tuple[GenerationRuntimeIdentity, ...]
    failed: tuple[GenerationRuntimeIdentity, ...]

    def __post_init__(self) -> None:
        for collection in (self.released, self.failed):
            if not isinstance(collection, tuple) or any(
                not isinstance(value, GenerationRuntimeIdentity)
                for value in collection
            ):
                raise TypeError(
                    "sweep identities must be tuples of generation identities"
                )


class CommandRunner(Protocol):
    async def __call__(self, command: DockerCommand) -> CommandExecution: ...


HealthChecker = Callable[[DockerGenerationPolicy], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _ObservedResource:
    state: RuntimeResourceState
    document: dict | None = None


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


async def _drain(stream: asyncio.StreamReader | None) -> tuple[str, bool]:
    if stream is None:
        return "", False
    kept = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = _OUTPUT_LIMIT - len(kept)
        if remaining > 0:
            kept.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return bytes(kept).decode("utf-8", errors="replace"), truncated


async def _default_command_runner(command: DockerCommand) -> CommandExecution:
    environment = None
    if command.environment:
        environment = dict(os.environ)
        environment.update(command.environment)
    try:
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise RuntimeUnavailable("Docker command could not be started") from exc
    stdout_task = asyncio.create_task(_drain(process.stdout))
    stderr_task = asyncio.create_task(_drain(process.stderr))
    try:
        async with asyncio.timeout(command.timeout_seconds):
            returncode = await process.wait()
            stdout_result, stderr_result = await asyncio.gather(
                stdout_task, stderr_task
            )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise RuntimeUnavailable("Docker command exceeded its fixed deadline") from exc
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    stdout, stdout_truncated = stdout_result
    stderr, stderr_truncated = stderr_result
    if stdout_truncated or stderr_truncated:
        raise RuntimeUnavailable("Docker command output exceeded its fixed bound")
    return CommandExecution(returncode, stdout, stderr)


def _safe_output(execution: CommandExecution, secrets: tuple[str, ...]) -> str:
    detail = (execution.stderr or execution.stdout).strip()[:4096]
    for secret in secrets:
        detail = detail.replace(secret, "<redacted>")
    return detail


def _is_not_found(execution: CommandExecution) -> bool:
    detail = f"{execution.stdout}\n{execution.stderr}".lower()
    return any(marker in detail for marker in _NOT_FOUND_MARKERS)


def _json_document(execution: CommandExecution, *, subject: str) -> dict:
    try:
        value = json.loads(execution.stdout)
    except (TypeError, ValueError) as exc:
        raise RuntimeUnavailable(
            f"Docker returned invalid {subject} inspection data"
        ) from exc
    if isinstance(value, list):
        if len(value) != 1:
            raise RuntimeUnavailable(
                f"Docker returned ambiguous {subject} inspection data"
            )
        value = value[0]
    if not isinstance(value, dict):
        raise RuntimeUnavailable(f"Docker returned invalid {subject} inspection data")
    return value


def _image_contract(execution: CommandExecution) -> tuple[tuple[str, ...], str]:
    entrypoint_text, separator, platform = execution.stdout.strip().partition("|")
    if not separator:
        raise RuntimeUnavailable("Docker image contract metadata is invalid")
    try:
        entrypoint = json.loads(entrypoint_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeUnavailable("Docker image contract metadata is invalid") from exc
    if not isinstance(entrypoint, list) or any(
        not isinstance(value, str) for value in entrypoint
    ):
        raise RuntimeUnavailable("Docker image contract metadata is invalid")
    return tuple(entrypoint), platform


def _resource_id(document: dict, *, role: str) -> str:
    value = document.get("Id")
    if not isinstance(value, str) or _DOCKER_OBJECT_ID.fullmatch(value) is None:
        raise RuntimeIdentityConflict(f"{role} Docker object identity is invalid")
    return value


def _environment_map(value: object, *, role: str) -> dict[str, str]:
    if not isinstance(value, list):
        raise RuntimeIdentityConflict(f"{role} container environment is missing")
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, str) or "=" not in item:
            raise RuntimeIdentityConflict(f"{role} container environment is invalid")
        name, contents = item.split("=", 1)
        if not name or name in result:
            raise RuntimeIdentityConflict(f"{role} container environment is invalid")
        result[name] = contents
    return result


def _normalize_tmpfs_options(value: str) -> frozenset[str]:
    result: set[str] = set()
    for option in value.split(","):
        if option.startswith("size="):
            match = _TMPFS_SIZE.fullmatch(option.removeprefix("size="))
            if match is None:
                raise RuntimeIdentityConflict("container tmpfs size is invalid")
            amount = int(match.group(1))
            suffix = match.group(2).lower()
            multiplier = {
                "": 1,
                "k": 1024,
                "m": 1024**2,
                "g": 1024**3,
                "t": 1024**4,
            }[suffix]
            result.add(f"size={amount * multiplier}")
        elif option:
            result.add(option)
    return frozenset(result)


def _tmpfs_policy(value: object) -> dict[str, frozenset[str]]:
    if not isinstance(value, dict):
        raise RuntimeIdentityConflict("container tmpfs policy is missing")
    result: dict[str, frozenset[str]] = {}
    for path, options in value.items():
        if not isinstance(path, str) or not isinstance(options, str):
            raise RuntimeIdentityConflict("container tmpfs policy is invalid")
        result[path] = _normalize_tmpfs_options(options)
    return result


def _expected_tmpfs(specifications: tuple[str, ...]) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for specification in specifications:
        path, separator, options = specification.partition(":")
        if not separator:
            raise RuntimeUnavailable("fixed tmpfs policy is invalid")
        result[path] = _normalize_tmpfs_options(options)
    return result


def _validate_runtime_labels(
    actual: object,
    identity: GenerationRuntimeIdentity,
    role: str,
) -> None:
    if not isinstance(actual, dict):
        raise RuntimeIdentityConflict(f"{role} runtime labels are missing")
    expected = runtime_labels(identity, role)
    runtime_actual = {
        name: value
        for name, value in actual.items()
        if isinstance(name, str) and name.startswith("openloop.runtime.")
    }
    if runtime_actual != expected:
        raise RuntimeIdentityConflict(f"{role} runtime identity does not match")


def _container_state(document: dict, *, role: str) -> RuntimeResourceState:
    state = document.get("State")
    if not isinstance(state, dict):
        raise RuntimeUnavailable(f"{role} container state metadata is invalid")
    status = state.get("Status")
    if status == "created":
        return RuntimeResourceState.CREATED
    if status == "running":
        return RuntimeResourceState.RUNNING
    if status in ("exited", "dead"):
        return RuntimeResourceState.EXITED
    raise RuntimeUnavailable(f"{role} container has unsupported Docker state")


class DockerOpenHandsRuntimeDriver(RuntimeDriver):
    def __init__(
        self,
        config: DockerRuntimeConfig,
        *,
        runner: CommandRunner | None = None,
        clock: Callable[[], datetime] | None = None,
        health_checker: HealthChecker | None = None,
    ) -> None:
        if not isinstance(config, DockerRuntimeConfig):
            raise TypeError("config must be a DockerRuntimeConfig")
        self.config = config
        self._runner = runner or _default_command_runner
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._health_checker = health_checker or self._default_health_check
        self._probe_lock = asyncio.Lock()
        self._probe_complete = False
        self._resource_locks: dict[str, _LockEntry] = {}

    @property
    def maximum_lifetime_seconds(self) -> int:
        return self.config.maximum_lifetime_seconds

    @staticmethod
    def _endpoint(policy: DockerGenerationPolicy) -> RelayClientEndpoint:
        return RelayClientEndpoint(
            socket_path=policy.paths.host_socket,
            conversation_id=policy.spec.conversation_id,
            relay_capability=policy.spec.relay_capability,
            session_api_key=policy.spec.session_api_key,
            mode=policy.compiled_relay.endpoint.mode,
        )

    def describe_endpoint(
        self, spec: OpenHandsGenerationSpec
    ) -> RelayClientEndpoint:
        if not isinstance(spec, OpenHandsGenerationSpec):
            raise TypeError("spec must be an OpenHandsGenerationSpec")
        try:
            return self._endpoint(DockerGenerationPolicy.build(self.config, spec))
        except RuntimeDriverError:
            raise
        except Exception as exc:
            raise RuntimeUnavailable(
                "fixed generation endpoint compilation failed"
            ) from exc

    def _now(self) -> datetime:
        try:
            value = self._clock()
        except Exception as exc:
            raise RuntimeUnavailable("runtime clock failed") from exc
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise RuntimeUnavailable("runtime clock did not return timezone-aware time")
        return value.astimezone(timezone.utc)

    @asynccontextmanager
    async def _locked(
        self, identity: GenerationRuntimeIdentity
    ) -> AsyncIterator[None]:
        key = f"{identity.job_id}:{identity.generation}"
        entry = self._resource_locks.setdefault(key, _LockEntry(asyncio.Lock()))
        entry.users += 1
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            entry.users -= 1
            if entry.users == 0 and self._resource_locks.get(key) is entry:
                self._resource_locks.pop(key)

    async def _invoke(self, command: DockerCommand) -> CommandExecution:
        try:
            execution = await self._runner(command)
        except RuntimeDriverError:
            raise
        except Exception as exc:
            raise RuntimeUnavailable("Docker command runner failed") from exc
        if not isinstance(execution, CommandExecution):
            raise RuntimeUnavailable("Docker command runner returned an invalid result")
        if (
            len(execution.stdout) > _OUTPUT_LIMIT
            or len(execution.stderr) > _OUTPUT_LIMIT
        ):
            raise RuntimeUnavailable("Docker command output exceeded its fixed bound")
        return execution

    async def _execute(
        self,
        command: DockerCommand,
        *,
        action: str,
        secrets: tuple[str, ...] = (),
    ) -> CommandExecution:
        execution = await self._invoke(command)
        if execution.returncode != 0:
            detail = _safe_output(execution, secrets)
            suffix = f": {detail}" if detail else ""
            raise RuntimeUnavailable(f"Docker {action} failed{suffix}")
        return execution

    async def probe(self) -> None:
        async with self._probe_lock:
            if self._probe_complete:
                return
            commands = image_contract_commands(self.config)
            await self._execute(commands[0], action="daemon probe")
            agent = await self._execute(commands[1], action="agent image inspection")
            relay = await self._execute(commands[2], action="relay image inspection")
            agent_entrypoint, agent_platform = _image_contract(agent)
            relay_entrypoint, relay_platform = _image_contract(relay)
            if agent_entrypoint != EXPECTED_AGENT_ENTRYPOINT:
                raise RuntimeUnavailable(
                    "OpenHands agent image entrypoint contract changed"
                )
            if relay_entrypoint != EXPECTED_RELAY_ENTRYPOINT:
                raise RuntimeUnavailable(
                    "HAProxy relay image entrypoint contract changed"
                )
            if agent_platform != self.config.platform:
                raise RuntimeUnavailable(
                    "OpenHands agent image platform contract changed"
                )
            if relay_platform != self.config.platform:
                raise RuntimeUnavailable(
                    "HAProxy relay image platform contract changed"
                )
            await self._execute(commands[3], action="agent deadline-wrapper probe")
            await self._execute(commands[4], action="relay deadline-wrapper probe")
            self._probe_complete = True

    def _validate_deadline(self, identity: GenerationRuntimeIdentity) -> None:
        remaining = (identity.deadline - self._now()).total_seconds()
        if remaining <= 0:
            raise RuntimeExpired("generation execution deadline has elapsed")
        if remaining > self.config.maximum_lifetime_seconds:
            raise RuntimeExpired(
                "generation execution deadline exceeds profile maximum"
            )

    async def _inspect_network(
        self,
        identity: GenerationRuntimeIdentity,
        names: GenerationNames,
    ) -> _ObservedResource:
        command = DockerCommand(
            (
                self.config.docker,
                "network",
                "inspect",
                names.network,
                "--format",
                "{{json .}}",
            )
        )
        execution = await self._invoke(command)
        if execution.returncode != 0:
            if _is_not_found(execution):
                return _ObservedResource(RuntimeResourceState.ABSENT)
            raise RuntimeUnavailable(
                f"Docker network inspection failed: {_safe_output(execution, ())}"
            )
        document = _json_document(execution, subject="network")
        if document.get("Name") != names.network or document.get("Driver") != "bridge":
            raise RuntimeIdentityConflict(
                "network immutable configuration does not match"
            )
        _validate_runtime_labels(document.get("Labels"), identity, "network")
        _resource_id(document, role="network")
        if (
            document.get("Scope") != "local"
            or document.get("EnableIPv6") is not False
            or document.get("Internal") is not False
            or document.get("Attachable") is not False
            or document.get("Ingress") is not False
            or document.get("ConfigOnly") is not False
            or document.get("Options") not in (None, {})
        ):
            raise RuntimeIdentityConflict("network hardening policy does not match")
        attachments = document.get("Containers")
        if not isinstance(attachments, dict):
            raise RuntimeIdentityConflict("network attachment metadata is missing")
        allowed_names = {names.agent, names.relay}
        for attachment in attachments.values():
            if (
                not isinstance(attachment, dict)
                or attachment.get("Name") not in allowed_names
            ):
                raise RuntimeIdentityConflict("network has a foreign attachment")
        return _ObservedResource(RuntimeResourceState.CREATED, document)

    async def _inspect_container(
        self,
        identity: GenerationRuntimeIdentity,
        names: GenerationNames,
        paths: GenerationPaths,
        role: str,
        spec: OpenHandsGenerationSpec | None = None,
    ) -> _ObservedResource:
        name = names.agent if role == "agent" else names.relay
        command = DockerCommand(
            (
                self.config.docker,
                "container",
                "inspect",
                name,
                "--format",
                "{{json .}}",
            )
        )
        execution = await self._invoke(command)
        if execution.returncode != 0:
            if _is_not_found(execution):
                return _ObservedResource(RuntimeResourceState.ABSENT)
            raise RuntimeUnavailable(
                f"Docker {role} inspection failed: {_safe_output(execution, ())}"
            )
        document = _json_document(execution, subject=role)
        if str(document.get("Name", "")).lstrip("/") != name:
            raise RuntimeIdentityConflict(f"{role} container name does not match")
        if document.get("Platform") != "linux":
            raise RuntimeIdentityConflict(f"{role} container platform does not match")
        config = document.get("Config")
        host = document.get("HostConfig")
        if not isinstance(config, dict) or not isinstance(host, dict):
            raise RuntimeIdentityConflict(f"{role} container configuration is missing")
        _validate_runtime_labels(config.get("Labels"), identity, role)
        _resource_id(document, role=role)
        expected_image = (
            self.config.resolved_agent_image
            if role == "agent"
            else self.config.relay_image
        )
        if config.get("Image") != expected_image:
            raise RuntimeIdentityConflict(f"{role} container image does not match")
        if config.get("User") != f"{self.config.uid}:{self.config.gid}":
            raise RuntimeIdentityConflict(f"{role} container user does not match")
        if config.get("Entrypoint") != ["/bin/sh"]:
            raise RuntimeIdentityConflict(f"{role} deadline entrypoint does not match")
        environment = _environment_map(config.get("Env"), role=role)
        if role == "agent":
            expected_environment = dict(AGENT_FIXED_ENVIRONMENT)
            if any(
                environment.get(name) != value
                for name, value in expected_environment.items()
            ):
                raise RuntimeIdentityConflict(
                    "agent fixed environment policy does not match"
                )
            expected_secret_values = (
                (spec.session_api_key, spec.conversation_secret)
                if spec is not None
                else None
            )
            for index, name in enumerate(AGENT_SECRET_ENVIRONMENT_NAMES):
                value = environment.get(name)
                valid = (
                    value == expected_secret_values[index]
                    if expected_secret_values is not None
                    else isinstance(value, str)
                    and _TOKEN_VALUE.fullmatch(value) is not None
                )
                if not valid:
                    raise RuntimeIdentityConflict(
                        "agent secret environment policy does not match"
                    )
            allowed_oh_names = set(expected_environment) | set(
                AGENT_SECRET_ENVIRONMENT_NAMES
            )
            if any(
                name.startswith("OH_") and name not in allowed_oh_names
                for name in environment
            ):
                raise RuntimeIdentityConflict(
                    "agent environment contains an unknown OpenHands setting"
                )
        elif any(name.startswith("OH_") for name in environment):
            raise RuntimeIdentityConflict(
                "relay environment contains an OpenHands credential"
            )
        policy_for_args = DockerGenerationPolicy.build(
            self.config,
            OpenHandsGenerationSpec(
                operation_id=identity.operation_id,
                job_id=identity.job_id,
                conversation_id=identity.job_id,
                generation=identity.generation,
                deadline=identity.deadline,
                relay_capability="x" * 43,
                session_api_key="y" * 43,
                conversation_secret="z" * 43,
            ),
        )
        if config.get("Cmd") != list(policy_for_args.expected_container_args(role)):
            raise RuntimeIdentityConflict(f"{role} deadline command does not match")
        if host.get("NetworkMode") != names.network:
            raise RuntimeIdentityConflict(f"{role} container network does not match")
        if host.get("ReadonlyRootfs") is not True:
            raise RuntimeIdentityConflict(f"{role} root filesystem is not read-only")
        if host.get("CapDrop") != ["ALL"]:
            raise RuntimeIdentityConflict(f"{role} capability policy does not match")
        if host.get("CapAdd") not in (None, []):
            raise RuntimeIdentityConflict(f"{role} capability policy does not match")
        security = host.get("SecurityOpt")
        normalized_security = {
            str(value).removesuffix(":true")
            for value in security
        } if isinstance(security, list) else set()
        if normalized_security != {"no-new-privileges"}:
            raise RuntimeIdentityConflict(f"{role} security policy does not match")
        restart = host.get("RestartPolicy")
        if (
            host.get("PortBindings") not in (None, {})
            or host.get("PublishAllPorts") is not False
        ):
            raise RuntimeIdentityConflict(f"{role} container publishes a port")
        if (
            host.get("Privileged") is not False
            or host.get("AutoRemove") is not False
            or not isinstance(restart, dict)
            or restart.get("Name") not in ("", "no")
            or restart.get("MaximumRetryCount") != 0
        ):
            raise RuntimeIdentityConflict(f"{role} lifecycle policy does not match")
        if (
            host.get("Devices") not in (None, [])
            or host.get("DeviceRequests") not in (None, [])
            or host.get("PidMode") == "host"
            or host.get("IpcMode") == "host"
            or host.get("UTSMode") == "host"
        ):
            raise RuntimeIdentityConflict(f"{role} isolation policy does not match")
        if role == "agent":
            if (
                host.get("Memory") != AGENT_MEMORY_BYTES
                or host.get("MemorySwap") != AGENT_MEMORY_BYTES
                or host.get("PidsLimit") != AGENT_PIDS_LIMIT
                or host.get("NanoCpus") != int(AGENT_CPUS * 1_000_000_000)
            ):
                raise RuntimeIdentityConflict("agent resource policy does not match")
            expected_tmpfs = _expected_tmpfs((AGENT_TMPFS,))
        else:
            runtime = RelayRuntimePolicy()
            if (
                host.get("Memory") != runtime.memory_bytes
                or host.get("MemorySwap") != runtime.memory_bytes
                or host.get("PidsLimit") != runtime.pids_limit
                or host.get("NanoCpus") != 0
            ):
                raise RuntimeIdentityConflict("relay resource policy does not match")
            expected_tmpfs = _expected_tmpfs(runtime.tmpfs)
        if _tmpfs_policy(host.get("Tmpfs")) != expected_tmpfs:
            raise RuntimeIdentityConflict(f"{role} tmpfs policy does not match")
        network_settings = document.get("NetworkSettings")
        if not isinstance(network_settings, dict):
            raise RuntimeIdentityConflict(f"{role} network metadata is missing")
        published = network_settings.get("Ports")
        if isinstance(published, dict) and any(
            bindings not in (None, []) for bindings in published.values()
        ):
            raise RuntimeIdentityConflict(f"{role} container publishes a port")
        if published is not None and not isinstance(published, dict):
            raise RuntimeIdentityConflict(f"{role} port metadata is invalid")
        attached_networks = network_settings.get("Networks")
        if not isinstance(attached_networks, dict) or set(attached_networks) != {
            names.network
        }:
            raise RuntimeIdentityConflict(f"{role} network attachment does not match")
        attachment = attached_networks[names.network]
        if not isinstance(attachment, dict):
            raise RuntimeIdentityConflict(f"{role} network attachment is invalid")
        aliases = attachment.get("Aliases")
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) for alias in aliases
        ):
            raise RuntimeIdentityConflict(f"{role} network aliases are invalid")
        if (role == "agent" and "agent" not in aliases) or (
            role == "relay" and "agent" in aliases
        ):
            raise RuntimeIdentityConflict(f"{role} network alias policy does not match")
        expected_mounts = {
            "agent": {
                (str(paths.workspace), "/workspace", True),
                (str(paths.state), "/openhands-state", True),
            },
            "relay": {
                (str(paths.artifacts), "/run/openloop/config", False),
                (str(paths.artifacts), "/run/openloop/secrets", False),
                (
                    str(paths.socket),
                    f"/run/openloop/jobs/{identity.job_id}/{identity.generation}",
                    True,
                ),
            },
        }[role]
        mounts = document.get("Mounts")
        if not isinstance(mounts, list):
            raise RuntimeIdentityConflict(f"{role} mount metadata is missing")
        actual_mounts = set()
        for mount in mounts:
            if not isinstance(mount, dict) or mount.get("Type") != "bind":
                raise RuntimeIdentityConflict(f"{role} mount policy does not match")
            actual_mounts.add(
                (mount.get("Source"), mount.get("Destination"), mount.get("RW"))
            )
        if actual_mounts != expected_mounts:
            raise RuntimeIdentityConflict(f"{role} mount policy does not match")
        return _ObservedResource(_container_state(document, role=role), document)

    async def _observation(
        self,
        identity: GenerationRuntimeIdentity,
        *,
        spec: OpenHandsGenerationSpec | None = None,
        healthy: bool | None = None,
    ) -> GenerationObservation:
        names = derive_generation_names(identity)
        paths = derive_generation_paths(self.config, identity)
        network, agent, relay = await asyncio.gather(
            self._inspect_network(identity, names),
            self._inspect_container(identity, names, paths, "agent", spec),
            self._inspect_container(identity, names, paths, "relay", spec),
        )
        try:
            artifacts_ready, workspace_ready = generation_filesystem_observation(
                paths, uid=self.config.uid
            )
        except RuntimeDriverError:
            raise
        except Exception as exc:
            raise RuntimeUnavailable(
                "generation filesystem inspection failed"
            ) from exc
        expired = self._now() >= identity.deadline
        structurally_healthy = (
            agent.state is RuntimeResourceState.RUNNING
            and relay.state is RuntimeResourceState.RUNNING
            and not expired
        )
        return GenerationObservation(
            identity=identity,
            network=network.state,
            agent=agent.state,
            relay=relay.state,
            artifacts_ready=artifacts_ready,
            workspace_ready=workspace_ready,
            healthy=structurally_healthy if healthy is None else healthy,
            expired=expired,
        )

    async def inspect(
        self, identity: GenerationRuntimeIdentity
    ) -> GenerationObservation:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        async with self._locked(identity):
            return await self._observation(identity)

    async def _ensure_network(self, policy: DockerGenerationPolicy) -> None:
        observed = await self._inspect_network(policy.spec.identity, policy.names)
        if observed.state is RuntimeResourceState.CREATED:
            return
        execution = await self._invoke(policy.network_create())
        if execution.returncode != 0:
            # A competing identical ensure may have won the create race.
            adopted = await self._inspect_network(policy.spec.identity, policy.names)
            if adopted.state is RuntimeResourceState.CREATED:
                return
            raise RuntimeUnavailable(
                f"Docker network creation failed: {_safe_output(execution, ())}"
            )
        adopted = await self._inspect_network(policy.spec.identity, policy.names)
        if adopted.state is not RuntimeResourceState.CREATED:
            raise RuntimeUnavailable("Docker did not persist the generation network")

    async def _ensure_container(
        self,
        policy: DockerGenerationPolicy,
        role: str,
    ) -> None:
        identity = policy.spec.identity
        observed = await self._inspect_container(
            identity, policy.names, policy.paths, role, policy.spec
        )
        if observed.state is RuntimeResourceState.ABSENT:
            create = policy.agent_create() if role == "agent" else policy.relay_create()
            execution = await self._invoke(create)
            if execution.returncode != 0:
                adopted = await self._inspect_container(
                    identity, policy.names, policy.paths, role, policy.spec
                )
                if adopted.state is RuntimeResourceState.ABSENT:
                    secrets = (
                        policy.spec.relay_capability,
                        policy.spec.session_api_key,
                        policy.spec.conversation_secret,
                    )
                    raise RuntimeUnavailable(
                        f"Docker {role} creation failed: "
                        f"{_safe_output(execution, secrets)}"
                    )
                observed = adopted
            else:
                observed = await self._inspect_container(
                    identity, policy.names, policy.paths, role, policy.spec
                )
        if observed.state is RuntimeResourceState.EXITED:
            if self._now() >= identity.deadline:
                raise RuntimeExpired("generation container self-expired")
            raise RuntimeUnavailable(f"{role} container exited before readiness")
        if observed.state is RuntimeResourceState.CREATED:
            if observed.document is None:
                raise RuntimeUnavailable(f"{role} inspection document is missing")
            object_id = _resource_id(observed.document, role=role)
            await self._execute(
                policy.start(
                    policy.names.agent if role == "agent" else policy.names.relay,
                    object_id,
                ),
                action=f"{role} start",
            )
            observed = await self._inspect_container(
                identity, policy.names, policy.paths, role, policy.spec
            )
        if observed.state is not RuntimeResourceState.RUNNING:
            raise RuntimeUnavailable(f"{role} container did not reach running state")

    async def _default_health_check(self, policy: DockerGenerationPolicy) -> None:
        endpoint = self._endpoint(policy)

        def request() -> bool:
            transport = httpx.HTTPTransport(uds=str(endpoint.socket_path))
            with httpx.Client(
                base_url=endpoint.logical_host,
                transport=transport,
                timeout=httpx.Timeout(2.0),
                headers={
                    RELAY_CAPABILITY_HEADER: endpoint.relay_capability,
                    AGENT_SESSION_HEADER: endpoint.session_api_key,
                },
            ) as client:
                return client.get("/health").status_code == 200

        stop = time.monotonic() + 15.0
        while time.monotonic() < stop and self._now() < policy.spec.deadline:
            try:
                if await asyncio.to_thread(request):
                    return
            except Exception:
                pass
            await asyncio.sleep(0.2)
        raise RuntimeHealthFailure("OpenHands relay health check failed")

    async def ensure(self, spec: OpenHandsGenerationSpec) -> EnsuredGeneration:
        if not isinstance(spec, OpenHandsGenerationSpec):
            raise TypeError("spec must be an OpenHandsGenerationSpec")
        identity = spec.identity
        self._validate_deadline(identity)
        async with self._locked(identity):
            self._validate_deadline(identity)
            try:
                policy = DockerGenerationPolicy.build(self.config, spec)
            except RuntimeDriverError:
                raise
            except Exception as exc:
                raise RuntimeUnavailable(
                    "fixed generation policy compilation failed"
                ) from exc
            try:
                await self.probe()
                try:
                    await asyncio.to_thread(
                        prepare_generation_filesystem,
                        policy.paths,
                        policy.compiled_relay,
                        uid=self.config.uid,
                    )
                except RuntimeDriverError:
                    raise
                except Exception as exc:
                    raise RuntimeUnavailable(
                        "generation filesystem preparation failed"
                    ) from exc
                await self._ensure_network(policy)
                await self._ensure_container(policy, "agent")
                await self._ensure_container(policy, "relay")
                try:
                    await self._health_checker(policy)
                except RuntimeDriverError:
                    raise
                except Exception as exc:
                    raise RuntimeHealthFailure(
                        "generation health checker failed"
                    ) from exc
                observation = await self._observation(
                    identity, spec=spec, healthy=True
                )
                if not observation.complete:
                    raise RuntimeHealthFailure(
                        "generation was incomplete after its health gate"
                    )
                return EnsuredGeneration(
                    handle=identity.opaque_handle,
                    endpoint=self._endpoint(policy),
                    observation=observation,
                )
            except BaseException as exc:
                if not isinstance(exc, RuntimeIdentityConflict):
                    try:
                        await self._release_unlocked(identity)
                    except Exception as cleanup_error:
                        if hasattr(exc, "add_note"):
                            exc.add_note(
                                "runtime cleanup also failed: "
                                f"{cleanup_error.__class__.__name__}"
                            )
                raise

    async def _stop_remove_container(
        self,
        identity: GenerationRuntimeIdentity,
        names: GenerationNames,
        paths: GenerationPaths,
        role: str,
    ) -> None:
        observed = await self._inspect_container(identity, names, paths, role)
        if observed.state is RuntimeResourceState.ABSENT:
            return
        if observed.document is None:
            raise RuntimeUnavailable(f"{role} inspection document is missing")
        object_id = _resource_id(observed.document, role=role)
        if observed.state is RuntimeResourceState.RUNNING:
            execution = await self._invoke(
                DockerCommand(
                    (
                        self.config.docker,
                        "stop",
                        "--time",
                        str(self.config.kill_after_seconds),
                        object_id,
                    ),
                    timeout_seconds=self.config.kill_after_seconds + 30.0,
                )
            )
            if execution.returncode != 0 and not _is_not_found(execution):
                raise RuntimeUnavailable(
                    f"Docker {role} stop failed: {_safe_output(execution, ())}"
                )
        execution = await self._invoke(
            DockerCommand(
                (self.config.docker, "rm", object_id), timeout_seconds=60.0
            )
        )
        if execution.returncode != 0 and not _is_not_found(execution):
            raise RuntimeUnavailable(
                f"Docker {role} removal failed: {_safe_output(execution, ())}"
            )

    async def _release_unlocked(
        self, identity: GenerationRuntimeIdentity
    ) -> ReleaseObservation:
        names = derive_generation_names(identity)
        paths = derive_generation_paths(self.config, identity)
        await self._stop_remove_container(identity, names, paths, "relay")
        await self._stop_remove_container(identity, names, paths, "agent")
        network = await self._inspect_network(identity, names)
        if network.state is RuntimeResourceState.CREATED:
            if network.document is None:
                raise RuntimeUnavailable("network inspection document is missing")
            object_id = _resource_id(network.document, role="network")
            execution = await self._invoke(
                DockerCommand(
                    (self.config.docker, "network", "rm", object_id),
                    timeout_seconds=60.0,
                )
            )
            if execution.returncode != 0 and not _is_not_found(execution):
                raise RuntimeUnavailable(
                    f"Docker network removal failed: {_safe_output(execution, ())}"
                )
        try:
            await asyncio.to_thread(
                release_generation_filesystem,
                paths,
                uid=self.config.uid,
            )
        except RuntimeDriverError:
            raise
        except Exception as exc:
            raise RuntimeUnavailable(
                "generation filesystem release failed"
            ) from exc
        return ReleaseObservation(identity=identity, released=True)

    async def release(
        self, identity: GenerationRuntimeIdentity
    ) -> ReleaseObservation:
        if not isinstance(identity, GenerationRuntimeIdentity):
            raise TypeError("identity must be a GenerationRuntimeIdentity")
        async with self._locked(identity):
            return await self._release_unlocked(identity)

    async def _listed_names(self, kind: str) -> tuple[str, ...]:
        if kind == "container":
            argv = (
                self.config.docker,
                "container",
                "ls",
                "-a",
                "--filter",
                "label=openloop.runtime.schema=v1",
                "--format",
                "{{.Names}}",
            )
        elif kind == "network":
            argv = (
                self.config.docker,
                "network",
                "ls",
                "--filter",
                "label=openloop.runtime.schema=v1",
                "--format",
                "{{.Name}}",
            )
        else:
            raise ValueError("unsupported Docker discovery kind")
        execution = await self._execute(
            DockerCommand(argv), action=f"{kind} expiry discovery"
        )
        names = tuple(
            line.strip() for line in execution.stdout.splitlines() if line.strip()
        )
        if len(names) > 10_000 or any(
            len(name.encode("utf-8")) > 255 or "\0" in name for name in names
        ):
            raise RuntimeUnavailable(f"Docker returned invalid {kind} discovery data")
        return names

    @staticmethod
    def _identity_from_discovery(
        labels: object,
        *,
        expected_role: str,
        resource_name: str,
    ) -> GenerationRuntimeIdentity | None:
        if not isinstance(labels, dict):
            return None
        try:
            if (
                labels.get("openloop.runtime.schema") != "v1"
                or labels.get("openloop.runtime.profile") != "openhands"
                or labels.get("openloop.runtime.role") != expected_role
            ):
                return None
            deadline_epoch = int(labels["openloop.runtime.deadline"])
            identity = GenerationRuntimeIdentity(
                operation_id=UUID(labels["openloop.runtime.operation"]),
                job_id=UUID(labels["openloop.runtime.job"]),
                generation=int(labels["openloop.runtime.generation"]),
                deadline=datetime.fromtimestamp(deadline_epoch, timezone.utc),
            )
            runtime_actual = {
                name: value
                for name, value in labels.items()
                if isinstance(name, str) and name.startswith("openloop.runtime.")
            }
            if runtime_actual != runtime_labels(identity, expected_role):
                return None
            expected_names = derive_generation_names(identity)
            expected_name = getattr(expected_names, expected_role)
            if resource_name != expected_name:
                return None
            return identity
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            OSError,
        ):
            return None

    async def _discover_expiry_identities(self) -> set[GenerationRuntimeIdentity]:
        container_names, network_names = await asyncio.gather(
            self._listed_names("container"), self._listed_names("network")
        )
        identities: set[GenerationRuntimeIdentity] = set()
        for name in container_names:
            execution = await self._invoke(
                DockerCommand(
                    (
                        self.config.docker,
                        "container",
                        "inspect",
                        name,
                        "--format",
                        "{{json .}}",
                    )
                )
            )
            if execution.returncode != 0:
                continue
            try:
                document = _json_document(execution, subject="container discovery")
                container_config = document.get("Config")
                labels = (
                    container_config.get("Labels")
                    if isinstance(container_config, dict)
                    else None
                )
                role = (
                    labels.get("openloop.runtime.role")
                    if isinstance(labels, dict)
                    else None
                )
                if role not in ("agent", "relay"):
                    continue
                identity = self._identity_from_discovery(
                    labels,
                    expected_role=role,
                    resource_name=str(document.get("Name", "")).lstrip("/"),
                )
            except RuntimeDriverError:
                identity = None
            if identity is not None:
                identities.add(identity)
        for name in network_names:
            execution = await self._invoke(
                DockerCommand(
                    (
                        self.config.docker,
                        "network",
                        "inspect",
                        name,
                        "--format",
                        "{{json .}}",
                    )
                )
            )
            if execution.returncode != 0:
                continue
            try:
                document = _json_document(execution, subject="network discovery")
                identity = self._identity_from_discovery(
                    document.get("Labels"),
                    expected_role="network",
                    resource_name=str(document.get("Name", "")),
                )
            except RuntimeDriverError:
                identity = None
            if identity is not None:
                identities.add(identity)
        return identities

    async def sweep_expired(self) -> ExpirySweepObservation:
        """Release fully identified generations beyond deadline plus grace."""
        now_epoch = int(self._now().timestamp())
        identities = await self._discover_expiry_identities()
        expired = sorted(
            (
                identity
                for identity in identities
                if now_epoch
                > identity.deadline_epoch + self.config.reconciliation_grace_seconds
            ),
            key=lambda value: (
                value.deadline_epoch,
                str(value.operation_id),
                str(value.job_id),
                value.generation,
            ),
        )
        released: list[GenerationRuntimeIdentity] = []
        failed: list[GenerationRuntimeIdentity] = []
        for identity in expired:
            try:
                await self.release(identity)
                released.append(identity)
            except RuntimeDriverError:
                failed.append(identity)
        return ExpirySweepObservation(tuple(released), tuple(failed))


__all__ = [
    "CommandExecution",
    "CommandRunner",
    "DockerOpenHandsRuntimeDriver",
    "ExpirySweepObservation",
    "HealthChecker",
]
