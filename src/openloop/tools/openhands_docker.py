"""Private, authenticated adapter for OpenHands 1.36.0 ``DockerWorkspace``."""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
import os
import re
import secrets
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from pydantic import Field

from openloop.openhands.runtime_profile import (
    CONVERSATION_LEASE_TTL_SECONDS,
    DEFAULT_OPENHANDS_SERVER_IMAGE,
    PINNED_OPENHANDS_VERSION,
    SUPPORTED_DOCKER_PLATFORMS,
    OpenHandsRuntimeProfileError,
    native_docker_platform,
    require_immutable_server_image,
    runtime_server_image,
)
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout


logger = logging.getLogger(__name__)

_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_REQUIRED_DISTRIBUTIONS = (
    "openhands-sdk",
    "openhands-workspace",
    "openhands-agent-server",
)
HardenedDockerWorkspaceError = OpenHandsRuntimeProfileError


@dataclass(frozen=True, slots=True, repr=False)
class HardenedDockerLaunch:
    """One container launch. Secret values never appear in its command or repr."""

    image: str
    workspace: Path
    state_dir: Path
    host_port: int
    session_api_key: str
    conversation_secret: str
    network: str | None = None
    platform: str = field(default_factory=native_docker_platform)
    # "loopback" publishes 127.0.0.1:<host_port> on the Docker daemon host —
    # correct when the runtime runs on that host. "network" publishes nothing;
    # the runtime dials the container by name over a shared user-defined
    # network (sibling-container/Compose deployments, where the daemon host's
    # loopback is unreachable).
    connect: str = "loopback"

    def __post_init__(self) -> None:
        require_immutable_server_image(self.image)
        if not 1 <= self.host_port <= 65535:
            raise HardenedDockerWorkspaceError("invalid OpenHands host port")
        if not self.session_api_key or not self.conversation_secret:
            raise HardenedDockerWorkspaceError("missing OpenHands runtime key")
        if self.platform not in SUPPORTED_DOCKER_PLATFORMS:
            raise HardenedDockerWorkspaceError(
                f"unsupported OpenHands Docker platform: {self.platform!r}"
            )
        if self.connect not in ("loopback", "network"):
            raise HardenedDockerWorkspaceError(
                f"unsupported OpenHands connect mode: {self.connect!r}"
            )
        if self.connect == "network" and not self.network:
            raise HardenedDockerWorkspaceError(
                "connect='network' requires a user-defined Docker network "
                "shared with the runtime container"
            )

    def environment(self) -> dict[str, str]:
        return {
            "OH_SESSION_API_KEYS_0": self.session_api_key,
            "OH_SECRET_KEY": self.conversation_secret,
            "OH_CONVERSATIONS_PATH": "/openhands-state/conversations",
            "OH_LEASE_TTL_SECONDS": CONVERSATION_LEASE_TTL_SECONDS,
            # command() overrides the image user with the launching uid, so
            # the image user's home directory is unusable and the image sets
            # no HOME of its own. /tmp is container-local and world-writable.
            "HOME": "/tmp",
            # Kept as belt-and-braces: with the uid override the checkout
            # owner normally matches the process, but scope Git's trust
            # exception to the one checkout anyway; do not mutate global
            # config inside the long-lived state mount.
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "/workspace",
        }

    def command(self, *, container_name: str) -> list[str]:
        command = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            # Deliberately no --rm: a boot crash must leave the exited
            # container behind so ``docker logs`` (which upstream embeds in
            # its health-failure error) still has evidence. cleanup() removes
            # the container explicitly instead.
            #
            # Run as the launching uid — the owner of both bind mounts. The
            # image's non-root ``openhands`` user cannot traverse the 0700
            # mkdtemp workspace or the state directory on a real Linux
            # daemon (Docker Desktop's file sharing masks this on macOS).
            # Same approach as the sealed sandbox (sandbox/runner.py).
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--ulimit",
            "nofile=65536:65536",
            "--name",
            container_name,
            "-v",
            f"{self.workspace}:/workspace:rw",
            "-v",
            f"{self.state_dir}:/openhands-state:rw",
        ]
        if self.connect == "loopback":
            # Only loopback mode exposes a daemon-host port. Network mode is
            # reached in-network by container name and publishes nothing.
            command.extend(("-p", f"127.0.0.1:{self.host_port}:8000"))
        for variable in self.environment():
            # Docker copies only these named values from the subprocess
            # environment; no ambient controller variable enters the container,
            # and the secret values never enter argv or command logs.
            command.extend(("-e", variable))
        if self.network:
            command.extend(("--network", self.network))
        command.extend((self.image, "--host", "0.0.0.0", "--port", "8000"))
        return command

    def __repr__(self) -> str:
        return (
            "HardenedDockerLaunch("
            f"image={self.image!r}, workspace={str(self.workspace)!r}, "
            f"state_dir={str(self.state_dir)!r}, host_port={self.host_port}, "
            "session_api_key=<redacted>, conversation_secret=<redacted>, "
            f"network={self.network!r}, platform={self.platform!r}, "
            f"connect={self.connect!r})"
        )


@dataclass(frozen=True, slots=True)
class ArchiveStreamResult:
    base_commit: str
    base_ref: str
    bytes_written: int


WorkspaceFactory = Callable[[HardenedDockerLaunch], object]
CommandRunner = Callable[
    [list[str], dict[str, str] | None, float | None], subprocess.CompletedProcess[str]
]


class HardenedDockerWorkspace:
    """OpenLoop-owned policy adapter around pinned upstream DockerWorkspace."""

    def __init__(
        self,
        *,
        layout: OpenHandsStateLayout,
        keys: OpenHandsKeyDeriver,
        server_image: str = DEFAULT_OPENHANDS_SERVER_IMAGE,
        network: str | None = None,
        workspace_factory: WorkspaceFactory | None = None,
        command_runner: CommandRunner | None = None,
        port_allocator: Callable[[], int] | None = None,
        platform: str | None = None,
        connect: str = "loopback",
    ) -> None:
        self.layout = layout
        self.keys = keys
        self.server_image = require_immutable_server_image(server_image)
        self.network = network
        # Fail closed at construction so the boot gate reports a bad connect
        # configuration instead of the first job discovering it.
        if connect not in ("loopback", "network"):
            raise HardenedDockerWorkspaceError(
                f"unsupported OpenHands connect mode: {connect!r}"
            )
        if connect == "network" and not network:
            raise HardenedDockerWorkspaceError(
                "connect='network' requires CODING_WORKER_OPENHANDS_NETWORK"
            )
        self.connect = connect
        self._command_runner = command_runner or self._run_command
        self._port_allocator = port_allocator or self._find_loopback_port
        self._workspace_factory = workspace_factory or self._create_sdk_workspace
        self.platform = platform or native_docker_platform()

    def probe(self) -> None:
        """Fail closed if the exact pinned SDK seam is not present."""
        for distribution in _REQUIRED_DISTRIBUTIONS:
            try:
                installed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise HardenedDockerWorkspaceError(
                    f"{distribution} is not installed"
                ) from exc
            if installed != PINNED_OPENHANDS_VERSION:
                raise HardenedDockerWorkspaceError(
                    f"{distribution} {installed} is incompatible; "
                    f"expected {PINNED_OPENHANDS_VERSION}"
                )

        from openhands.workspace import DockerWorkspace

        signature = inspect.signature(DockerWorkspace._start_container)
        if tuple(signature.parameters) != ("self", "image", "context"):
            raise HardenedDockerWorkspaceError(
                "OpenHands DockerWorkspace launch seam is incompatible"
            )
        required_fields = {"api_key", "host", "host_port", "server_image", "volumes"}
        if not required_fields.issubset(DockerWorkspace.model_fields):
            raise HardenedDockerWorkspaceError(
                "OpenHands DockerWorkspace fields are incompatible"
            )

    def create(self, workspace: Path, job_id: str) -> object:
        paths = self.layout.for_job(job_id)
        resolved_workspace = workspace.resolve(strict=True)
        if paths.root.is_relative_to(
            resolved_workspace
        ) or resolved_workspace.is_relative_to(paths.root):
            raise HardenedDockerWorkspaceError(
                "OpenHands state directory must be disjoint from the Git checkout"
            )
        port = self._port_allocator()
        if port < 1:
            raise HardenedDockerWorkspaceError(
                "no loopback port available for OpenHands agent-server"
            )
        launch = HardenedDockerLaunch(
            image=runtime_server_image(self.server_image, self.platform),
            workspace=resolved_workspace,
            state_dir=paths.agent_server,
            host_port=port,
            session_api_key=secrets.token_urlsafe(32),
            conversation_secret=self.keys.conversation_secret(job_id),
            network=self.network,
            platform=self.platform,
            connect=self.connect,
        )
        return self._workspace_factory(launch)

    def stream_git_delta(
        self,
        workspace: object,
        sink: BinaryIO,
        *,
        base_ref: str,
    ) -> ArchiveStreamResult:
        """Stream authenticated ``GET /api/file/archive`` bytes to a host sink."""
        if not base_ref or base_ref.startswith("-"):
            raise HardenedDockerWorkspaceError("invalid git-delta base ref")
        client = getattr(workspace, "client", None)
        api_key = getattr(workspace, "api_key", None)
        if client is None or not api_key:
            raise HardenedDockerWorkspaceError(
                "authenticated OpenHands workspace client is unavailable"
            )
        written = 0
        with client.stream(
            "GET",
            "/api/file/archive",
            params={
                "path": "/workspace",
                "format": "git-delta",
                "base_ref": base_ref,
            },
        ) as response:
            response.raise_for_status()
            base_commit = response.headers.get("X-Archive-Base-Commit", "")
            if not _GIT_OBJECT_ID.fullmatch(base_commit):
                raise HardenedDockerWorkspaceError(
                    "OpenHands archive returned an invalid base commit"
                )
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                result = sink.write(chunk)
                if result is not None and result != len(chunk):
                    raise HardenedDockerWorkspaceError(
                        "workspace artifact sink performed a short write"
                    )
                written += len(chunk)
        return ArchiveStreamResult(
            base_commit=base_commit, base_ref=base_ref, bytes_written=written
        )

    def attach_conversation(
        self,
        workspace: object,
        *,
        agent: object,
        conversation_id: uuid.UUID,
        callbacks: list | None = None,
        max_iterations: int = 500,
    ) -> object:
        """Attach only to an already-loaded persisted conversation.

        Pinned ``RemoteConversation`` creates a new conversation with the
        caller-supplied ID after a 404. During the stale lease window that would
        replace a conversation the new server has deliberately not acquired.
        The hardened boundary therefore performs an authenticated existence
        check and refuses to call the SDK constructor unless attachment is safe.
        """
        client = getattr(workspace, "client", None)
        api_key = getattr(workspace, "api_key", None)
        if client is None or not api_key:
            raise HardenedDockerWorkspaceError(
                "authenticated OpenHands workspace client is unavailable"
            )
        response = client.get(f"/api/conversations/{conversation_id}")
        if response.status_code == 404:
            raise HardenedDockerWorkspaceError(
                "persisted OpenHands conversation is not available for attach; "
                "its ownership lease may still be active"
            )
        try:
            response.raise_for_status()
        except Exception as exc:
            raise HardenedDockerWorkspaceError(
                "failed to verify persisted OpenHands conversation"
            ) from exc

        from openhands.sdk.conversation.impl.remote_conversation import (
            RemoteConversation,
        )

        return RemoteConversation(
            agent=agent,
            workspace=workspace,
            conversation_id=conversation_id,
            callbacks=callbacks,
            max_iteration_per_run=max_iterations,
            visualizer=None,
            delete_on_close=False,
        )

    def _create_sdk_workspace(self, launch: HardenedDockerLaunch) -> object:
        self.probe()
        from openhands.sdk.workspace import RemoteWorkspace
        from openhands.workspace import DockerWorkspace

        runner = self._command_runner

        class _PinnedHardenedDockerWorkspace(DockerWorkspace):
            # Upstream's field is repr-visible; override it because this is a
            # per-container credential, not ordinary debug configuration.
            api_key: str | None = Field(default=None, exclude=True, repr=False)

            def _start_container(self, image: str, context) -> None:
                self._image_name = image
                self.host_port = launch.host_port

                version = runner(["docker", "version"], None, 10.0)
                if version.returncode != 0:
                    raise HardenedDockerWorkspaceError("Docker is not available")

                container_name = f"agent-server-{uuid.uuid4()}"
                run_env = dict(os.environ)
                run_env.update(launch.environment())
                result = runner(
                    launch.command(container_name=container_name),
                    run_env,
                    self.health_check_timeout,
                )
                if result.returncode != 0:
                    raise HardenedDockerWorkspaceError(
                        "failed to start hardened OpenHands agent-server"
                    )
                self._container_id = result.stdout.strip()
                if not self._container_id:
                    raise HardenedDockerWorkspaceError(
                        "Docker returned no OpenHands container ID"
                    )
                logger.info("started hardened OpenHands agent-server container")

                if self.detach_logs:
                    self._logs_thread = threading.Thread(
                        target=self._stream_docker_logs, daemon=True
                    )
                    self._logs_thread.start()

                if launch.connect == "network":
                    # Sibling-container mode: the runtime shares a
                    # user-defined network with the agent and resolves it by
                    # container name; the daemon host's loopback is not
                    # reachable from this network namespace.
                    object.__setattr__(self, "host", f"http://{container_name}:8000")
                else:
                    object.__setattr__(
                        self, "host", f"http://127.0.0.1:{self.host_port}"
                    )
                # Deliberately preserve ``self.api_key``. Pinned upstream sets it
                # to None here, silently disabling client authentication.
                try:
                    self._wait_for_health(timeout=self.health_check_timeout)
                    RemoteWorkspace.model_post_init(self, context)
                except BaseException:
                    self.cleanup()
                    raise

            def cleanup(self) -> None:
                # ``command()`` omits ``--rm`` so a crashed container keeps
                # its logs until this point; upstream cleanup only stops the
                # container, so remove it explicitly afterwards.
                container_id = getattr(self, "_container_id", None)
                try:
                    super().cleanup()
                finally:
                    if container_id:
                        try:
                            runner(["docker", "rm", "-f", container_id], None, 30.0)
                        except Exception:
                            logger.warning(
                                "failed to remove agent-server container %s",
                                container_id,
                            )

        return _PinnedHardenedDockerWorkspace(
            server_image=launch.image,
            host_port=launch.host_port,
            working_dir="/workspace",
            volumes=[
                f"{launch.workspace}:/workspace:rw",
                f"{launch.state_dir}:/openhands-state:rw",
            ],
            forward_env=[],
            network=launch.network,
            platform=launch.platform,
            extra_ports=False,
            detach_logs=False,
            api_key=launch.session_api_key,
        )

    @staticmethod
    def _run_command(
        command: list[str], environment: dict[str, str] | None, timeout: float | None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _find_loopback_port() -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
