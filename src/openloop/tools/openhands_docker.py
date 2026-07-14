"""Private, authenticated adapter for OpenHands 1.31.0 ``DockerWorkspace``."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pydantic import Field

from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout


logger = logging.getLogger(__name__)

PINNED_OPENHANDS_VERSION = "1.31.0"
DEFAULT_OPENHANDS_SERVER_IMAGE = (
    "ghcr.io/openhands/agent-server@"
    "sha256:08d3994f9287f8d52b07907ac1575ecfaa48b972697ddae4f1cb5c2f03713fab"
)
CONVERSATION_LEASE_TTL_SECONDS = "45"

_DIGEST_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_REQUIRED_DISTRIBUTIONS = (
    "openhands-sdk",
    "openhands-workspace",
    "openhands-agent-server",
)


class HardenedDockerWorkspaceError(RuntimeError):
    """The pinned authenticated Docker boundary cannot be constructed safely."""


def require_immutable_server_image(image: str) -> str:
    if not isinstance(image, str) or not _DIGEST_IMAGE.fullmatch(image):
        raise HardenedDockerWorkspaceError(
            "OpenHands agent-server image must be pinned by sha256 digest"
        )
    return image


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
    platform: str = "linux/amd64"

    def __post_init__(self) -> None:
        require_immutable_server_image(self.image)
        if not 1 <= self.host_port <= 65535:
            raise HardenedDockerWorkspaceError("invalid OpenHands host port")
        if not self.session_api_key or not self.conversation_secret:
            raise HardenedDockerWorkspaceError("missing OpenHands runtime key")

    def environment(self) -> dict[str, str]:
        return {
            "OH_SESSION_API_KEYS_0": self.session_api_key,
            "OH_SECRET_KEY": self.conversation_secret,
            "OH_CONVERSATIONS_PATH": "/openhands-state/conversations",
            "OH_LEASE_TTL_SECONDS": CONVERSATION_LEASE_TTL_SECONDS,
        }

    def command(self, *, container_name: str) -> list[str]:
        command = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            "--rm",
            "--ulimit",
            "nofile=65536:65536",
            "--name",
            container_name,
            "-v",
            f"{self.workspace}:/workspace:rw",
            "-v",
            f"{self.state_dir}:/openhands-state:rw",
            "-p",
            f"127.0.0.1:{self.host_port}:8000",
        ]
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
            f"network={self.network!r}, platform={self.platform!r})"
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
    ) -> None:
        self.layout = layout
        self.keys = keys
        self.server_image = require_immutable_server_image(server_image)
        self.network = network
        self._command_runner = command_runner or self._run_command
        self._port_allocator = port_allocator or self._find_loopback_port
        self._workspace_factory = workspace_factory or self._create_sdk_workspace

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
        if (
            paths.root.is_relative_to(resolved_workspace)
            or resolved_workspace.is_relative_to(paths.root)
        ):
            raise HardenedDockerWorkspaceError(
                "OpenHands state directory must be disjoint from the Git checkout"
            )
        port = self._port_allocator()
        if port < 1:
            raise HardenedDockerWorkspaceError(
                "no loopback port available for OpenHands agent-server"
            )
        launch = HardenedDockerLaunch(
            image=self.server_image,
            workspace=resolved_workspace,
            state_dir=paths.agent_server,
            host_port=port,
            session_api_key=secrets.token_urlsafe(32),
            conversation_secret=self.keys.conversation_secret(job_id),
            network=self.network,
        )
        return self._workspace_factory(launch)

    def stream_git_delta(
        self,
        workspace: object,
        sink: BinaryIO,
        *,
        base_ref: str,
    ) -> ArchiveStreamResult:
        """Stream the authenticated ``GET /file/archive`` response to a host sink."""
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
            "/file/archive",
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
