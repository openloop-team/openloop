"""Where model-influenced commands execute (hardening Phase 3).

The coding worker applies model-generated edits. *Where* that execution happens
is a security boundary, so it sits behind one seam:

- :class:`HostSandbox` — a subprocess on the host, today's behavior and the
  default. Fine for the light diff-apply worker; no isolation.
- :class:`DockerSandbox` — each command runs in a throwaway container with the
  workspace bind-mounted. **Default-deny egress** (``--network none``), no
  environment forwarded (so no LLM key or credential can leak in — the model
  call stays in the controller), all capabilities dropped, no privilege
  escalation, and ``--rm`` so the container is reaped even when the command
  fails. This is the isolation unit later phases build on (per-tenant
  sandboxes, the OpenHands backend).

The orchestrator's credential-bearing git operations intentionally do NOT go
through a sandbox: they are the trusted boundary and never execute
model-generated content. Only the worker's edit application does.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable


class SandboxError(RuntimeError):
    """A sandboxed command failed."""


class SandboxUnavailable(RuntimeError):
    """The sandbox backend cannot run on this host (e.g. no docker)."""


@runtime_checkable
class Sandbox(Protocol):
    """Executes one command against a workspace and returns its stdout."""

    async def exec(
        self, workspace: Path, *cmd: str, stdin: str | None = None
    ) -> str: ...


async def _run(*cmd: str, cwd: Path | None = None, stdin: str | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(stdin.encode() if stdin is not None else None)
    if proc.returncode != 0:
        raise SandboxError(
            f"`{' '.join(cmd)}` failed ({proc.returncode}): {err.decode().strip()}"
        )
    return out.decode()


class HostSandbox:
    """Runs the command as a plain subprocess in the workspace. No isolation."""

    async def exec(
        self, workspace: Path, *cmd: str, stdin: str | None = None
    ) -> str:
        return await _run(*cmd, cwd=workspace, stdin=stdin)


# Label stamped on every sandbox container so leftovers are findable (and the
# teardown test can assert there are none).
_CONTAINER_LABEL = "openloop.sandbox=worker"


class DockerSandbox:
    """Runs each command in a throwaway container over the mounted workspace.

    The container gets exactly one thing from the host: the workspace bind
    mount. No environment is forwarded (``docker run`` passes none by default
    and this class never adds ``-e``), the network is ``none`` unless an
    explicit egress network is configured, and the process runs as the host
    uid/gid so workspace files stay writable/removable by the app afterwards.
    """

    def __init__(
        self,
        image: str = "alpine/git",
        *,
        network: str = "none",
        docker_bin: str = "docker",
    ) -> None:
        self.image = image
        # "none" = default-deny egress. Point this at a user-defined network
        # fronted by an egress proxy to move to an allowlist model later.
        self.network = network
        self._docker = docker_bin

    # Generous: the first probe on a fresh host pulls the sandbox image.
    _PROBE_RUN_TIMEOUT_SECONDS = 180

    def probe(self, workspace_root: Path | None = None) -> None:
        """Prove the WHOLE sandbox path at boot, not just daemon reachability.

        Raises :class:`SandboxUnavailable` unless a real container run — the
        configured image, network, uid mapping, and a bind mount under the
        configured workspace root — succeeds AND the container's write is
        visible back on this side of the mount. That round-trip is what
        catches the containerized-deploy pitfall: sibling containers resolve
        ``-v`` paths on the *host*, so a workspace root that isn't host-shared
        mounts some other directory and the write never comes back.

        Mirrors the boot-time posture of the GitHub App wiring (sign a real
        JWT, don't just import PyJWT). Synchronous on purpose so app wiring
        can gate tool registration at boot (fail-closed: no host fallback).
        """
        import shutil as _shutil
        import subprocess
        import tempfile as _tempfile

        # Step 1: CLI + daemon ping, so a missing binary or dead daemon gets
        # its own clear error instead of surfacing as a failed container run.
        try:
            subprocess.run(
                [self._docker, "version", "--format", "{{.Server.Version}}"],
                check=True, capture_output=True, timeout=10,
            )
        except Exception as exc:
            raise SandboxUnavailable(
                f"docker is not usable ({exc}); refusing to run the coding "
                "worker unsandboxed"
            ) from exc

        # Step 2: dress rehearsal of the real invocation. `git init` is the
        # probe command because git is the only binary the worker requires of
        # the image, and it *writes* — proving the mount is writable by the
        # mapped uid, not just present.
        if workspace_root is not None:
            workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            _tempfile.mkdtemp(prefix="openloop-sandbox-probe-", dir=workspace_root)
        )
        try:
            args = self._args(
                workspace,
                ("git", "init", "--quiet", "/workspace/probe"),
                interactive=False,
            )
            try:
                subprocess.run(
                    args, check=True, capture_output=True, text=True,
                    timeout=self._PROBE_RUN_TIMEOUT_SECONDS,
                )
            except subprocess.CalledProcessError as exc:
                raise SandboxUnavailable(
                    "sandbox probe run failed (image "
                    f"{self.image!r}, network {self.network!r}): "
                    f"{(exc.stderr or '').strip()}"
                ) from exc
            except Exception as exc:
                raise SandboxUnavailable(
                    f"sandbox probe run failed (image {self.image!r}, "
                    f"network {self.network!r}): {exc}"
                ) from exc
            if not (workspace / "probe" / ".git").is_dir():
                raise SandboxUnavailable(
                    "sandbox probe wrote inside the container but the write "
                    f"is not visible at {workspace} — the workspace root is "
                    "not shared with the host. In a containerized deploy, "
                    "CODING_WORKER_WORKSPACE_DIR must be a host path mounted "
                    "into the runtime at the same location."
                )
        finally:
            _shutil.rmtree(workspace, ignore_errors=True)

    def _args(
        self, workspace: Path, cmd: tuple[str, ...], *, interactive: bool
    ) -> list[str]:
        args = [self._docker, "run", "--rm"]
        if interactive:
            args.append("-i")  # pipe stdin through
        args += [
            "--label", _CONTAINER_LABEL,
            "--network", self.network,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
        ]
        if sys.platform != "win32":
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
        args += [
            "-v", f"{workspace}:/workspace",
            "-w", "/workspace",
            # Override the image entrypoint so any command runs, not just the
            # image's default binary (alpine/git's entrypoint is `git`).
            "--entrypoint", cmd[0],
            self.image,
            *cmd[1:],
        ]
        return args

    async def exec(
        self, workspace: Path, *cmd: str, stdin: str | None = None
    ) -> str:
        if not cmd:
            raise ValueError("empty sandbox command")
        args = self._args(workspace, cmd, interactive=stdin is not None)
        return await _run(*args, stdin=stdin)
