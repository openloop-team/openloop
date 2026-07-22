"""Host execution seam for the builtin coding worker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol, runtime_checkable


class SandboxError(RuntimeError):
    """A sandboxed command failed."""


class SandboxUnavailable(RuntimeError):
    """A requested sandbox backend is unavailable."""


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
