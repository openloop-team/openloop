"""Unit tests for the host execution sandbox."""

import pytest

from openloop.sandbox import HostSandbox, SandboxError


async def test_host_sandbox_runs_in_workspace(tmp_path):
    (tmp_path / "hello.txt").write_text("hi\n")
    out = await HostSandbox().exec(tmp_path, "ls")
    assert "hello.txt" in out


async def test_host_sandbox_pipes_stdin(tmp_path):
    out = await HostSandbox().exec(tmp_path, "cat", stdin="from stdin")
    assert out == "from stdin"


async def test_host_sandbox_raises_on_failure(tmp_path):
    with pytest.raises(SandboxError, match="no such file"):
        await HostSandbox().exec(
            tmp_path,
            "python3",
            "-c",
            "import sys; sys.stderr.write('no such file'); sys.exit(1)",
        )
