"""Integration: the REAL docker sandbox — egress denial, teardown, isolation.

Verifies the Phase 3 exit criteria against an actual docker daemon:

- a blocked destination is unreachable from inside the sandbox (network none);
- a crashed (failing) command leaves no container behind (``--rm`` reaps it);
- the worker's ``git apply`` genuinely runs in the container over the mounted
  workspace;
- the container sees none of the controller's environment (no LLM key leak).

Skipped cleanly when no docker daemon is available.
"""

import shutil
import subprocess

import pytest

from openloop.sandbox import DockerSandbox, SandboxError, SandboxUnavailable


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=True, capture_output=True, timeout=10,
        )
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _docker_usable(), reason="no usable docker daemon"
)

# Small image with git + busybox networking tools; pulled once per machine.
IMAGE = "alpine/git"


@pytest.fixture(scope="module", autouse=True)
def _pull_image():
    subprocess.run(["docker", "pull", "-q", IMAGE], check=True, timeout=300)


def _sandbox(**kwargs) -> DockerSandbox:
    return DockerSandbox(IMAGE, **kwargs)


def test_probe_passes_end_to_end(tmp_path):
    """The full-fidelity boot probe against the real daemon: configured image,
    network none, bind-mount round-trip under a real workspace root."""
    _sandbox().probe(workspace_root=tmp_path / "workspaces")
    # Probe cleans up after itself.
    assert list((tmp_path / "workspaces").iterdir()) == []


def test_probe_fails_closed_on_bad_image(tmp_path):
    with pytest.raises(SandboxUnavailable, match="probe run failed"):
        DockerSandbox("openloop-no-such-image:none").probe(
            workspace_root=tmp_path
        )


def test_probe_fails_closed_on_bad_network(tmp_path):
    with pytest.raises(SandboxUnavailable, match="probe run failed"):
        _sandbox(network="openloop-no-such-network").probe(workspace_root=tmp_path)


async def test_egress_to_any_destination_is_blocked(tmp_path):
    """Default-deny: even github.com is unreachable from inside the sandbox."""
    with pytest.raises(SandboxError):
        await _sandbox().exec(
            tmp_path, "git", "ls-remote", "https://github.com/git/git"
        )


async def test_failing_command_leaves_no_container_behind(tmp_path):
    with pytest.raises(SandboxError, match="boom"):
        await _sandbox().exec(tmp_path, "sh", "-c", "echo boom >&2; exit 7")

    leftovers = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=openloop.sandbox=worker"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    assert leftovers == ""  # --rm reaped the failed container


async def test_git_apply_runs_in_container_over_mounted_workspace(tmp_path):
    (tmp_path / "x").write_text("a\n")
    diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"

    await _sandbox().exec(
        tmp_path, "git", "apply", "--whitespace=nowarn", stdin=diff
    )

    # The edit landed on the host through the bind mount…
    assert (tmp_path / "x").read_text() == "b\n"
    # …and stayed writable/removable by this process (uid-matched user).
    (tmp_path / "x").unlink()


async def test_controller_environment_is_invisible_inside_sandbox(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    out = await _sandbox().exec(tmp_path, "env")
    assert "sk-super-secret" not in out
    assert "ANTHROPIC_API_KEY" not in out
