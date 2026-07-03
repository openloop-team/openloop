"""Unit tests for the execution sandboxes (no docker daemon needed)."""

from pathlib import Path

import pytest

from openloop.sandbox import DockerSandbox, HostSandbox, SandboxError
from openloop.sandbox import runner as sandbox_runner


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
            tmp_path, "python3", "-c",
            "import sys; sys.stderr.write('no such file'); sys.exit(1)",
        )


def _docker_args(sandbox: DockerSandbox, *cmd, stdin=None):
    return sandbox._args(Path("/ws"), cmd, interactive=stdin is not None)


def test_docker_args_default_deny_egress():
    args = _docker_args(DockerSandbox(), "git", "apply")
    net = args[args.index("--network") + 1]
    assert net == "none"


def test_docker_args_never_forward_environment():
    """The LLM key / any credential lives in the controller env; the container
    must get none of it. docker run forwards no env by default — assert we
    never add any."""
    args = _docker_args(DockerSandbox(), "git", "apply", stdin="diff")
    assert "-e" not in args
    assert "--env" not in args
    assert not any(a.startswith("--env") for a in args)


def test_docker_args_isolation_flags_and_teardown():
    args = _docker_args(DockerSandbox(), "sh", "-c", "true")
    assert "--rm" in args  # container reaped even when the command fails
    assert "ALL" == args[args.index("--cap-drop") + 1]
    assert "no-new-privileges" == args[args.index("--security-opt") + 1]
    assert "openloop.sandbox=worker" == args[args.index("--label") + 1]


def test_docker_args_mount_entrypoint_and_command():
    args = _docker_args(DockerSandbox(image="img"), "git", "apply", "-p1")
    assert "/ws:/workspace" == args[args.index("-v") + 1]
    assert "/workspace" == args[args.index("-w") + 1]
    # Entrypoint override so any command runs, not the image default.
    assert "git" == args[args.index("--entrypoint") + 1]
    assert args[args.index("img"):] == ["img", "apply", "-p1"]


def test_docker_args_stdin_only_when_needed():
    assert "-i" not in _docker_args(DockerSandbox(), "ls")
    assert "-i" in _docker_args(DockerSandbox(), "cat", stdin="x")


def test_docker_configurable_egress_network():
    args = _docker_args(DockerSandbox(network="egress-proxy"), "git", "fetch")
    assert "egress-proxy" == args[args.index("--network") + 1]


class _FakeCompleted:
    def __init__(self):
        self.stdout = ""
        self.stderr = ""
        self.returncode = 0


def _fake_probe_subprocess(monkeypatch, *, container_writes: bool):
    """Fake subprocess.run for probe(): version ping succeeds; the container
    run records its args and, when ``container_writes``, simulates the
    container's `git init` landing through the bind mount."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if "run" in args and container_writes:
            mount = args[args.index("-v") + 1]
            host_side = Path(mount.split(":", 1)[0])
            (host_side / "probe" / ".git").mkdir(parents=True)
        return _FakeCompleted()

    monkeypatch.setattr("subprocess.run", fake_run)
    return calls


def test_probe_rehearses_the_real_invocation(monkeypatch, tmp_path):
    calls = _fake_probe_subprocess(monkeypatch, container_writes=True)
    root = tmp_path / "workspaces"

    DockerSandbox(image="img", network="egress-proxy").probe(workspace_root=root)

    ping, run = calls
    assert ping[:2] == ["docker", "version"]
    # The probe run uses the REAL configured image + network + isolation args,
    # and mounts a throwaway workspace under the configured root.
    assert "img" in run
    assert "egress-proxy" == run[run.index("--network") + 1]
    assert "--rm" in run and "ALL" == run[run.index("--cap-drop") + 1]
    mount_host = run[run.index("-v") + 1].split(":", 1)[0]
    assert Path(mount_host).parent == root
    # git is the probe command — the one binary the worker requires.
    assert run[run.index("--entrypoint") + 1] == "git"
    # The throwaway probe workspace is cleaned up afterwards.
    assert list(root.iterdir()) == []


def test_probe_detects_unshared_workspace_root(monkeypatch, tmp_path):
    """The containerized-deploy pitfall: docker mounted SOME host dir, the
    container wrote happily, but nothing came back through our path — the
    workspace root is not host-shared. Must fail closed with a pointed error."""
    _fake_probe_subprocess(monkeypatch, container_writes=False)

    with pytest.raises(sandbox_runner.SandboxUnavailable, match="not shared"):
        DockerSandbox().probe(workspace_root=tmp_path / "ws")


def test_probe_missing_cli_gets_its_own_error():
    with pytest.raises(sandbox_runner.SandboxUnavailable, match="not usable"):
        DockerSandbox(docker_bin="definitely-not-docker").probe()


async def test_docker_exec_builds_args_and_delegates(monkeypatch, tmp_path):
    recorded = {}

    async def fake_run(*cmd, cwd=None, stdin=None):
        recorded["cmd"] = cmd
        recorded["stdin"] = stdin
        return "ok"

    monkeypatch.setattr(sandbox_runner, "_run", fake_run)
    out = await DockerSandbox().exec(tmp_path, "git", "apply", stdin="diff")

    assert out == "ok"
    assert recorded["stdin"] == "diff"
    assert recorded["cmd"][:3] == ("docker", "run", "--rm")
    assert f"{tmp_path}:/workspace" in recorded["cmd"]
