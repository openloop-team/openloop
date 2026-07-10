"""Integration: sealed runs against the REAL docker daemon (Phase 0).

Exercises every live claim of the four locks:

- outputs round-trip under ``--network none``; egress genuinely blocked;
- nonzero exit is data (no exception), controller env invisible;
- a SIGTERM-trapping script still dies at its deadline (layer 1, classified
  whatever the numeric exit code) and the container self-removes;
- a chatty-but-successful run is NOT killed (drain-vs-block phantom timeout);
- the mount split holds (read-only rootfs, ro inputs, rw outputs only);
- the disk watchdog kills a bind-mount filler (best-effort, lock 3);
- the sealed probe passes end-to-end (timeout as PID 1).

Skipped cleanly when no docker daemon is available. Uses python:3.12-slim —
python + GNU coreutils ``timeout``; the pandas stack is not needed here.
"""

import shutil
import subprocess

import pytest

from openloop.sandbox import DockerSandbox, Mount, SandboxLimits, SealedSpec


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

IMAGE = "python:3.12-slim"


@pytest.fixture(scope="module", autouse=True)
def _pull_image():
    subprocess.run(["docker", "pull", "-q", IMAGE], check=True, timeout=600)


def _sandbox() -> DockerSandbox:
    return DockerSandbox(IMAGE, kind="analysis")


def _workspace(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    (inputs / "data.csv").write_text("a,b\n1,2\n")
    return inputs, outputs


def _spec(inputs, outputs, script, **overrides) -> SealedSpec:
    defaults = dict(
        job_id="itest",
        command=("python", "-"),
        limits=SandboxLimits(
            timeout_seconds=60, kill_after_seconds=5, memory="256m",
            pids_limit=64,
        ),
        mounts=(
            Mount(inputs, "/workspace/inputs", read_only=True),
            Mount(outputs, "/workspace/outputs"),
        ),
        stdin=script,
    )
    defaults.update(overrides)
    return SealedSpec(**defaults)


def _no_leftovers():
    out = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=openloop.sandbox=analysis"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    assert out == ""


async def test_outputs_round_trip_under_network_none(tmp_path):
    inputs, outputs = _workspace(tmp_path)
    script = (
        "data = open('/workspace/inputs/data.csv').read()\n"
        "open('/workspace/outputs/report.md', 'w').write(f'# rows\\n{data}')\n"
    )
    result = await _sandbox().run(_spec(inputs, outputs, script))
    assert result.exit_code == 0, result.stderr
    assert result.timed_out is False and result.killed is False
    assert (outputs / "report.md").read_text().startswith("# rows")
    _no_leftovers()


async def test_egress_is_blocked(tmp_path):
    inputs, outputs = _workspace(tmp_path)
    script = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://example.com', timeout=5)\n"
    )
    result = await _sandbox().run(_spec(inputs, outputs, script))
    assert result.exit_code != 0  # data, not an exception
    assert result.timed_out is False


async def test_nonzero_exit_is_data_not_an_exception(tmp_path):
    inputs, outputs = _workspace(tmp_path)
    result = await _sandbox().run(
        _spec(inputs, outputs, "import sys; sys.stderr.write('boom'); sys.exit(7)")
    )
    assert result.exit_code == 7
    assert "boom" in result.stderr
    assert result.timed_out is False


async def test_controller_environment_is_invisible(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    inputs, outputs = _workspace(tmp_path)
    result = await _sandbox().run(
        _spec(inputs, outputs, "import os; print(dict(os.environ))")
    )
    assert result.exit_code == 0
    assert "sk-super-secret" not in result.stdout


async def test_sigterm_trapping_script_dies_at_deadline(tmp_path):
    """Layer 1: `timeout` as PID 1 with --kill-after. The child ignores
    SIGTERM, so KILL ends it — exit 137 on GNU, which is exactly why
    timed_out is classified from elapsed time, never the numeric code."""
    inputs, outputs = _workspace(tmp_path)
    script = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(600)\n"
    )
    spec = _spec(
        inputs, outputs, script,
        limits=SandboxLimits(timeout_seconds=3, kill_after_seconds=2),
    )
    result = await _sandbox().run(spec)
    assert result.timed_out is True
    assert result.killed is False  # layer 1 handled it; no docker kill needed
    assert result.exit_code != 0
    assert result.duration_seconds >= 3
    _no_leftovers()  # --rm self-cleaned: the orphan-self-cleans path


async def test_chatty_successful_run_is_not_phantom_killed(tmp_path):
    """The drain-vs-block case (lock 1): prints far past the cap, then exits 0
    having written the report. Must complete — truncated, not killed."""
    inputs, outputs = _workspace(tmp_path)
    script = (
        "import sys\n"
        "for _ in range(2000): sys.stdout.write('x' * 1000 + '\\n')\n"
        "open('/workspace/outputs/report.md', 'w').write('done')\n"
    )
    # Cap far below the ~2MB the script prints.
    result = await _sandbox().run(
        _spec(
            inputs, outputs, script,
            limits=SandboxLimits(
                timeout_seconds=60, kill_after_seconds=5, stream_cap_bytes=10_000
            ),
        )
    )
    assert result.exit_code == 0, result.stderr
    assert result.timed_out is False and result.killed is False
    assert result.stdout_truncated is True
    assert len(result.stdout) == 10_000
    assert (outputs / "report.md").read_text() == "done"


async def test_mount_split_and_read_only_rootfs_hold(tmp_path):
    inputs, outputs = _workspace(tmp_path)
    script = (
        "import sys\n"
        "failures = []\n"
        "for path in ('/workspace/inputs/x', '/etc/x', '/x'):\n"
        "    try:\n"
        "        open(path, 'w')\n"
        "        failures.append(path)\n"
        "    except OSError:\n"
        "        pass\n"
        "open('/tmp/scratch', 'w').write('ok')\n"  # tmpfs stays writable
        "sys.exit(1 if failures else 0)\n"
    )
    result = await _sandbox().run(_spec(inputs, outputs, script))
    assert result.exit_code == 0, result.stdout + result.stderr


async def test_disk_watchdog_kills_a_bind_mount_filler(tmp_path):
    """Lock 3's best-effort mitigation: fill outputs/ fast, sleep — the du
    poll breaches, the runner kills by name. A disk kill is NOT a timeout."""
    inputs, outputs = _workspace(tmp_path)
    script = (
        "import time\n"
        "open('/workspace/outputs/fill', 'wb').write(b'\\0' * 30_000_000)\n"
        "time.sleep(600)\n"
    )
    spec = _spec(
        inputs, outputs, script,
        limits=SandboxLimits(timeout_seconds=120, kill_after_seconds=5),
        watch_dir=outputs,
        watch_max_bytes=1_000_000,
        watch_interval_seconds=0.5,
    )
    result = await _sandbox().run(spec)
    assert result.killed is True
    assert result.kill_reason == "disk"
    assert result.timed_out is False
    assert result.duration_seconds < 60  # long before the wall clock
    _no_leftovers()


def test_sealed_probe_passes_end_to_end(tmp_path):
    """Image + mounts + uid + round-trip + `timeout` as PID 1, for real."""
    _sandbox().probe_sealed(workspace_root=tmp_path / "workspaces")
    assert list((tmp_path / "workspaces").iterdir()) == []
