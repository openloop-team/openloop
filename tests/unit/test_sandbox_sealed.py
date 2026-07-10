"""Unit tests for the sealed-exec primitive (Phase 0 — no docker needed).

Covers the four pre-code locks at the unit level:
lock 1 — arg building, drain semantics, timed_out classification;
lock 2 — named container, deadline label, no --init, probe failure modes;
lock 3 — limits flags, tmpfs, read-only rootfs, mount split.
(Lock 4 lives in test_sandbox_readout.py.)
"""

import asyncio
from pathlib import Path

import pytest

from openloop.sandbox import DockerSandbox, Mount, SandboxLimits, SealedSpec
from openloop.sandbox import runner as sandbox_runner
from openloop.sandbox.runner import (
    _classify_timed_out,
    _drain_capped,
    sweep_expired_sandboxes,
)


def _spec(**overrides) -> SealedSpec:
    defaults = dict(
        job_id="job1",
        command=("python", "-"),
        limits=SandboxLimits(
            timeout_seconds=300,
            kill_after_seconds=10,
            memory="512m",
            cpus=1.0,
            pids_limit=128,
            tmp_size="64m",
        ),
        mounts=(
            Mount(Path("/host/in"), "/workspace/inputs", read_only=True),
            Mount(Path("/host/out"), "/workspace/outputs"),
        ),
        stdin="print('hi')",
    )
    defaults.update(overrides)
    return SealedSpec(**defaults)


def _args(spec=None) -> list[str]:
    sandbox = DockerSandbox("img", kind="analysis")
    return sandbox._sealed_args(spec or _spec(), "openloop-job1-abc", 1234567890)


# ---------------------------------------------------------------- lock 2/3: argv


def test_sealed_args_never_pass_init():
    """--init would demote `timeout` to PID 2, where same-uid model code can
    kill -9 it — layer 1 silently gone. The hard constraint of lock 2."""
    assert "--init" not in _args()


def test_sealed_args_timeout_is_the_entrypoint():
    args = _args()
    assert args[args.index("--entrypoint") + 1] == "timeout"
    # -k (GNU + busybox compatible), then the deadline, then the command.
    tail = args[args.index("img") + 1 :]
    assert tail == ["-k", "10", "300", "python", "-"]


def test_sealed_args_name_and_labels():
    args = _args()
    assert args[args.index("--name") + 1] == "openloop-job1-abc"
    labels = [args[i + 1] for i, a in enumerate(args) if a == "--label"]
    assert "openloop.sandbox=analysis" in labels
    assert "openloop.deadline=1234567890" in labels


def test_sealed_args_mount_split_ro_inputs_rw_outputs():
    args = _args()
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
    assert "/host/in:/workspace/inputs:ro" in mounts
    assert "/host/out:/workspace/outputs" in mounts


def test_sealed_args_limits_and_hardening_flags():
    args = _args()
    assert "--read-only" in args
    assert args[args.index("--tmpfs") + 1] == "/tmp:size=64m"
    assert args[args.index("--pids-limit") + 1] == "128"
    assert args[args.index("--memory") + 1] == "512m"
    # No swap headroom unless explicitly granted: swap cap == memory cap.
    assert args[args.index("--memory-swap") + 1] == "512m"
    assert args[args.index("--cpus") + 1] == "1.0"
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--cap-drop") + 1] == "ALL"


def test_sealed_args_never_forward_environment():
    args = _args()
    assert "-e" not in args
    assert not any(a.startswith("--env") for a in args)


def test_sealed_args_stdin_only_when_needed():
    assert "-i" in _args(_spec(stdin="code"))
    assert "-i" not in _args(_spec(stdin=None))


def test_sealed_args_explicit_swap_headroom_is_honored():
    spec = _spec(
        limits=SandboxLimits(timeout_seconds=60, memory="256m", memory_swap="512m")
    )
    args = _args(spec)
    assert args[args.index("--memory-swap") + 1] == "512m"


# ------------------------------------------------------------- lock 1: drain


async def _feed_reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def test_drain_below_cap_keeps_everything():
    reader = await _feed_reader(b"hello")
    data, truncated = await _drain_capped(reader, cap=100)
    assert data == b"hello"
    assert truncated is False


async def test_drain_past_cap_retains_prefix_and_flags_truncation():
    reader = await _feed_reader(b"x" * 1000)
    data, truncated = await _drain_capped(reader, cap=10)
    assert data == b"x" * 10
    assert truncated is True


async def test_drain_reads_to_eof_past_the_cap():
    """The load-bearing half of lock 1: the pipe is drained, not abandoned —
    an abandoned pipe blocks the child's write() and manufactures a phantom
    timeout. EOF must be reached even when the cap was hit long before."""
    reader = asyncio.StreamReader()
    for _ in range(50):
        reader.feed_data(b"y" * 1000)
    reader.feed_eof()
    data, truncated = await _drain_capped(reader, cap=5)
    assert data == b"y" * 5 and truncated is True
    assert reader.at_eof()


# ---------------------------------------------------- lock 1/2: classification


@pytest.mark.parametrize(
    ("kill_reason", "elapsed", "timeout", "exit_code", "expected"),
    [
        # The runner's own docker kill for timeout — first-hand knowledge.
        ("timeout", 100.0, 300.0, 137, True),
        # Disk watchdog kill is a kill, not a timeout.
        ("disk", 5.0, 300.0, 137, False),
        # Layer-1 self-termination: ran to the deadline, nonzero exit —
        # classified whatever the numeric code (124 GNU, 137 kill-after,
        # 143 busybox...).
        (None, 300.5, 300.0, 124, True),
        (None, 300.5, 300.0, 137, True),
        (None, 300.5, 300.0, 143, True),
        # Fast nonzero exit = the code's own failure, not a timeout.
        (None, 5.0, 300.0, 1, False),
        # Ran long but exited 0 = success that used its budget.
        (None, 301.0, 300.0, 0, False),
    ],
)
def test_timed_out_is_classified_never_decoded(
    kill_reason, elapsed, timeout, exit_code, expected
):
    assert (
        _classify_timed_out(kill_reason, elapsed, timeout, exit_code) is expected
    )


# ------------------------------------------------------------- lock 2: sweep


class _FakeDocker:
    """Records docker CLI calls; serves a canned `docker ps` listing."""

    def __init__(self, listing: str):
        self.listing = listing
        self.calls: list[tuple[str, ...]] = []

    async def __call__(self, *cmd: str) -> str:
        self.calls.append(cmd)
        if cmd[1] == "ps":
            return self.listing
        return ""


async def test_sweep_reaps_only_past_deadline_plus_grace():
    now = 1_000_000.0
    fake = _FakeDocker(
        "openloop-old-1\t999000\n"  # deadline long past (+grace) -> reap
        "openloop-live-2\t999990\n"  # within grace -> skip
    )
    reaped = await sweep_expired_sandboxes(
        grace_seconds=120, now=lambda: now, runner=fake
    )
    assert reaped == ["openloop-old-1"]
    killed = [c for c in fake.calls if c[1] == "kill"]
    assert killed == [("docker", "kill", "openloop-old-1")]
    # rm -f belt-and-suspenders follows the kill.
    assert ("docker", "rm", "-f", "openloop-old-1") in fake.calls


async def test_sweep_never_touches_missing_or_malformed_deadlines():
    """Unlabeled containers may not be OpenLoop's at all; malformed labels are
    skipped, never reaped."""
    fake = _FakeDocker(
        "some-container\t\n"  # no deadline label
        "other-container\tnot-a-number\n"
    )
    reaped = await sweep_expired_sandboxes(
        grace_seconds=0, now=lambda: 9e12, runner=fake
    )
    assert reaped == []
    assert all(c[1] == "ps" for c in fake.calls)


async def test_sweep_filters_on_the_analysis_kind_label():
    fake = _FakeDocker("")
    await sweep_expired_sandboxes(runner=fake)
    ps = fake.calls[0]
    assert "label=openloop.sandbox=analysis" in ps


async def test_sweep_survives_a_dead_daemon():
    async def broken(*cmd):
        raise RuntimeError("daemon down")

    assert await sweep_expired_sandboxes(runner=broken) == []


async def test_sweep_is_idempotent_when_kill_races_another_sweeper():
    """Concurrent sweeps: the loser's kill/rm fail — suppressed, still safe."""

    class _Racy(_FakeDocker):
        async def __call__(self, *cmd):
            self.calls.append(cmd)
            if cmd[1] == "ps":
                return self.listing
            raise RuntimeError("No such container")

    fake = _Racy("openloop-old-1\t1000\n")
    reaped = await sweep_expired_sandboxes(
        grace_seconds=0, now=lambda: 9e9, runner=fake
    )
    assert reaped == ["openloop-old-1"]  # idempotent: already-gone is fine


# -------------------------------------------------- lock 2: probe fail modes


class _FakeCompleted:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _fake_sealed_probe(monkeypatch, *, returncode=0, writes=True, stderr=""):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if "run" in args:
            if writes:
                mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
                out = next(m for m in mounts if "/workspace/outputs" in m)
                Path(out.split(":", 1)[0], "probe").write_text("ok")
            return _FakeCompleted(returncode, stderr)
        return _FakeCompleted()

    monkeypatch.setattr("subprocess.run", fake_run)
    return calls


def test_probe_sealed_passes_and_rehearses_the_real_argv(monkeypatch, tmp_path):
    calls = _fake_sealed_probe(monkeypatch)
    DockerSandbox("img", kind="analysis").probe_sealed(workspace_root=tmp_path)
    run = next(c for c in calls if "run" in c)
    assert run[run.index("--entrypoint") + 1] == "timeout"
    assert "--init" not in run
    # Probe cleans up its throwaway workspace.
    assert list(tmp_path.iterdir()) == []


def test_probe_sealed_names_a_demoted_pid1(monkeypatch, tmp_path):
    _fake_sealed_probe(monkeypatch, returncode=41, writes=False, stderr="PID1=tini")
    with pytest.raises(sandbox_runner.SandboxUnavailable, match="PID 1"):
        DockerSandbox("img").probe_sealed(workspace_root=tmp_path)


def test_probe_sealed_names_writable_inputs(monkeypatch, tmp_path):
    _fake_sealed_probe(monkeypatch, returncode=43, writes=False)
    with pytest.raises(sandbox_runner.SandboxUnavailable, match="read-only"):
        DockerSandbox("img").probe_sealed(workspace_root=tmp_path)


def test_probe_sealed_detects_unshared_workspace_root(monkeypatch, tmp_path):
    _fake_sealed_probe(monkeypatch, returncode=0, writes=False)
    with pytest.raises(sandbox_runner.SandboxUnavailable, match="not\\s.*shared"):
        DockerSandbox("img").probe_sealed(workspace_root=tmp_path)
