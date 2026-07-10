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
import contextlib
import logging
import math
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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
# teardown test can assert there are none). The value is the sandbox *kind*
# ("worker" = coding worker's diff-apply, "analysis" = sealed analysis runs);
# the deadline sweep filters on kind so it never touches containers that carry
# no deadline (coding worker) or aren't OpenLoop's at all (unlabeled).
_LABEL_KEY = "openloop.sandbox"
# Absolute unix-epoch deadline stamped on sealed runs (start + wall-clock
# timeout). The orphan sweep keys on it: reap-safety derives from the run's own
# contract, never from replica liveness (see docs/sealed-analysis-worker.md).
_DEADLINE_LABEL = "openloop.deadline"

# Layer-2 stagger: the controller's docker-kill backstop waits this long past
# the in-container deadline (+ kill-after), so a clean layer-1 self-exit
# normally wins and the kill path stays the exception.
_KILL_STAGGER_SECONDS = 15.0
# After a docker kill the CLI client should exit promptly; a client still
# running past this is wedged and gets killed itself (never hang the caller).
_POST_KILL_CLIENT_TIMEOUT_SECONDS = 30.0


@dataclass(slots=True, frozen=True)
class SandboxLimits:
    """Resource/time bounds for one sealed run (docs/sealed-analysis-worker.md §4.1).

    ``timeout_seconds`` is the wall-clock deadline, enforced three times from
    this one value: in-container (``timeout`` as PID 1), the controller's
    docker-kill backstop, and the deadline sweep. ``memory`` must be sized as
    heap **plus** ``tmp_size`` — tmpfs pages count against the memory limit.
    ``stream_cap_bytes`` bounds what is *retained* per output stream; the pipes
    are always drained to EOF (see :func:`_drain_capped`).
    """

    timeout_seconds: float
    kill_after_seconds: float = 10.0
    memory: str | None = None  # docker size string, e.g. "512m"
    memory_swap: str | None = None  # default: same as memory (no swap headroom)
    cpus: float | None = None
    pids_limit: int | None = None
    read_only_rootfs: bool = True
    tmp_size: str | None = "64m"  # tmpfs /tmp; None only for loose profiles
    stream_cap_bytes: int = 262_144


@dataclass(slots=True, frozen=True)
class Mount:
    """One bind mount for a sealed run. ``inputs/`` is read-only by contract."""

    source: Path
    target: str
    read_only: bool = False


@dataclass(slots=True, frozen=True)
class SealedSpec:
    """One sealed execution: command + mounts + limits, no env, no network.

    ``job_id`` rides the container name (kill handle) — the name gets a fresh
    uuid per attempt so re-drives never collide. The optional disk watchdog
    (``watch_dir``/``watch_max_bytes``) is best-effort mitigation, not a
    guarantee: no docker flag caps a bind mount (§ lock 3).
    """

    job_id: str
    command: tuple[str, ...]
    limits: SandboxLimits
    mounts: tuple[Mount, ...] = ()
    stdin: str | None = None
    watch_dir: Path | None = None
    watch_max_bytes: int | None = None
    watch_interval_seconds: float = 2.0


@dataclass(slots=True)
class SandboxResult:
    """What one sealed run produced. A nonzero exit is data, not an exception.

    ``killed`` = this runner issued the layer-2 ``docker kill`` (first-hand
    knowledge). ``timed_out`` = deadline classification — **never** decoded
    from the numeric exit code (GNU ``timeout`` reports 124 on the polite path
    but 137 when ``--kill-after`` fires; busybox has used 128+signal), see
    :func:`_classify_timed_out`. ``exit_code`` is recorded verbatim for
    diagnostics only. ``kill_reason`` distinguishes the watchdog's disk kill
    from the timeout backstop.
    """

    exit_code: int
    stdout: str
    stderr: str
    killed: bool
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    duration_seconds: float
    kill_reason: str | None = None  # "timeout" | "disk" | None


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
        kind: str = "worker",
    ) -> None:
        self.image = image
        # "none" = default-deny egress. Point this at a user-defined network
        # fronted by an egress proxy to move to an allowlist model later.
        self.network = network
        self._docker = docker_bin
        # Sandbox kind, stamped as the openloop.sandbox label value. "worker"
        # (default) = the coding worker's diff-apply; "analysis" = sealed runs.
        self.kind = kind

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
        self._assert_docker_usable()

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
            "--label", f"{_LABEL_KEY}={self.kind}",
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

    # ------------------------------------------------------------------
    # Sealed runs (analysis worker, Phase 0 — docs/sealed-analysis-worker.md)
    # ------------------------------------------------------------------

    def _sealed_args(
        self, spec: SealedSpec, name: str, deadline_epoch: int
    ) -> list[str]:
        """``docker run`` argv for one sealed run.

        Everything the four Phase 0 locks pin lives here, in one place:
        named container (the kill handle), kind + deadline labels, no env,
        the mount split, resource limits, and ``timeout`` as the entrypoint.

        HARD CONSTRAINT: never add ``--init``. Layer-1 deadline enforcement is
        the kernel's PID-1 special-casing — with ``--init``, tini takes PID 1,
        ``timeout`` drops to PID 2, and model code (same uid via ``--user``)
        can ``kill -9`` it, outliving its deadline. The sealed probe asserts
        ``/proc/1/comm`` == ``timeout`` so a future ``--init`` fails at boot.
        """
        limits = spec.limits
        args = [self._docker, "run", "--rm", "--name", name]
        if spec.stdin is not None:
            args.append("-i")
        args += [
            "--label", f"{_LABEL_KEY}={self.kind}",
            "--label", f"{_DEADLINE_LABEL}={deadline_epoch}",
            "--network", self.network,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
        ]
        if limits.read_only_rootfs:
            args.append("--read-only")
        if limits.tmp_size:
            args += ["--tmpfs", f"/tmp:size={limits.tmp_size}"]
        if limits.pids_limit is not None:
            args += ["--pids-limit", str(limits.pids_limit)]
        if limits.memory:
            # No swap headroom unless explicitly granted: OOM at the cap.
            args += [
                "--memory", limits.memory,
                "--memory-swap", limits.memory_swap or limits.memory,
            ]
        if limits.cpus is not None:
            args += ["--cpus", str(limits.cpus)]
        if sys.platform != "win32":
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for mount in spec.mounts:
            suffix = ":ro" if mount.read_only else ""
            args += ["-v", f"{mount.source}:{mount.target}{suffix}"]
        # cwd = /tmp: the one always-writable path under a read-only rootfs
        # (incidental cwd writes by generated code shouldn't crash the run).
        args += ["-w", "/tmp"]
        # `timeout` IS the entrypoint (PID 1): -k works on GNU and busybox.
        args += [
            "--entrypoint", "timeout",
            self.image,
            "-k", str(int(math.ceil(limits.kill_after_seconds))),
            str(int(math.ceil(limits.timeout_seconds))),
            *spec.command,
        ]
        return args

    async def run(self, spec: SealedSpec) -> SandboxResult:
        """Execute one sealed command; a nonzero exit is data, not an exception.

        Capture is bounded-retention / unbounded-drain: both pipes are read
        concurrently and to EOF — past the cap, bytes are discarded, never
        left unread, so a chatty-but-successful run can't wedge on a full
        pipe buffer and get killed by its own deadline (the phantom-timeout
        trap). The deadline itself is enforced in layers: in-container
        ``timeout`` (PID 1) at T, this runner's ``docker kill`` at
        T + kill-after + stagger, the label sweep at T + grace.
        """
        if not spec.command:
            raise ValueError("empty sealed command")
        name = f"openloop-{spec.job_id}-{uuid.uuid4().hex[:8]}"
        deadline_epoch = int(time.time() + spec.limits.timeout_seconds)
        args = self._sealed_args(spec, name, deadline_epoch)

        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=(
                asyncio.subprocess.PIPE
                if spec.stdin is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        cap = spec.limits.stream_cap_bytes
        out_task = asyncio.create_task(_drain_capped(proc.stdout, cap))
        err_task = asyncio.create_task(_drain_capped(proc.stderr, cap))
        feed_task = (
            asyncio.create_task(_feed_stdin(proc, spec.stdin))
            if spec.stdin is not None
            else None
        )
        wait_task = asyncio.create_task(proc.wait())
        watch_task = (
            asyncio.create_task(
                _watch_disk(
                    spec.watch_dir,
                    spec.watch_max_bytes,
                    spec.watch_interval_seconds,
                )
            )
            if spec.watch_dir is not None and spec.watch_max_bytes is not None
            else None
        )

        client_budget = (
            math.ceil(spec.limits.timeout_seconds)
            + math.ceil(spec.limits.kill_after_seconds)
            + _KILL_STAGGER_SECONDS
        )
        kill_reason: str | None = None
        try:
            racers = {wait_task} | ({watch_task} if watch_task else set())
            done, _ = await asyncio.wait(
                racers,
                timeout=client_budget,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task not in done:
                # Watchdog breach, or the client outlived even the staggered
                # budget (layer 1 failed). Kill by name — cancelling the CLI
                # client would NOT stop the container.
                kill_reason = "disk" if watch_task in done else "timeout"
                await self._kill_container(name)
                try:
                    await asyncio.wait_for(
                        wait_task, timeout=_POST_KILL_CLIENT_TIMEOUT_SECONDS
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    # Container is gone but the CLI is wedged: kill the client
                    # itself rather than hang the caller.
                    proc.kill()
                    await wait_task
        finally:
            if watch_task is not None:
                watch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watch_task
            if feed_task is not None:
                with contextlib.suppress(Exception):
                    await feed_task

        # The client has exited, so both pipes are at EOF; drains finish now.
        stdout, stdout_truncated = await out_task
        stderr, stderr_truncated = await err_task
        elapsed = time.monotonic() - start
        exit_code = wait_task.result()
        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            killed=kill_reason is not None,
            timed_out=_classify_timed_out(
                kill_reason, elapsed, spec.limits.timeout_seconds, exit_code
            ),
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            duration_seconds=elapsed,
            kill_reason=kill_reason,
        )

    async def _kill_container(self, name: str) -> None:
        """Layer-2 backstop: kill by name, then ``rm -f`` belt-and-suspenders.

        Every step is best-effort with its own timeout — a wedged daemon must
        not hang the orchestrator on the *cleanup* path, and errors are
        expected when layer 1 already reaped the container (``--rm``).
        """
        for sub in (("kill", name), ("rm", "-f", name)):
            try:
                await asyncio.wait_for(
                    _run(self._docker, *sub), timeout=10
                )
            except Exception:  # noqa: BLE001 — already-gone is the normal case
                logger.debug("docker %s %s failed (best-effort)", sub[0], name)

    # Generous: the first probe on a fresh host pulls the analysis image.
    _SEALED_PROBE_TIMEOUT_SECONDS = 240

    # Exit codes the probe payload uses to name its failure mode precisely.
    _PROBE_SCRIPT = (
        'comm="$(cat /proc/1/comm)"; '
        'if [ "$comm" != "timeout" ]; then echo "PID1=$comm" >&2; exit 41; fi; '
        "python -c 'open(\"/workspace/outputs/probe\",\"w\").write(\"ok\")' "
        "|| exit 42; "
        "echo x > /workspace/inputs/leak 2>/dev/null && exit 43; "
        "exit 0"
    )

    def probe_sealed(self, workspace_root: Path | None = None) -> None:
        """Prove the WHOLE sealed path at boot: image, mounts, PID 1, round-trip.

        Mirrors :meth:`probe` (fail-closed, synchronous, dress-rehearsal
        posture) for the analysis worker's invocation shape. Asserts, with a
        pointed error each: the image runs with ``timeout`` as **PID 1** (a
        future ``--init`` fails here, at boot, instead of silently demoting
        layer-1 deadline enforcement), ``python`` can write ``outputs/``, the
        ``inputs/`` mount is genuinely read-only, and the write is visible
        host-side (the unshared-workspace-root pitfall).
        """
        import shutil as _shutil
        import subprocess
        import tempfile as _tempfile

        self._assert_docker_usable()

        if workspace_root is not None:
            workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            _tempfile.mkdtemp(prefix="openloop-sealed-probe-", dir=workspace_root)
        )
        try:
            inputs = workspace / "inputs"
            outputs = workspace / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            (inputs / "seed").write_text("s\n")
            spec = SealedSpec(
                job_id="probe",
                command=("sh", "-c", self._PROBE_SCRIPT),
                limits=SandboxLimits(
                    timeout_seconds=60,
                    memory="256m",
                    pids_limit=64,
                ),
                mounts=(
                    Mount(inputs, "/workspace/inputs", read_only=True),
                    Mount(outputs, "/workspace/outputs"),
                ),
            )
            name = f"openloop-sealed-probe-{uuid.uuid4().hex[:8]}"
            args = self._sealed_args(
                spec, name, int(time.time() + spec.limits.timeout_seconds)
            )
            try:
                completed = subprocess.run(
                    args, capture_output=True, text=True,
                    timeout=self._SEALED_PROBE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                raise SandboxUnavailable(
                    f"sealed probe run failed (image {self.image!r}): {exc}"
                ) from exc
            if completed.returncode == 41:
                raise SandboxUnavailable(
                    "sealed probe: PID 1 is not `timeout` "
                    f"({completed.stderr.strip()}). Is --init being passed? "
                    "Layer-1 deadline enforcement requires timeout as PID 1 "
                    "(docs/sealed-analysis-worker.md, Phase 0 lock 2)."
                )
            if completed.returncode == 42:
                raise SandboxUnavailable(
                    f"sealed probe: `python` in image {self.image!r} cannot "
                    "write /workspace/outputs — wrong image or uid mapping."
                )
            if completed.returncode == 43:
                raise SandboxUnavailable(
                    "sealed probe: the inputs/ mount is WRITABLE inside the "
                    "sandbox — the read-only mount split is not in effect."
                )
            if completed.returncode != 0:
                raise SandboxUnavailable(
                    "sealed probe run failed (image "
                    f"{self.image!r}, network {self.network!r}, exit "
                    f"{completed.returncode}): {completed.stderr.strip()}"
                )
            probe_file = outputs / "probe"
            if not probe_file.exists() or probe_file.read_text() != "ok":
                raise SandboxUnavailable(
                    "sealed probe wrote inside the container but the write is "
                    f"not visible at {outputs} — the workspace root is not "
                    "shared with the host. In a containerized deploy, the "
                    "workspace dir must be a host path mounted into the "
                    "runtime at the same location."
                )
        finally:
            _shutil.rmtree(workspace, ignore_errors=True)

    def _assert_docker_usable(self) -> None:
        """CLI + daemon ping with its own clear error (shared by both probes)."""
        import subprocess

        try:
            subprocess.run(
                [self._docker, "version", "--format", "{{.Server.Version}}"],
                check=True, capture_output=True, timeout=10,
            )
        except Exception as exc:
            raise SandboxUnavailable(
                f"docker is not usable ({exc}); refusing to run the worker "
                "unsandboxed"
            ) from exc


async def _drain_capped(
    stream: asyncio.StreamReader, cap: int
) -> tuple[bytes, bool]:
    """Bounded retention, unbounded drain (Phase 0 lock 1).

    Reads the pipe **to EOF** — past the cap, bytes are discarded, never left
    unread. A truncating read-then-stop is the trap: the pipe buffer fills,
    the child's ``write()`` blocks, and the deadline then kills a run whose
    computation already succeeded (a phantom timeout). Closing the read end
    early is equally wrong (the child eats EPIPE on its next print). Memory
    is O(cap); drain cost is bounded by the wall-clock deadline.
    """
    buf = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        if len(buf) < cap:
            take = min(len(chunk), cap - len(buf))
            buf.extend(chunk[:take])
            if take < len(chunk):
                truncated = True
        else:
            truncated = True
    return bytes(buf), truncated


async def _feed_stdin(proc: asyncio.subprocess.Process, data: str) -> None:
    """Write the script to the child's stdin and close it.

    A child that exits before reading everything (crash, in-container timeout)
    surfaces as BrokenPipe here — that is the child's story to tell through
    its exit code, never a feed error.
    """
    assert proc.stdin is not None
    try:
        proc.stdin.write(data.encode())
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(Exception):
            proc.stdin.close()


def _dir_bytes(root: Path) -> int:
    """Apparent size of a directory tree, hostile-content-safe: lstat only,
    never open — the tree belongs to model-authored code."""
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    total += st.st_size
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
        except OSError:
            continue
    return total


async def _watch_disk(root: Path, max_bytes: int, interval: float) -> str:
    """Best-effort during-run disk watchdog (Phase 0 lock 3).

    No docker flag caps a bind mount, so this poll bounds overshoot to roughly
    write-rate x poll-interval. It is mitigation, never containment — the
    post-hoc read-out cap is the guarantee. Returns when breached; the caller
    kills the container.
    """
    while True:
        await asyncio.sleep(interval)
        if _dir_bytes(root) > max_bytes:
            return "disk"


def _classify_timed_out(
    kill_reason: str | None,
    elapsed: float,
    timeout_seconds: float,
    exit_code: int,
) -> bool:
    """Deadline classification from the runner's own first-hand knowledge.

    Never decoded from the numeric exit code — the convention varies by binary
    (GNU 124 vs busybox 128+signal) AND by which signal ended the child (GNU
    reports 137 when --kill-after's KILL fires). A layer-1 self-termination is
    recognized as: ran at least the deadline and exited nonzero. A disk kill is
    a kill, not a timeout.
    """
    if kill_reason == "timeout":
        return True
    if kill_reason is not None:
        return False
    return elapsed >= timeout_seconds and exit_code != 0


async def sweep_expired_sandboxes(
    *,
    docker_bin: str = "docker",
    kind: str = "analysis",
    grace_seconds: float = 120.0,
    now: "Callable[[], float] | None" = None,
    runner: "Callable[..., Awaitable[str]] | None" = None,
) -> list[str]:
    """Reap sealed containers past their stamped deadline + grace (lock 2).

    Cleanup, not enforcement: layer 1 (``timeout`` as PID 1) already
    self-terminates the run and ``--rm`` self-removes it — this sweep mops up
    what a failed ``--rm`` or a wedged PID 1 left behind. Reap-safety derives
    from the run's own contract (the ``openloop.deadline`` label), never from
    replica liveness, so it is correct under any replica count and any
    coordination state, and idempotent under concurrent sweeps.

    NEVER touches a container without the deadline label: unlabeled containers
    may not be OpenLoop's at all, and coding-worker containers carry no
    deadline yet. Malformed labels are skipped, not reaped.
    """
    run = runner or _run
    current = (now or time.time)()
    try:
        listing = await run(
            docker_bin, "ps",
            "--filter", f"label={_LABEL_KEY}={kind}",
            "--format", '{{.Names}}\t{{.Label "' + _DEADLINE_LABEL + '"}}',
        )
    except Exception:  # noqa: BLE001 — a dead daemon means nothing to sweep
        logger.warning("sandbox sweep: docker ps failed", exc_info=True)
        return []
    reaped: list[str] = []
    for line in listing.splitlines():
        name, _, deadline_raw = line.strip().partition("\t")
        if not name:
            continue
        try:
            deadline = float(deadline_raw)
        except ValueError:
            continue  # no/malformed deadline label -> never touch it
        if current <= deadline + grace_seconds:
            continue
        for sub in (("kill", name), ("rm", "-f", name)):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(run(docker_bin, *sub), timeout=10)
        logger.info(
            "sandbox sweep: reaped expired container %s (deadline %s)",
            name, deadline_raw,
        )
        reaped.append(name)
    return reaped
