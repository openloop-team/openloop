"""Bounded subprocess cleanup drains pipes and never skips later processes."""

from __future__ import annotations

import io
import subprocess

import pytest

from tests.support.processes import cleanup_process, cleanup_processes


class _FakeProcess:
    def __init__(
        self,
        *,
        running: bool = True,
        time_out_once: bool = False,
        error: Exception | None = None,
        terminate_error: Exception | None = None,
    ) -> None:
        self.returncode = None if running else 0
        self.time_out_once = time_out_once
        self.error = error
        self.terminate_error = terminate_error
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("stdout")
        self.stderr = io.StringIO("stderr")
        self.calls: list[str] = []

    def poll(self):
        self.calls.append("poll")
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        if self.terminate_error is not None:
            raise self.terminate_error

    def kill(self):
        self.calls.append("kill")

    def communicate(self, *, timeout):
        self.calls.append(f"communicate:{timeout}")
        if self.time_out_once:
            self.time_out_once = False
            raise subprocess.TimeoutExpired("fake", timeout)
        if self.error is not None:
            raise self.error
        self.returncode = -9 if "kill" in self.calls else 0
        self.stdin.close()
        self.stdout.close()
        self.stderr.close()
        return "stdout", "stderr"


def test_cleanup_drains_and_reaps_an_already_exited_process():
    process = _FakeProcess(running=False)

    cleanup_process(process, timeout=0.25)

    assert "terminate" not in process.calls
    assert process.calls[-1] == "communicate:0.25"
    assert process.stdout.closed
    assert process.stderr.closed


def test_cleanup_terminates_then_drains_a_running_process():
    process = _FakeProcess()

    cleanup_process(process, timeout=0.25)

    assert process.calls == ["poll", "poll", "terminate", "communicate:0.25"]
    assert process.returncode == 0
    assert process.stdout.closed


def test_cleanup_timeout_kills_then_drains_and_is_idempotent():
    process = _FakeProcess(time_out_once=True)

    cleanup_process(process, timeout=0.25)
    calls_after_first_cleanup = list(process.calls)
    cleanup_process(process, timeout=0.25)

    assert process.calls == calls_after_first_cleanup + ["poll"]
    assert calls_after_first_cleanup == [
        "poll",
        "poll",
        "terminate",
        "communicate:0.25",
        "kill",
        "communicate:0.25",
    ]
    assert process.returncode == -9
    assert process.stdout.closed


def test_multi_process_cleanup_attempts_all_and_raises_exception_group():
    first = _FakeProcess(
        terminate_error=PermissionError("first cleanup failed")
    )
    second = _FakeProcess(error=RuntimeError("second cleanup failed"))

    with pytest.raises(ExceptionGroup) as captured:
        cleanup_processes(first, second, timeout=0.25)

    assert [str(error) for error in captured.value.exceptions] == [
        "first cleanup failed",
        "second cleanup failed",
    ]
    assert "terminate" in first.calls
    assert "terminate" in second.calls
    assert first.stdout.closed
    assert second.stdout.closed
