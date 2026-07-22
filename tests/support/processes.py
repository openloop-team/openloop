"""Bounded, pipe-draining cleanup for subprocesses owned by tests."""

from __future__ import annotations

import math
import subprocess
from typing import Any


DEFAULT_PROCESS_CLEANUP_TIMEOUT_SECONDS = 5.0


def _timeout(value: float) -> float:
    try:
        selected = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("cleanup timeout must be positive and finite") from error
    if isinstance(value, bool) or not math.isfinite(selected) or selected <= 0:
        raise ValueError("cleanup timeout must be positive and finite")
    return selected


def _pipes_closed(process: subprocess.Popen[Any]) -> bool:
    return all(
        stream is None or stream.closed
        for stream in (process.stdin, process.stdout, process.stderr)
    )


def _close_pipes(process: subprocess.Popen[Any]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def cleanup_process(
    process: subprocess.Popen[Any] | None,
    *,
    timeout: float = DEFAULT_PROCESS_CLEANUP_TIMEOUT_SECONDS,
) -> None:
    """Terminate, drain, kill if needed, and reap one test-owned process."""
    if process is None:
        return
    selected_timeout = _timeout(timeout)
    try:
        if process.poll() is not None and _pipes_closed(process):
            return

        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.communicate(timeout=selected_timeout)
            return
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass

        process.communicate(timeout=selected_timeout)
    except BaseException:
        _close_pipes(process)
        raise


def cleanup_processes(
    *processes: subprocess.Popen[Any] | None,
    timeout: float = DEFAULT_PROCESS_CLEANUP_TIMEOUT_SECONDS,
) -> None:
    """Clean every distinct process, then raise all cleanup failures together."""
    errors: list[Exception] = []
    seen: set[int] = set()
    for process in processes:
        if process is None or id(process) in seen:
            continue
        seen.add(id(process))
        try:
            cleanup_process(process, timeout=timeout)
        except Exception as error:  # noqa: BLE001 - aggregate after all attempts
            errors.append(error)
    if errors:
        raise ExceptionGroup("subprocess cleanup failed", errors)


__all__ = [
    "DEFAULT_PROCESS_CLEANUP_TIMEOUT_SECONDS",
    "cleanup_process",
    "cleanup_processes",
]
