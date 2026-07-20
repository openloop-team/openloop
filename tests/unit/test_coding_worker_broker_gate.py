"""The OpenHands broker flag is opt-in and fails closed on any wrong combo.

`coding_worker_openhands_broker_enabled` only selects the broker path for the
openhands backend on the docker sandbox, and only when a broker handle was
actually composed. Every other combination disables the worker rather than
silently falling back to the direct in-process container launch path.
"""

import base64

from openloop.config import Settings
from openloop.wiring.builders import build_coding_worker

_MASTER_KEY = base64.b64encode(b"x" * 32).decode()


def test_builtin_worker_builds_with_flag_off():
    # Baseline: the default (flag off, builtin/host) yields a real worker.
    settings = Settings(
        coding_worker_backend="builtin", coding_worker_sandbox="host"
    )
    assert build_coding_worker(settings) is not None


def test_broker_flag_requires_openhands_backend():
    settings = Settings(
        coding_worker_backend="builtin",
        coding_worker_sandbox="host",
        coding_worker_openhands_broker_enabled=True,
    )
    assert build_coding_worker(settings) is None


def test_broker_flag_requires_docker_sandbox():
    settings = Settings(
        coding_worker_backend="openhands",
        coding_worker_sandbox="host",
        coding_worker_openhands_broker_enabled=True,
        coding_worker_openhands_cold_resume_enabled=False,
    )
    assert build_coding_worker(settings) is None


def test_broker_flag_on_without_handle_disables_worker():
    settings = Settings(
        coding_worker_backend="openhands",
        coding_worker_sandbox="docker",
        coding_worker_openhands_broker_enabled=True,
        coding_worker_openhands_state_master_key=_MASTER_KEY,
    )
    # Flag on but no broker handle composed → fail closed, no direct fallback.
    assert build_coding_worker(settings, broker_handle=None) is None
