"""The broker feature flag must fail closed until broker wiring lands.

`coding_worker_openhands_broker_enabled` exists so operators can opt into the
broker path, but the broker-backed adapter is not wired yet (arch step 4,
phases 2-3). Until it lands, honoring the flag by building the direct
in-process ``HardenedDockerWorkspace`` would silently hand back the very launch
path the broker replaces. `build_coding_worker` must instead disable the worker.
"""

from openloop.config import Settings
from openloop.wiring.builders import build_coding_worker


def test_builtin_worker_builds_with_flag_off():
    # Baseline: the default (flag off, builtin/host) yields a real worker, so the
    # flag-on assertion below is meaningful rather than vacuously None.
    settings = Settings(
        coding_worker_backend="builtin", coding_worker_sandbox="host"
    )
    assert build_coding_worker(settings) is not None


def test_broker_flag_on_disables_worker_until_wired():
    settings = Settings(
        coding_worker_backend="builtin",
        coding_worker_sandbox="host",
        coding_worker_openhands_broker_enabled=True,
    )
    # Fail closed: opting into the broker must not silently fall back to the
    # direct launch path.
    assert build_coding_worker(settings) is None
