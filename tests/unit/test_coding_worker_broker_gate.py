"""The OpenHands broker flag is opt-in and fails closed on any wrong combo.

`coding_worker_openhands_broker_enabled` only selects the broker path for the
openhands backend on the docker sandbox, and only when a broker handle was
actually composed. Every other combination disables the worker rather than
silently falling back to the direct in-process container launch path.
"""

import base64
import logging
from types import SimpleNamespace

import pytest

from openloop.config import Settings
from openloop.tools.openhands_broker_workspace import BrokerWorkspaceAdapter
from openloop.wiring import builders
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


@pytest.mark.parametrize("broker_mode", ["coprocess", "external"])
def test_broker_flag_on_without_handle_disables_worker(broker_mode):
    settings = Settings(
        broker_mode=broker_mode,
        coding_worker_backend="openhands",
        coding_worker_sandbox="docker",
        coding_worker_openhands_broker_enabled=True,
        coding_worker_openhands_state_master_key=_MASTER_KEY,
    )
    # Flag on but no broker handle composed → fail closed, no direct fallback.
    assert build_coding_worker(settings, broker_handle=None) is None


class _Ingress:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.max_ages: list[int] = []

    def prune_stale(self, *, max_age_seconds: int) -> int:
        self.max_ages.append(max_age_seconds)
        if self.error is not None:
            raise self.error
        return 2


class _ExternalBrokerHandle:
    def __init__(self, ingress: _Ingress) -> None:
        self.client = object()
        self.loop = object()
        self.receipt_issuer = object()
        self.workspace_ingress = ingress
        self.owner = SimpleNamespace(tenant_id="openloop")
        self.reconciler = None
        self.checkpoint_store = object()

    def bind_checkpoint_store(self, artifact_store):
        return self.checkpoint_store


def _external_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        broker_mode="external",
        coding_worker_backend="openhands",
        coding_worker_sandbox="docker",
        coding_worker_openhands_broker_enabled=True,
        coding_worker_openhands_cold_resume_enabled=False,
        coding_worker_openhands_state_dir=str(tmp_path / "state"),
        coding_worker_openhands_state_master_key=_MASTER_KEY,
    )


def test_external_handle_without_reconciler_builds_worker_and_sweeps_ingress(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(builders.OpenHandsCodingWorker, "probe", lambda self: None)
    ingress = _Ingress()
    handle = _ExternalBrokerHandle(ingress)

    worker = build_coding_worker(_external_settings(tmp_path), broker_handle=handle)

    assert worker is not None
    assert isinstance(worker._docker_adapter, BrokerWorkspaceAdapter)
    assert worker._docker_adapter._checkpoint_store is handle.checkpoint_store
    assert handle.reconciler is None
    assert ingress.max_ages == [builders._INGRESS_STALE_SWEEP_SECONDS]


def test_ingress_sweep_failure_does_not_disable_external_worker(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(builders.OpenHandsCodingWorker, "probe", lambda self: None)
    ingress = _Ingress(error=RuntimeError("sweep failed"))
    handle = _ExternalBrokerHandle(ingress)

    with caplog.at_level(logging.WARNING, logger="openloop"):
        worker = build_coding_worker(
            _external_settings(tmp_path), broker_handle=handle
        )

    assert worker is not None
    assert ingress.max_ages == [builders._INGRESS_STALE_SWEEP_SECONDS]
    assert "broker workspace ingress stale sweep failed" in caplog.text
