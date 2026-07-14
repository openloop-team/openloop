"""Real OpenHands 1.31.0 adapter-shape test with Docker execution faked."""

from __future__ import annotations

import base64
import importlib.util
import os
import subprocess

import pytest

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

if importlib.util.find_spec("openhands.workspace") is None:
    pytest.skip("OpenHands optional dependency is not installed", allow_module_level=True)

from openhands.workspace import DockerWorkspace

from openloop.tools.openhands_docker import HardenedDockerWorkspace
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout


def _keys():
    encoded = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    return OpenHandsKeyDeriver.from_base64(encoded, master_key_id="key-v1")


def test_real_131_workspace_seam_preserves_authenticated_client(
    monkeypatch, tmp_path
):
    calls = []

    def runner(command, environment, timeout):
        calls.append((command, environment, timeout))
        if command[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "container-id\n", "")

    # Avoid a real health request; this test targets the released class shape,
    # Pydantic initialization, command policy, and client credential handoff.
    monkeypatch.setattr(DockerWorkspace, "_wait_for_health", lambda self, **kw: None)

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    adapter = HardenedDockerWorkspace(
        layout=OpenHandsStateLayout(tmp_path / "state"),
        keys=_keys(),
        command_runner=runner,
        port_allocator=lambda: 32123,
    )

    target = adapter.create(checkout, "job-1")
    try:
        run_command, run_environment, _ = calls[1]
        session_key = run_environment["OH_SESSION_API_KEYS_0"]
        assert target.api_key == session_key
        assert target.host == "http://127.0.0.1:32123"
        assert target.model_dump().get("api_key") is None
        assert session_key not in repr(target)
        assert session_key not in " ".join(run_command)
        assert "127.0.0.1:32123:8000" in run_command
    finally:
        # The runner returned a fake ID; suppress upstream's destructor cleanup.
        target._container_id = None
