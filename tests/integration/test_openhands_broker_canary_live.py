"""Opt-in Phase 5 canary against broker-owned sibling Docker containers.

The controller runs in a small local Linux image so native relay UDS traffic
stays within one kernel on Docker Desktop. Current source is mounted read-only;
the model endpoint remains deterministic and provider-free on the host.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from tests.integration.test_openhands_cold_resume_live import _fake_openai


pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.serial,
    pytest.mark.skipif(
        os.environ.get("OPENLOOP_RUN_BROKER_CANARY") != "1",
        reason="set OPENLOOP_RUN_BROKER_CANARY=1 for the Phase 5 Docker canary",
    ),
]

_CANARY_IMAGE = "openloop-phase5-canary:local"


def _build_canary_image(workspace: Path) -> None:
    result = subprocess.run(
        [
            "docker",
            "build",
            "--file",
            str(workspace / "tests/support/Dockerfile.phase5-canary"),
            "--tag",
            _CANARY_IMAGE,
            str(workspace),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def run_phase5_checkpoint_park_resume_finalize_real_docker():
    """Execute the reusable real-Docker lifecycle without pytest gating."""
    workspace = Path(__file__).resolve().parents[2]
    suffix = uuid.uuid4().hex[:12]
    network = f"olp5-canary-{suffix}"
    proxy = f"olp5-docker-proxy-{suffix}"
    controller = f"olp5-controller-{suffix}"
    volume = f"p5{suffix[:6]}"
    shared = Path(f"/var/lib/docker/volumes/{volume}/_data")
    try:
        _build_canary_image(workspace)
        created_volume = subprocess.run(
            ["docker", "volume", "create", volume],
            check=False,
            capture_output=True,
            text=True,
        )
        assert created_volume.returncode == 0, created_volume.stderr
        initialized_volume = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "0:0",
                "--volume",
                f"{volume}:/shared",
                _CANARY_IMAGE,
                "-c",
                "import os; os.chmod('/shared', 0o777)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert initialized_volume.returncode == 0, initialized_volume.stderr
        created = subprocess.run(
            ["docker", "network", "create", network],
            check=False,
            capture_output=True,
            text=True,
        )
        assert created.returncode == 0, created.stderr
        started = subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                proxy,
                "--user",
                "0:0",
                "--network",
                network,
                "--network-alias",
                "docker-proxy",
                "--volume",
                "/var/run/docker.sock:/var/run/docker.sock",
                "--mount",
                f"type=bind,src={workspace},dst=/workspace/openloop,readonly",
                _CANARY_IMAGE,
                "/workspace/openloop/tests/support/docker_socket_proxy.py",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert started.returncode == 0, started.stderr
        for _ in range(100):
            logs = subprocess.run(
                ["docker", "logs", proxy],
                check=False,
                capture_output=True,
                text=True,
            )
            if "PHASE5_DOCKER_PROXY_READY" in logs.stdout:
                break
            time.sleep(0.05)
        else:
            raise AssertionError((logs.stderr + logs.stdout)[-4000:])

        with _fake_openai() as fake:
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                controller,
                "--user",
                "1000:1000",
                "--network",
                network,
                "--mount",
                f"type=bind,src={workspace},dst=/workspace/openloop,readonly",
                "--mount",
                f"type=volume,src={volume},dst={shared}",
                "--env",
                "PYTHONPATH=/workspace/openloop/src:/workspace/openloop",
                "--env",
                "DOCKER_HOST=tcp://docker-proxy:2375",
                "--env",
                f"OPENLOOP_CANARY_MODEL_PORT={fake.server_port}",
                "--env",
                f"OPENLOOP_CANARY_SHARED_ROOT={shared}",
                _CANARY_IMAGE,
                "/workspace/openloop/tests/support/phase5_canary_runner.py",
            ]
            topology = os.environ.get("OPENLOOP_CANARY_BROKER_MODE")
            if topology:
                command[command.index(_CANARY_IMAGE):command.index(_CANARY_IMAGE)] = [
                    "--env",
                    f"OPENLOOP_CANARY_BROKER_MODE={topology}",
                ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=360,
            )

        assert result.returncode == 0, (result.stderr + result.stdout)[-8000:]
        proof = next(
            line.removeprefix("PHASE5_CANARY_OK ")
            for line in result.stdout.splitlines()
            if line.startswith("PHASE5_CANARY_OK ")
        )
        payload = json.loads(proof)
        assert payload["status"] == "terminal"
        assert payload["topology"] == os.environ.get(
            "OPENLOOP_CANARY_BROKER_MODE", "coprocess"
        )
        assert payload["generations"][-1] == 2
        assert "parking" in payload["statuses"]
        assert "finalizing" in payload["statuses"]
        assert payload["statuses"][-1] == "terminal"
        assert fake.agent_calls == 2
    finally:
        subprocess.run(
            ["docker", "stop", controller], check=False, capture_output=True
        )
        subprocess.run(
            ["docker", "stop", proxy], check=False, capture_output=True
        )
        subprocess.run(
            ["docker", "network", "rm", network],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["docker", "volume", "rm", volume],
            check=False,
            capture_output=True,
        )


def test_phase5_checkpoint_park_resume_finalize_real_docker():
    run_phase5_checkpoint_park_resume_finalize_real_docker()
