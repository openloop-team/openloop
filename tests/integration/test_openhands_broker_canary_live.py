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

from openloop.tools.openhands_relay_profile import DEFAULT_HAPROXY_RELAY_IMAGE
from tests.support.fake_openai import fake_openai

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
_ADAPTER_DATA_PATH = "/run/openloop-docker"
_ADAPTER_HEALTH_PATH = "/run/openloop-health"
_ADAPTER_CONFIG_PATH = "/usr/local/etc/haproxy/haproxy.cfg"
_ADAPTER_HEALTH_COMMAND = (
    "printf 'GET /healthz HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n' "
    "| socat -t 3 - UNIX-CONNECT:/run/openloop-health/health.sock "
    "| grep -q '^HTTP/1\\.[01] 200'"
)


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
    adapter = f"olp5-docker-adapter-{suffix}"
    controller = f"olp5-controller-{suffix}"
    volume = f"p5{suffix[:6]}"
    adapter_volume = f"olp5-docker-adapter-{suffix}"
    shared = Path(f"/var/lib/docker/volumes/{volume}/_data")
    try:
        _build_canary_image(workspace)
        pulled_adapter = subprocess.run(
            ["docker", "pull", DEFAULT_HAPROXY_RELAY_IMAGE],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert pulled_adapter.returncode == 0, pulled_adapter.stderr[-4000:]
        created_volume = subprocess.run(
            ["docker", "volume", "create", volume],
            check=False,
            capture_output=True,
            text=True,
        )
        assert created_volume.returncode == 0, created_volume.stderr
        created_adapter_volume = subprocess.run(
            ["docker", "volume", "create", adapter_volume],
            check=False,
            capture_output=True,
            text=True,
        )
        assert created_adapter_volume.returncode == 0, created_adapter_volume.stderr
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
                adapter,
                "--user",
                "0:10777",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--mount",
                (
                    "type=bind,"
                    f"src={os.environ.get('DOCKER_SOCKET', '/var/run/docker.sock')},"
                    "dst=/var/run/docker.sock"
                ),
                "--mount",
                (
                    "type=bind,"
                    f"src={workspace / 'ops/docker-socket-adapter/haproxy.cfg'},"
                    f"dst={_ADAPTER_CONFIG_PATH},readonly"
                ),
                "--mount",
                (f"type=volume,src={adapter_volume},dst={_ADAPTER_DATA_PATH}"),
                "--tmpfs",
                (f"{_ADAPTER_HEALTH_PATH}:rw,nosuid,nodev,noexec,size=64k,mode=0700"),
                DEFAULT_HAPROXY_RELAY_IMAGE,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert started.returncode == 0, started.stderr
        for _ in range(100):
            health = subprocess.run(
                [
                    "docker",
                    "exec",
                    adapter,
                    "sh",
                    "-ec",
                    _ADAPTER_HEALTH_COMMAND,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if health.returncode == 0:
                break
            time.sleep(0.1)
        else:
            logs = subprocess.run(
                ["docker", "logs", adapter],
                check=False,
                capture_output=True,
                text=True,
            )
            raise AssertionError((logs.stderr + logs.stdout)[-4000:])
        socket_state = subprocess.run(
            [
                "docker",
                "exec",
                adapter,
                "stat",
                "-c",
                "%u:%g:%a:%F",
                f"{_ADAPTER_DATA_PATH}/docker.sock",
                f"{_ADAPTER_HEALTH_PATH}/health.sock",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert socket_state.returncode == 0, socket_state.stderr
        assert socket_state.stdout.splitlines() == [
            "0:10777:660:socket",
            "0:10777:600:socket",
        ]
        inspected = subprocess.run(
            ["docker", "inspect", adapter],
            check=False,
            capture_output=True,
            text=True,
        )
        assert inspected.returncode == 0, inspected.stderr
        adapter_document = json.loads(inspected.stdout)[0]
        adapter_host = adapter_document["HostConfig"]
        assert adapter_document["Config"]["User"] == "0:10777"
        assert adapter_host["NetworkMode"] == "none"
        assert adapter_host["ReadonlyRootfs"] is True
        assert adapter_host["CapDrop"] == ["ALL"]
        assert adapter_host["CapAdd"] is None
        assert adapter_host["SecurityOpt"] == ["no-new-privileges"]
        assert adapter_host["PortBindings"] == {}
        assert not any(
            value.startswith(("BROKER_", "OPENAI_API_KEY=", "SLACK_"))
            for value in (adapter_document["Config"].get("Env") or [])
        )

        with fake_openai() as fake:
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                controller,
                "--user",
                "1000:10777",
                "--network",
                network,
                "--mount",
                f"type=bind,src={workspace},dst=/workspace/openloop,readonly",
                "--mount",
                f"type=volume,src={volume},dst={shared}",
                "--mount",
                (f"type=volume,src={adapter_volume},dst={_ADAPTER_DATA_PATH},readonly"),
                "--env",
                "PYTHONPATH=/workspace/openloop/src:/workspace/openloop",
                "--env",
                (f"DOCKER_HOST=unix://{_ADAPTER_DATA_PATH}/docker.sock"),
                "--env",
                f"OPENLOOP_CANARY_MODEL_PORT={fake.server_port}",
                "--env",
                f"OPENLOOP_CANARY_SHARED_ROOT={shared}",
                _CANARY_IMAGE,
                "/workspace/openloop/tests/support/phase5_canary_runner.py",
            ]
            topology = os.environ.get("OPENLOOP_CANARY_BROKER_MODE")
            if topology:
                command[command.index(_CANARY_IMAGE) : command.index(_CANARY_IMAGE)] = [
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
        assert "/var/run/docker.sock" not in command
        assert "tcp://docker-proxy:2375" not in command
        proof = next(
            line.removeprefix("PHASE5_CANARY_OK ")
            for line in result.stdout.splitlines()
            if line.startswith("PHASE5_CANARY_OK ")
        )
        payload = json.loads(proof)
        assert payload["status"] == "terminal"
        assert payload["topology"] == os.environ.get(
            "OPENLOOP_CANARY_BROKER_MODE", "subprocess"
        )
        assert payload["generations"][-1] == 2
        assert "parking" in payload["statuses"]
        assert "finalizing" in payload["statuses"]
        assert payload["statuses"][-1] == "terminal"
        assert fake.agent_calls == 2
    finally:
        subprocess.run(["docker", "stop", controller], check=False, capture_output=True)
        subprocess.run(["docker", "stop", adapter], check=False, capture_output=True)
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
        subprocess.run(
            ["docker", "volume", "rm", adapter_volume],
            check=False,
            capture_output=True,
        )


def test_phase5_checkpoint_park_resume_finalize_real_docker():
    run_phase5_checkpoint_park_resume_finalize_real_docker()
