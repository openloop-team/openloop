"""Opt-in live proof for the HAProxy Docker-socket permission adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest

from openloop.tools.openhands_relay_profile import DEFAULT_HAPROXY_RELAY_IMAGE


pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.serial,
    pytest.mark.skipif(
        os.environ.get("OPENLOOP_RUN_DOCKER_SOCKET_ADAPTER_CANARY") != "1",
        reason=(
            "set OPENLOOP_RUN_DOCKER_SOCKET_ADAPTER_CANARY=1 for the "
            "Docker-socket adapter canary"
        ),
    ),
]

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "ops/docker-socket-adapter/haproxy.cfg"
CLIENT_IMAGE = "docker:27-cli"
DATA_PATH = "/run/openloop-docker"
DATA_SOCKET = f"{DATA_PATH}/docker.sock"
HEALTH_PATH = "/run/openloop-health"
HEALTH_SOCKET = f"{HEALTH_PATH}/health.sock"
RAW_SOCKET = "/var/run/docker.sock"
HEALTH_COMMAND = (
    "printf 'GET /healthz HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n' "
    "| socat -t 3 - UNIX-CONNECT:/run/openloop-health/health.sock "
    "| grep -q '^HTTP/1\\.[01] 200'"
)
RESOURCE_LABEL = "openloop.canary=docker-socket-adapter"


def _docker(
    *arguments: str,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("docker", *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _success(
    *arguments: str,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    result = _docker(*arguments, timeout=timeout)
    assert result.returncode == 0, (result.stderr + result.stdout)[-12000:]
    return result


def _start_adapter(
    *,
    name: str,
    volume: str,
    upstream: Path | None,
) -> None:
    arguments = [
        "run",
        "--detach",
        "--name",
        name,
        "--label",
        RESOURCE_LABEL,
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
            f"type=bind,src={CONFIG},"
            "dst=/usr/local/etc/haproxy/haproxy.cfg,readonly"
        ),
        "--mount",
        f"type=volume,src={volume},dst={DATA_PATH}",
        "--tmpfs",
        f"{HEALTH_PATH}:rw,nosuid,nodev,noexec,size=64k,mode=0700",
    ]
    if upstream is not None:
        arguments.extend(
            (
                "--mount",
                f"type=bind,src={upstream},dst={RAW_SOCKET}",
            )
        )
    arguments.append(DEFAULT_HAPROXY_RELAY_IMAGE)
    _success(*arguments)


def _wait_for_health(name: str) -> None:
    deadline = time.monotonic() + 30
    last = None
    while time.monotonic() < deadline:
        last = _docker("exec", name, "sh", "-ec", HEALTH_COMMAND, timeout=5)
        if last.returncode == 0:
            return
        time.sleep(0.25)
    assert last is not None
    logs = _docker("logs", name)
    raise AssertionError(
        (last.stderr + last.stdout + logs.stderr + logs.stdout)[-12000:]
    )


def _wait_for_health_socket(name: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = _docker("exec", name, "test", "-S", HEALTH_SOCKET, timeout=5)
        if result.returncode == 0:
            return
        time.sleep(0.1)
    logs = _docker("logs", name)
    raise AssertionError((logs.stderr + logs.stdout)[-12000:])


def _inspect(name: str) -> dict:
    return json.loads(_success("inspect", name).stdout)[0]


def _assert_adapter_hardening(document: dict) -> None:
    host = document["HostConfig"]
    config = document["Config"]

    assert config["User"] == "0:10777"
    assert host["NetworkMode"] == "none"
    assert host["ReadonlyRootfs"] is True
    assert host["CapDrop"] == ["ALL"]
    assert host["CapAdd"] is None
    assert host["SecurityOpt"] == ["no-new-privileges"]
    assert host["PortBindings"] == {}
    assert document["NetworkSettings"]["Ports"] == {}
    assert host["Tmpfs"] == {
        HEALTH_PATH: "rw,nosuid,nodev,noexec,size=64k,mode=0700"
    }
    environment = config.get("Env") or []
    assert not any(
        value.startswith(
            (
                "OPENAI_API_KEY=",
                "BROKER_",
                "DATABASE_URL=",
                "PGPASSWORD=",
                "SLACK_",
            )
        )
        for value in environment
    )


def test_haproxy_adapter_permissions_health_and_read_only_connectivity() -> None:
    suffix = uuid.uuid4().hex[:12]
    adapter = f"openloop-docker-adapter-{suffix}"
    unavailable = f"openloop-docker-adapter-down-{suffix}"
    volume = f"openloop-docker-adapter-{suffix}"
    unavailable_volume = f"openloop-docker-adapter-down-{suffix}"
    upstream = Path(os.environ.get("DOCKER_SOCKET", RAW_SOCKET))

    _success("pull", DEFAULT_HAPROXY_RELAY_IMAGE, timeout=300)
    _success("pull", CLIENT_IMAGE, timeout=300)
    _success(
        "run",
        "--rm",
        "--user",
        "0:10777",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--mount",
        (
            f"type=bind,src={CONFIG},"
            "dst=/usr/local/etc/haproxy/haproxy.cfg,readonly"
        ),
        "--entrypoint",
        "haproxy",
        DEFAULT_HAPROXY_RELAY_IMAGE,
        "-c",
        "-f",
        "/usr/local/etc/haproxy/haproxy.cfg",
    )
    try:
        _success("volume", "create", "--label", RESOURCE_LABEL, volume)
        _start_adapter(name=adapter, volume=volume, upstream=upstream)
        _wait_for_health(adapter)

        socket_state = _success(
            "exec",
            adapter,
            "stat",
            "-c",
            "%u:%g:%a:%F",
            DATA_SOCKET,
            HEALTH_SOCKET,
        ).stdout.splitlines()
        assert socket_state == [
            "0:10777:660:socket",
            "0:10777:600:socket",
        ]

        document = _inspect(adapter)
        _assert_adapter_hardening(document)
        mounts = {
            mount["Destination"]: mount for mount in document["Mounts"]
        }
        assert mounts[DATA_PATH]["Type"] == "volume"
        assert mounts[DATA_PATH]["RW"] is True
        assert mounts[RAW_SOCKET]["Type"] == "bind"
        assert mounts[
            "/usr/local/etc/haproxy/haproxy.cfg"
        ]["RW"] is False

        probe = _success(
            "run",
            "--rm",
            "--user",
            "10002:10777",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=volume,src={volume},dst={DATA_PATH},readonly",
            "--env",
            f"DOCKER_HOST=unix://{DATA_SOCKET}",
            CLIENT_IMAGE,
            "version",
            "--format",
            "{{.Server.Version}}",
        )
        assert probe.stdout.strip()

        _success(
            "volume",
            "create",
            "--label",
            RESOURCE_LABEL,
            unavailable_volume,
        )
        _start_adapter(
            name=unavailable,
            volume=unavailable_volume,
            upstream=None,
        )
        _wait_for_health_socket(unavailable)
        time.sleep(5)
        unhealthy = _docker(
            "exec",
            unavailable,
            "sh",
            "-ec",
            HEALTH_COMMAND,
            timeout=5,
        )
        assert unhealthy.returncode != 0
        _assert_adapter_hardening(_inspect(unavailable))
    finally:
        _docker("rm", "--force", adapter)
        _docker("rm", "--force", unavailable)
        _docker("volume", "rm", volume)
        _docker("volume", "rm", unavailable_volume)
