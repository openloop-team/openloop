"""Opt-in canaries for the real external broker process boundary."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import yaml

from tests.integration.test_openhands_cold_resume_live import _fake_openai
from tests.integration.test_openhands_broker_canary_live import (
    run_phase5_checkpoint_park_resume_finalize_real_docker as _run_phase5_canary,
)
from openloop.wiring.broker import _derive_receipt_key


pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.serial,
]

_CANARY_VOLUME_LABEL = "openloop.phase-h-canary"


@pytest.mark.skipif(
    os.environ.get("OPENLOOP_RUN_BROKER_CANARY") != "1",
    reason="set OPENLOOP_RUN_BROKER_CANARY=1 for the external broker canary",
)
def test_external_subprocess_checkpoint_park_resume_finalize_real_docker(
    monkeypatch,
):
    """Run the Phase-5 lifecycle with the broker in a separate real process."""
    monkeypatch.setenv("OPENLOOP_CANARY_BROKER_MODE", "subprocess")
    _run_phase5_canary()


def _root(seed: int) -> str:
    return base64.b64encode(bytes([seed]) * 32).decode()


def _compose_files(project: Path, workspace: Path) -> tuple[Path, Path, Path]:
    base = yaml.safe_load((workspace / "docker-compose.yml").read_text())
    # The canary is isolated by its project name and does not publish services;
    # removing the base development ports avoids collisions with a local stack.
    for service in base["services"].values():
        service.pop("ports", None)
    base_path = project / "docker-compose.yml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))

    broker_path = project / "docker-compose.broker.yml"
    broker_path.write_text((workspace / "docker-compose.broker.yml").read_text())

    canary_path = project / "docker-compose.canary.yml"
    canary_path.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "runtime": {
                        "build": {"context": str(workspace)},
                        "command": [
                            "sh",
                            "-c",
                            "trap 'exit 0' TERM INT; while :; do sleep 3600; done",
                        ],
                        "volumes": [
                            {
                                "type": "bind",
                                "source": str(workspace),
                                "target": "/workspace/openloop",
                                "read_only": True,
                            }
                        ],
                    },
                    "broker": {
                        "build": {"context": str(workspace)},
                        "environment": {
                            # Docker Desktop's VM-local socket is root:root 0755,
                            # so its numeric GID cannot authorize the non-root
                            # broker. This canary-only bridge is the same adapter
                            # used by the established Phase-5 Docker canary.
                            "DOCKER_HOST": "tcp://docker-proxy:2375"
                        },
                        "depends_on": {
                            "docker-proxy": {"condition": "service_started"}
                        },
                    },
                    "broker-init": {"build": {"context": str(workspace)}},
                    "docker-proxy": {
                        "image": "python:3.12-slim",
                        "user": "0:0",
                        "command": [
                            "python",
                            "/workspace/openloop/tests/support/docker_socket_proxy.py",
                        ],
                        "volumes": [
                            {
                                "type": "bind",
                                "source": str(workspace),
                                "target": "/workspace/openloop",
                                "read_only": True,
                            },
                            {
                                "type": "bind",
                                "source": "${DOCKER_SOCKET:-/var/run/docker.sock}",
                                "target": "/var/run/docker.sock",
                            },
                        ],
                    },
                }
            },
            sort_keys=False,
        )
    )
    (project / "agents").mkdir()
    (project / "secrets").mkdir()
    (project / "secrets/github-app.pem").write_text("canary-placeholder\n")
    return base_path, broker_path, canary_path


def _write_partitioned_environments(project: Path) -> None:
    identity_seed = bytes([4]) * 32
    receipt_root = bytes([3]) * 32
    identity_public = (
        Ed25519PrivateKey.from_private_bytes(identity_seed)
        .public_key()
        .public_bytes_raw()
    )
    receipt_public = (
        _derive_receipt_key(receipt_root, "broker-receipt", "receipt-key-v1")
        .public_key()
        .public_bytes_raw()
    )
    database_url = "postgresql://openloop:change-me@postgres:5432/openloop"
    (project / ".env").write_text(
        "\n".join(
            (
                "POSTGRES_USER=openloop",
                "POSTGRES_PASSWORD=change-me",
                "POSTGRES_DB=openloop",
                f"DATABASE_URL={database_url}",
                "STORAGE_MODE=memory",
                "CODING_WORKER_ENABLED=false",
                "BROKER_IDENTITY_KEY_ID=identity-v1",
                f"BROKER_IDENTITY_PRIVATE_KEY={base64.b64encode(identity_seed).decode()}",
                "BROKER_RECEIPT_CURRENT_VERSION=receipt-key-v1",
                "BROKER_RECEIPT_DOMAIN=broker-receipt",
                "BROKER_RECEIPT_ROOTS="
                + json.dumps(
                    {
                        "receipt-key-v1": base64.b64encode(receipt_root).decode()
                    },
                    separators=(",", ":"),
                ),
                "",
            )
        )
    )
    (project / ".env.broker").write_text(
        "\n".join(
            (
                f"DATABASE_URL={database_url}",
                "BROKER_CAPABILITY_ROOTS="
                + json.dumps({"cap-key-v1": _root(1)}, separators=(",", ":")),
                "BROKER_CAPABILITY_CURRENT_VERSION=cap-key-v1",
                "BROKER_RUNTIME_ROOTS="
                + json.dumps(
                    {"runtime-key-v1": _root(2)}, separators=(",", ":")
                ),
                "BROKER_RUNTIME_CURRENT_VERSION=runtime-key-v1",
                "BROKER_IDENTITY_PUBLIC_KEYS="
                + json.dumps(
                    {
                        "identity-v1": base64.b64encode(identity_public).decode()
                    },
                    separators=(",", ":"),
                ),
                "BROKER_RECEIPT_PUBLIC_KEYS="
                + json.dumps(
                    {
                        "receipt-key-v1": base64.b64encode(
                            receipt_public
                        ).decode()
                    },
                    separators=(",", ":"),
                ),
                "",
            )
        )
    )


def _compose(
    project: Path,
    files: tuple[Path, Path, Path],
    project_name: str,
    environment: dict[str, str],
    *arguments: str,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", "--project-directory", str(project)]
    for path in files:
        command.extend(("-f", str(path)))
    command.extend(("--project-name", project_name, *arguments))
    return subprocess.run(
        command,
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, (result.stderr + result.stdout)[-12000:]
    return result.stdout


def _volume_owner(name: str) -> str | None:
    inspected = subprocess.run(
        [
            "docker",
            "volume",
            "inspect",
            "--format",
            f'{{{{ index .Labels "{_CANARY_VOLUME_LABEL}" }}}}',
            name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspected.returncode != 0:
        return None
    return inspected.stdout.strip()


def _create_short_broker_volume() -> tuple[str, str]:
    marker = uuid.uuid4().hex
    for _attempt in range(16):
        # Five characters keeps the generated host relay UDS at <=100 bytes.
        name = f"h{uuid.uuid4().hex[:4]}"
        if _volume_owner(name) is not None:
            continue
        created = subprocess.run(
            [
                "docker",
                "volume",
                "create",
                "--label",
                f"{_CANARY_VOLUME_LABEL}={marker}",
                name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        _assert_success(created)
        if _volume_owner(name) == marker:
            return name, marker
    raise AssertionError("could not allocate a short canary-owned Docker volume")


def _remove_broker_volume(name: str, marker: str) -> None:
    if _volume_owner(name) != marker:
        return
    removed = subprocess.run(
        ["docker", "volume", "rm", name],
        check=False,
        capture_output=True,
        text=True,
    )
    _assert_success(removed)


def _wait_for_compose_broker(
    project: Path,
    files: tuple[Path, Path, Path],
    project_name: str,
    environment: dict[str, str],
) -> None:
    deadline = time.monotonic() + 180
    last = None
    while time.monotonic() < deadline:
        last = _compose(
            project,
            files,
            project_name,
            environment,
            "exec",
            "-T",
            "broker",
            "openloop-broker",
            "--healthcheck",
            timeout=10,
        )
        if last.returncode == 0:
            return
        time.sleep(1)
    assert last is not None
    raise AssertionError((last.stderr + last.stdout)[-12000:])


@pytest.mark.skipif(
    os.environ.get("OPENLOOP_RUN_BROKER_COMPOSE_CANARY") != "1",
    reason=(
        "set OPENLOOP_RUN_BROKER_COMPOSE_CANARY=1 for the distinct-uid "
        "Compose broker canary"
    ),
)
def test_compose_external_broker_distinct_uids_secret_partition_and_real_job():
    workspace = Path(__file__).resolve().parents[2]
    project_name = f"olbc{uuid.uuid4().hex[:10]}"
    # Keep the daemon-host root short enough that the broker's generated relay
    # socket remains within Linux's sockaddr_un budget.
    broker_volume, volume_marker = _create_short_broker_volume()
    # Docker Desktop cannot host AF_UNIX sockets on a macOS bind mount. Its
    # VM-native named-volume path is still an absolute daemon-host path, so it
    # preserves the same-path bind invariant the broker uses for sibling mounts.
    broker_root = Path(f"/var/lib/docker/volumes/{broker_volume}/_data")
    with tempfile.TemporaryDirectory(prefix="openloop-compose-canary-") as temp:
        project = Path(temp)
        files = _compose_files(project, workspace)
        _write_partitioned_environments(project)
        environment = os.environ.copy()
        docker_socket = environment.get("DOCKER_SOCKET", "/var/run/docker.sock")
        docker_gid = os.stat(docker_socket).st_gid if Path(docker_socket).exists() else 0
        environment.update(
            {
                "OPENLOOP_BROKER_ROOT": str(broker_root),
                "OPENLOOP_BROKER_UID": "10002",
                "OPENLOOP_DATA_GID": "10777",
                "DOCKER_GID": str(docker_gid),
                "DOCKER_SOCKET": docker_socket,
            }
        )
        try:
            started = _compose(
                project,
                files,
                project_name,
                environment,
                "up",
                "--build",
                "--detach",
                timeout=600,
            )
            if started.returncode != 0:
                logs = _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "logs",
                    "--no-color",
                    "broker-init",
                    "broker",
                )
                raise AssertionError(
                    (started.stderr + started.stdout + logs.stdout + logs.stderr)[
                        -20000:
                    ]
                )
            _wait_for_compose_broker(project, files, project_name, environment)
            _assert_success(
                _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "exec",
                    "-T",
                    "broker",
                    "docker",
                    "version",
                )
            )

            canary_root = broker_root / "canary"
            for command in (
                ("mkdir", "-p", str(canary_root)),
                ("chown", "1000:10777", str(canary_root)),
                ("chmod", "2750", str(canary_root)),
            ):
                _assert_success(
                    _compose(
                        project,
                        files,
                        project_name,
                        environment,
                        "exec",
                        "-T",
                        "--user",
                        "0",
                        "runtime",
                        *command,
                    )
                )

            with _fake_openai() as fake:
                result = _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "exec",
                    "-T",
                    "-e",
                    "PYTHONPATH=/workspace/openloop/src:/workspace/openloop",
                    "-e",
                    "OPENLOOP_CANARY_BROKER_MODE=managed",
                    "-e",
                    f"OPENLOOP_CANARY_SHARED_ROOT={canary_root}",
                    "-e",
                    f"OPENLOOP_CANARY_MODEL_PORT={fake.server_port}",
                    "runtime",
                    "python",
                    "/workspace/openloop/tests/support/phase5_canary_runner.py",
                    timeout=420,
                )
            if result.returncode != 0:
                logs = _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "logs",
                    "--no-color",
                    "broker",
                )
                audit = _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "exec",
                    "-T",
                    "postgres",
                    "psql",
                    "-U",
                    "openloop",
                    "-d",
                    "openloop",
                    "-Atc",
                    (
                        "SELECT method || ':' || decision || ':' || reason_code "
                        "FROM broker_rpc_audit ORDER BY sequence"
                    ),
                )
                raise AssertionError(
                    (
                        result.stderr
                        + result.stdout
                        + logs.stdout
                        + logs.stderr
                        + audit.stdout
                        + audit.stderr
                    )[-20000:]
                )
            output = result.stdout
            proof = next(
                line.removeprefix("PHASE5_CANARY_OK ")
                for line in output.splitlines()
                if line.startswith("PHASE5_CANARY_OK ")
            )
            payload = json.loads(proof)
            assert payload["topology"] == "managed"
            assert payload["status"] == "terminal"
            assert payload["generations"][-1] == 2
            assert fake.agent_calls == 2

            no_socket = _compose(
                project,
                files,
                project_name,
                environment,
                "exec",
                "-T",
                "runtime",
                "test",
                "!",
                "-e",
                "/var/run/docker.sock",
            )
            _assert_success(no_socket)
            runtime_uid = _assert_success(
                _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "exec",
                    "-T",
                    "runtime",
                    "id",
                    "-u",
                )
            ).strip()
            broker_uid = _assert_success(
                _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "exec",
                    "-T",
                    "broker",
                    "id",
                    "-u",
                )
            ).strip()
            broker_gid = _assert_success(
                _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "exec",
                    "-T",
                    "broker",
                    "id",
                    "-g",
                )
            ).strip()
            assert runtime_uid == "1000"
            assert broker_uid == "10002"
            assert broker_gid == "10777"
            assert runtime_uid != broker_uid

            runtime_env = set(
                _assert_success(
                    _compose(
                        project,
                        files,
                        project_name,
                        environment,
                        "exec",
                        "-T",
                        "runtime",
                        "env",
                    )
                ).splitlines()
            )
            broker_env = set(
                _assert_success(
                    _compose(
                        project,
                        files,
                        project_name,
                        environment,
                        "exec",
                        "-T",
                        "broker",
                        "env",
                    )
                ).splitlines()
            )
            assert not any(
                line.startswith(("BROKER_CAPABILITY_ROOTS=", "BROKER_RUNTIME_ROOTS="))
                for line in runtime_env
            )
            assert not any(
                line.startswith(("BROKER_IDENTITY_PRIVATE_KEY=", "BROKER_RECEIPT_ROOTS="))
                for line in broker_env
            )
        finally:
            _compose(
                project,
                files,
                project_name,
                environment,
                "down",
                "--volumes",
                "--remove-orphans",
                "--timeout",
                "5",
                timeout=120,
            )
            _remove_broker_volume(broker_volume, volume_marker)
