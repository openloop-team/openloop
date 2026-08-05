"""Opt-in canaries for the real external broker process boundary."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openloop.wiring.broker import _derive_receipt_key
from tests.integration.test_openhands_broker_canary_live import (
    run_phase5_checkpoint_park_resume_finalize_real_docker as _run_phase5_canary,
)
from tests.support.fake_openai import fake_openai

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
    adapter_config = project / "ops/docker-socket-adapter/haproxy.cfg"
    adapter_config.parent.mkdir(parents=True)
    shutil.copy2(
        workspace / "ops/docker-socket-adapter/haproxy.cfg",
        adapter_config,
    )

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
    secrets = project / "secrets"
    (secrets / "postgres_password").write_text("change-me\n")
    (secrets / "broker_identity_private_key").write_text(
        base64.b64encode(identity_seed).decode() + "\n"
    )
    (secrets / "broker_receipt_roots").write_text(
        json.dumps(
            {"receipt-key-v1": base64.b64encode(receipt_root).decode()},
            separators=(",", ":"),
        )
        + "\n"
    )
    (secrets / "broker_capability_roots").write_text(
        json.dumps({"cap-key-v1": _root(1)}, separators=(",", ":")) + "\n"
    )
    (secrets / "broker_runtime_roots").write_text(
        json.dumps(
            {"runtime-key-v1": _root(2)},
            separators=(",", ":"),
        )
        + "\n"
    )
    (project / ".env").write_text(
        "\n".join(
            (
                "POSTGRES_USER=openloop",
                "POSTGRES_PASSWORD=change-me",
                "POSTGRES_DB=openloop",
                "",
            )
        )
    )
    (project / ".runtime.env").write_text(
        "\n".join(
            (
                "STORAGE_MODE=memory",
                "CODING_WORKER_ENABLED=false",
                "BROKER_IDENTITY_KEY_ID=identity-v1",
                "BROKER_RECEIPT_CURRENT_VERSION=receipt-key-v1",
                "BROKER_RECEIPT_DOMAIN=broker-receipt",
                "",
            )
        )
    )
    (project / ".broker.env").write_text(
        "\n".join(
            (
                "BROKER_CAPABILITY_CURRENT_VERSION=cap-key-v1",
                "BROKER_RUNTIME_CURRENT_VERSION=runtime-key-v1",
                "BROKER_IDENTITY_PUBLIC_KEYS="
                + json.dumps(
                    {"identity-v1": base64.b64encode(identity_public).decode()},
                    separators=(",", ":"),
                ),
                "BROKER_RECEIPT_PUBLIC_KEYS="
                + json.dumps(
                    {"receipt-key-v1": base64.b64encode(receipt_public).decode()},
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
        environment.update(
            {
                "OPENLOOP_BROKER_ROOT": str(broker_root),
                "OPENLOOP_BROKER_UID": "10002",
                "OPENLOOP_DATA_GID": "10777",
                "OPENLOOP_IMAGE": f"{project_name}-openloop:canary",
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
                    "docker-socket-adapter",
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
            socket_state = _assert_success(
                _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "exec",
                    "-T",
                    "docker-socket-adapter",
                    "stat",
                    "-c",
                    "%u:%g:%a:%F",
                    "/run/openloop-docker/docker.sock",
                    "/run/openloop-health/health.sock",
                )
            ).splitlines()
            assert socket_state == [
                "0:10777:660:socket",
                "0:10777:600:socket",
            ]

            adapter_id = _assert_success(
                _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "ps",
                    "-q",
                    "docker-socket-adapter",
                )
            ).strip()
            adapter_document = json.loads(
                _assert_success(
                    subprocess.run(
                        ["docker", "inspect", adapter_id],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                )
            )[0]
            adapter_host = adapter_document["HostConfig"]
            assert adapter_document["Config"]["User"] == "0:10777"
            assert adapter_host["NetworkMode"] == "none"
            assert adapter_host["ReadonlyRootfs"] is True
            assert adapter_host["CapDrop"] == ["ALL"]
            assert adapter_host["CapAdd"] is None
            assert adapter_host["SecurityOpt"] == ["no-new-privileges:true"]
            assert adapter_host["PortBindings"] == {}
            adapter_mounts = {
                mount["Destination"]: mount for mount in adapter_document["Mounts"]
            }
            assert adapter_mounts["/var/run/docker.sock"]["Type"] == "bind"
            assert adapter_mounts["/run/openloop-docker"]["Type"] == "volume"
            assert adapter_mounts["/run/openloop-docker"]["RW"] is True
            broker_id = _assert_success(
                _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "ps",
                    "-q",
                    "broker",
                )
            ).strip()
            runtime_id = _assert_success(
                _compose(
                    project,
                    files,
                    project_name,
                    environment,
                    "ps",
                    "-q",
                    "runtime",
                )
            ).strip()
            broker_document = json.loads(
                _assert_success(
                    subprocess.run(
                        ["docker", "inspect", broker_id],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                )
            )[0]
            runtime_document = json.loads(
                _assert_success(
                    subprocess.run(
                        ["docker", "inspect", runtime_id],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                )
            )[0]
            assert broker_document["Image"] == runtime_document["Image"]
            broker_mounts = {
                mount["Destination"]: mount for mount in broker_document["Mounts"]
            }
            runtime_mounts = {
                mount["Destination"]: mount for mount in runtime_document["Mounts"]
            }
            assert "/var/run/docker.sock" not in broker_mounts
            assert broker_mounts["/run/openloop-docker"]["RW"] is False
            assert "/var/run/docker.sock" not in runtime_mounts
            assert "/run/openloop-docker" not in runtime_mounts

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

            with fake_openai() as fake:
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
                    "docker-socket-adapter",
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
            no_forwarded_socket = _compose(
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
                "/run/openloop-docker/docker.sock",
            )
            _assert_success(no_forwarded_socket)
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
            broker_groups = set(
                _assert_success(
                    _compose(
                        project,
                        files,
                        project_name,
                        environment,
                        "exec",
                        "-T",
                        "broker",
                        "id",
                        "-G",
                    )
                ).split()
            )
            assert runtime_uid == "1000"
            assert broker_uid == "10002"
            assert broker_gid == "10777"
            assert "10777" in broker_groups
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
            expected_database_url = (
                "DATABASE_URL=postgresql://openloop@postgres:5432/openloop"
            )
            assert expected_database_url in runtime_env
            assert expected_database_url in broker_env
            assert "DOCKER_HOST=unix:///run/openloop-docker/docker.sock" in broker_env
            assert not any(line.startswith("DOCKER_HOST=") for line in runtime_env)
            secret_environment_names = (
                "PGPASSWORD=",
                "POSTGRES_PASSWORD=",
                "BROKER_CAPABILITY_ROOTS=",
                "BROKER_RUNTIME_ROOTS=",
                "BROKER_IDENTITY_PRIVATE_KEY=",
                "BROKER_RECEIPT_ROOTS=",
            )
            for container_env in (runtime_env, broker_env):
                assert not any(
                    line.startswith(secret_environment_names) for line in container_env
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
