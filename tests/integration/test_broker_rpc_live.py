"""Opt-in hardened Linux container proof for authenticated broker RPC."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import time
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from openloop.broker.models import (
    BrokerOwner,
    IsolationMode,
    JobAuthorizationMetadata,
)
from openloop.broker_rpc.capability import (
    CapabilityRootRing,
    JobCapabilityAuthority,
)
from openloop.broker_rpc.identity import WorkloadIdentityIssuer, WorkloadIntent


pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("OPENLOOP_BROKER_RPC_LIVE") != "1",
        reason="set OPENLOOP_BROKER_RPC_LIVE=1 for the hardened RPC proof",
    ),
]

_LABEL = "openloop.spike=broker-rpc"
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _SOURCE_ROOT / "tests/support/Dockerfile.broker-rpc-probe"
_PROBE = _SOURCE_ROOT / "tests/support/broker_rpc_probe.py"
_POSTGRES_IMAGE = (
    "postgres@sha256:57c72fd2a128e416c7fcc499958864df5301e940bca0a56f58fddf30ffc07777"
)
_SOCKET_PATH = "/run/openloop/control/broker.sock"
_OWNER = BrokerOwner("tenant-a", "workload-a")
_OTHER_OWNER = BrokerOwner("tenant-b", "workload-b")


def _docker(
    args: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"docker {args[0]} failed: {detail[:4000]}")
    return result


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        _docker(["version", "--format", "{{.Server.Version}}"], timeout=10)
    except Exception:
        return False
    return True


def _listed(kind: str) -> list[str]:
    command = {
        "container": ["ps", "-aq", "--filter", f"label={_LABEL}"],
        "network": ["network", "ls", "-q", "--filter", f"label={_LABEL}"],
        "volume": ["volume", "ls", "-q", "--filter", f"label={_LABEL}"],
    }[kind]
    return [line for line in _docker(command).stdout.splitlines() if line]


def _cleanup_labeled() -> None:
    containers = _listed("container")
    if containers:
        _docker(["rm", "-f", *containers], check=False, timeout=60)
    networks = _listed("network")
    for network in networks:
        _docker(["network", "rm", network], check=False, timeout=30)
    volumes = _listed("volume")
    if volumes:
        _docker(["volume", "rm", "-f", *volumes], check=False, timeout=60)


def _probe_image() -> str:
    fingerprint = hashlib.sha256()
    paths = [
        _DOCKERFILE,
        _PROBE,
        _SOURCE_ROOT / "pyproject.toml",
        _SOURCE_ROOT / "uv.lock",
        *sorted((_SOURCE_ROOT / "src/openloop/broker").rglob("*.py")),
        *sorted((_SOURCE_ROOT / "src/openloop/broker/migrations").glob("*.sql")),
        *sorted((_SOURCE_ROOT / "src/openloop/broker_rpc").glob("*.py")),
    ]
    for path in paths:
        fingerprint.update(path.relative_to(_SOURCE_ROOT).as_posix().encode())
        fingerprint.update(path.read_bytes())
    image = f"openloop-broker-rpc-probe:{fingerprint.hexdigest()[:16]}"
    if _docker(["image", "inspect", image], check=False).returncode == 0:
        return image
    _docker(
        [
            "build",
            "--file",
            str(_DOCKERFILE),
            "--label",
            _LABEL,
            "--tag",
            image,
            str(_SOURCE_ROOT),
        ],
        timeout=1800,
    )
    return image


def _write_keys(root: Path):
    private_key = Ed25519PrivateKey.generate()
    public_path = root / "issuer-public.pem"
    root_path = root / "capability-root"
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    capability_root = secrets.token_bytes(32)
    root_text = base64.urlsafe_b64encode(capability_root).rstrip(b"=") + b"\n"
    root_path.write_bytes(root_text)
    public_path.chmod(0o444)
    root_path.chmod(0o400)
    return private_key, capability_root, root_text.decode("ascii").strip()


def _tokens(private_key: Ed25519PrivateKey):
    issuer = WorkloadIdentityIssuer(
        private_key=private_key,
        key_id="issuer-v1",
        issuer="openloop-control",
        audience="openloop:broker-control",
        clock=lambda: datetime.now(UTC),
    )
    worker_ids = {_OWNER: (uuid4(), uuid4()), _OTHER_OWNER: (uuid4(), uuid4())}
    issued: list[str] = []

    def issue(
        intent: WorkloadIntent,
        *,
        owner: BrokerOwner = _OWNER,
        isolation: IsolationMode = IsolationMode.DEDICATED,
        required: IsolationMode = IsolationMode.SHARED,
    ) -> str:
        worker, assignment = worker_ids[owner]
        token = issuer.issue(
            owner=owner,
            worker_instance_id=worker,
            assignment_id=assignment,
            isolation_mode=isolation,
            required_isolation=required,
            intents={intent},
        ).value
        issued.append(token)
        return token

    return issue, issued


def _wait_for_postgres(container: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = _docker(
            ["exec", container, "pg_isready", "-U", "brokercheck"],
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return
        time.sleep(0.2)
    raise AssertionError("PostgreSQL probe did not become ready")


def _wait_for_broker(container: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        logs = _docker(["logs", container], check=False, timeout=5)
        if '{"ready":true}' in logs.stdout:
            return
        state = _docker(
            ["inspect", "--format", "{{.State.Running}}", container],
            check=False,
            timeout=5,
        )
        if state.stdout.strip() == "false":
            raise AssertionError((logs.stdout + logs.stderr)[-4000:])
        time.sleep(0.2)
    raise AssertionError("broker probe did not become ready")


def _socket_inode(image: str, volume: str) -> int:
    result = _docker(
        [
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            "-v",
            f"{volume}:/run/openloop/control:ro",
            image,
            "-c",
            f"import os; print(os.stat('{_SOCKET_PATH}').st_ino)",
        ]
    )
    return int(result.stdout.strip())


def _hold_socket_inode(image: str, volume: str, name: str) -> tuple[str, int]:
    result = _docker(
        [
            "run",
            "-d",
            "--name",
            name,
            "--label",
            _LABEL,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--entrypoint",
            "python",
            "-v",
            f"{volume}:/run/openloop/control:ro",
            image,
            "-c",
            (
                f"import os,time; fd=os.open('{_SOCKET_PATH}',os.O_PATH); "
                "print(os.fstat(fd).st_ino,flush=True); time.sleep(120)"
            ),
        ]
    )
    container = result.stdout.strip()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        output = _docker(["logs", container], check=False).stdout.strip()
        if output.isdigit():
            return container, int(output)
        time.sleep(0.1)
    raise AssertionError("old socket inode holder did not become ready")


def _start_broker(
    *, image: str, network: str, volume: str, config_root: Path, name: str
) -> str:
    result = _docker(
        [
            "run",
            "-d",
            "--name",
            name,
            "--label",
            _LABEL,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--network",
            network,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=16m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "-v",
            f"{config_root}:/run/openloop/config:ro",
            "-v",
            f"{volume}:/run/openloop/control:rw",
            image,
            "broker",
        ]
    )
    container = result.stdout.strip()
    _wait_for_broker(container)
    return container


def _run_client(
    *, image: str, volume: str, name: str, payload: dict[str, object]
):
    result = _docker(
        [
            "run",
            "-i",
            "--name",
            name,
            "--label",
            _LABEL,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=16m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "128m",
            "--cpus",
            "1",
            "--pids-limit",
            "32",
            "-v",
            f"{volume}:/run/openloop/control:rw",
            image,
            "client",
        ],
        input_text=json.dumps(payload, separators=(",", ":")),
        timeout=120,
    )
    output = result.stdout.strip().splitlines()
    if not output:
        raise AssertionError("client probe produced no result")
    return json.loads(output[-1]), result


def _inspect(container: str) -> dict[str, object]:
    return json.loads(_docker(["inspect", container]).stdout)[0]


def _assert_hardened_broker(container: str, network: str) -> None:
    inspected = _inspect(container)
    host = inspected["HostConfig"]
    destinations = {mount["Destination"] for mount in inspected["Mounts"]}
    assert host["ReadonlyRootfs"] is True
    assert host["CapDrop"] == ["ALL"]
    assert "no-new-privileges" in host["SecurityOpt"]
    assert destinations == {"/run/openloop/config", "/run/openloop/control"}
    assert all("docker.sock" not in destination for destination in destinations)
    assert set(inspected["NetworkSettings"]["Networks"]) == {network}
    assert not inspected["NetworkSettings"]["Ports"]
    tcp_listeners = _docker(
        [
            "exec",
            container,
            "python",
            "-c",
            (
                "import os; from pathlib import Path; "
                "owned={os.readlink(p)[8:-1] for p in Path('/proc/1/fd').iterdir() "
                "if os.readlink(p).startswith('socket:[')}; "
                "print(sum(line.split()[3]=='0A' and line.split()[9] in owned "
                "for name in ('/proc/net/tcp','/proc/net/tcp6') for line in "
                "Path(name).read_text().splitlines()[1:]))"
            ),
        ]
    )
    assert tcp_listeners.stdout.strip() == "0"


def _assert_networkless_client(container: str) -> None:
    inspected = _inspect(container)
    assert inspected["HostConfig"]["NetworkMode"] == "none"
    assert inspected["HostConfig"]["ReadonlyRootfs"] is True
    assert inspected["HostConfig"]["CapDrop"] == ["ALL"]
    assert {mount["Destination"] for mount in inspected["Mounts"]} == {
        "/run/openloop/control"
    }
    assert not inspected["NetworkSettings"]["Ports"]


@pytest.mark.skipif(not _docker_usable(), reason="no usable Docker daemon")
def test_hardened_linux_broker_rpc_has_no_docker_or_tcp_authority() -> None:
    _cleanup_labeled()
    parent = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path("/tmp")
    root = Path(tempfile.mkdtemp(prefix="olbroker-rpc-", dir=parent))
    root.chmod(0o700)
    suffix = uuid4().hex[:10]
    network = f"olbroker-rpc-{suffix}"
    volume = f"olbroker-rpc-{suffix}"
    postgres = f"olbroker-rpc-postgres-{suffix}"
    broker_one = f"olbroker-rpc-broker-a-{suffix}"
    broker_two = f"olbroker-rpc-broker-b-{suffix}"
    client_one = f"olbroker-rpc-client-a-{suffix}"
    client_two = f"olbroker-rpc-client-b-{suffix}"
    inode_holder = f"olbroker-rpc-inode-{suffix}"
    all_output = ""
    tokens: list[str] = []
    try:
        image = _probe_image()
        private_key, capability_root, root_text = _write_keys(root)
        config = {
            "postgres_dsn": "postgresql://brokercheck:brokercheck@postgres/brokercheck",
            "issuer": "openloop-control",
            "audience": "openloop:broker-control",
            "issuer_key_id": "issuer-v1",
            "issuer_public_key_path": "/run/openloop/config/issuer-public.pem",
            "capability_key_version": "cap-v1",
            "capability_root_path": "/run/openloop/config/capability-root",
            "socket_path": _SOCKET_PATH,
        }
        config_path = root / "broker.json"
        config_path.write_text(json.dumps(config, separators=(",", ":")))
        config_path.chmod(0o400)

        _docker(["network", "create", "--internal", "--label", _LABEL, network])
        _docker(["volume", "create", "--label", _LABEL, volume])
        _docker(
            [
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "root",
                "--entrypoint",
                "sh",
                "-v",
                f"{volume}:/run/openloop/control",
                image,
                "-c",
                f"chown {os.getuid()}:{os.getgid()} /run/openloop/control && "
                "chmod 0770 /run/openloop/control",
            ]
        )
        _docker(
            [
                "run",
                "-d",
                "--name",
                postgres,
                "--label",
                _LABEL,
                "--network",
                network,
                "--network-alias",
                "postgres",
                "--tmpfs",
                "/var/lib/postgresql/data:rw,nosuid,nodev,size=256m",
                "-e",
                "POSTGRES_USER=brokercheck",
                "-e",
                "POSTGRES_PASSWORD=brokercheck",
                "-e",
                "POSTGRES_DB=brokercheck",
                _POSTGRES_IMAGE,
            ],
            timeout=300,
        )
        _wait_for_postgres(postgres)

        broker_id = _start_broker(
            image=image,
            network=network,
            volume=volume,
            config_root=root,
            name=broker_one,
        )
        _assert_hardened_broker(broker_id, network)
        _, first_inode = _hold_socket_inode(image, volume, inode_holder)

        issue, issued = _tokens(private_key)
        initial_payload = {
            "phase": "initial",
            "socket_path": _SOCKET_PATH,
            "idempotency_key": "live-broker-create-0001",
            "dedicated_idempotency_key": "live-broker-dedicated-01",
            "tokens": {
                "create": issue(WorkloadIntent.CREATE_JOB),
                "replay": issue(WorkloadIntent.CREATE_JOB),
                "inspect": issue(WorkloadIntent.INSPECT_JOB),
                "cross_tenant": issue(
                    WorkloadIntent.INSPECT_JOB, owner=_OTHER_OWNER
                ),
                "wrong_capability": issue(WorkloadIntent.INSPECT_JOB),
                "create_dedicated": issue(
                    WorkloadIntent.CREATE_JOB,
                    required=IsolationMode.DEDICATED,
                ),
                "downgrade": issue(
                    WorkloadIntent.INSPECT_JOB,
                    isolation=IsolationMode.SHARED,
                    required=IsolationMode.SHARED,
                ),
            },
        }
        tokens.extend(issued)
        initial, first_process = _run_client(
            image=image,
            volume=volume,
            name=client_one,
            payload=initial_payload,
        )
        all_output += first_process.stdout + first_process.stderr
        _assert_networkless_client(client_one)
        assert initial == {
            "capability_fingerprint": initial["capability_fingerprint"],
            "cross_tenant": "NOT_FOUND_OR_UNAUTHORIZED",
            "dedicated_job_id": initial["dedicated_job_id"],
            "downgrade": "NOT_FOUND_OR_UNAUTHORIZED",
            "inspect": True,
            "job_id": initial["job_id"],
            "replay": True,
            "same_capability": True,
            "same_job": True,
            "wrong_capability": "NOT_FOUND_OR_UNAUTHORIZED",
        }

        first_broker_logs = _docker(["logs", broker_one], check=False).stdout
        _docker(["stop", "--time", "10", broker_one], timeout=30)
        _docker(["rm", broker_one])
        broker_id = _start_broker(
            image=image,
            network=network,
            volume=volume,
            config_root=root,
            name=broker_two,
        )
        _assert_hardened_broker(broker_id, network)
        assert _socket_inode(image, volume) != first_inode

        issue_after, issued_after = _tokens(private_key)
        restart_payload = {
            "phase": "restart",
            "socket_path": _SOCKET_PATH,
            "idempotency_key": "live-broker-create-0001",
            "tokens": {
                "replay": issue_after(WorkloadIntent.CREATE_JOB),
                "inspect": issue_after(WorkloadIntent.INSPECT_JOB),
            },
        }
        tokens.extend(issued_after)
        restarted, second_process = _run_client(
            image=image,
            volume=volume,
            name=client_two,
            payload=restart_payload,
        )
        all_output += second_process.stdout + second_process.stderr
        _assert_networkless_client(client_two)
        assert restarted == {
            "capability_fingerprint": initial["capability_fingerprint"],
            "inspect": True,
            "job_id": initial["job_id"],
            "replay": True,
        }

        counts = _docker(
            [
                "exec",
                postgres,
                "psql",
                "-U",
                "brokercheck",
                "-d",
                "brokercheck",
                "-Atc",
                "SELECT (SELECT count(*) FROM broker_audit), "
                "(SELECT count(*) FROM broker_rpc_audit), "
                "(SELECT min(peer_pid) FROM broker_rpc_audit), "
                "(SELECT max(peer_pid) FROM broker_rpc_audit)",
            ]
        ).stdout.strip()
        assert counts == "2|9|0|0"
        rows = _docker(
            [
                "exec",
                postgres,
                "psql",
                "-U",
                "brokercheck",
                "-d",
                "brokercheck",
                "-Atc",
                "SELECT row_to_json(j)::text FROM broker_jobs j ORDER BY job_id",
            ]
        ).stdout
        audit_rows = _docker(
            [
                "exec",
                postgres,
                "psql",
                "-U",
                "brokercheck",
                "-d",
                "brokercheck",
                "-Atc",
                "SELECT row_to_json(a)::text FROM broker_rpc_audit a "
                "ORDER BY sequence",
            ]
        ).stdout
        logs = first_broker_logs + _docker(
            ["logs", broker_two], check=False
        ).stdout
        postgres_logs = _docker(["logs", postgres], check=False).stdout
        database_digest = JobCapabilityAuthority(
            CapabilityRootRing({"cap-v1": capability_root}, current_version="cap-v1")
        ).digest_for(_OWNER, UUID(initial["job_id"]), "cap-v1", 1)
        authority = JobCapabilityAuthority(
            CapabilityRootRing({"cap-v1": capability_root}, current_version="cap-v1")
        )
        capabilities = (
            authority.derive(
                _OWNER,
                UUID(initial["job_id"]),
                JobAuthorizationMetadata("cap-v1", 1, database_digest),
            ).value,
            authority.derive(
                _OWNER,
                UUID(initial["dedicated_job_id"]),
                JobAuthorizationMetadata(
                    "cap-v1",
                    1,
                    authority.digest_for(
                        _OWNER,
                        UUID(initial["dedicated_job_id"]),
                        "cap-v1",
                        1,
                    ),
                ),
            ).value,
        )
        combined = all_output + logs + postgres_logs + rows + audit_rows
        for secret in (*tokens, root_text, *capabilities):
            assert secret not in combined
        assert database_digest not in all_output + logs
        assert database_digest not in audit_rows
        assert all(capability not in rows for capability in capabilities)
    finally:
        _cleanup_labeled()
        shutil.rmtree(root, ignore_errors=True)
        assert _listed("container") == []
        assert _listed("network") == []
        assert _listed("volume") == []
