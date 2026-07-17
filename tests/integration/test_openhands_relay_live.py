"""Opt-in live proof of OpenHands HTTP/WebSocket traffic through HAProxy UDS.

Run with ``OPENHANDS_RELAY_LIVE=1``. Real digest-pinned HAProxy and OpenHands
containers are used, but the agent loop and all model-provider calls stay off.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pytest


os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

if importlib.util.find_spec("openhands.workspace") is None:
    pytest.skip(
        "OpenHands optional dependency is not installed", allow_module_level=True
    )

from openloop.tools.openhands_docker import (
    CONVERSATION_LEASE_TTL_SECONDS,
    DEFAULT_OPENHANDS_SERVER_IMAGE,
    PINNED_OPENHANDS_VERSION,
    native_docker_platform,
    runtime_server_image,
)
from openloop.tools.openhands_relay import (
    CONTAINER_RELAY_CAPABILITY_FILE,
    CONTAINER_RELAY_CONFIG_FILE,
    DEFAULT_HAPROXY_RELAY_IMAGE,
    CompiledOpenHandsRelay,
    RelayClientEndpoint,
    RelayMode,
    compile_openhands_relay,
    install_relay_artifacts,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("OPENHANDS_RELAY_LIVE") != "1",
        reason="set OPENHANDS_RELAY_LIVE=1 for the UDS/HAProxy proof",
    ),
]

_LABEL = "openloop.spike=openhands-relay"
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_PROBE_FILE = _SOURCE_ROOT / "tests" / "support" / "openhands_relay_probe.py"
_PROBE_DOCKERFILE = (
    _SOURCE_ROOT / "tests" / "support" / "Dockerfile.openhands-relay-probe"
)
_HEADER_SINK = r"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return
    def do_GET(self):
        body = json.dumps({
            "relay_header_present": "x-openloop-relay-capability" in self.headers,
            "host": self.headers.get("Host"),
        }, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
""".strip()


def _docker(
    args: list[str],
    *,
    environment: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: float = 120.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *args],
        env=environment,
        input=input_text,
        check=False,
        capture_output=True,
        text=True,
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


@dataclass
class _Resources:
    root: Path
    containers: list[str] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)

    def add_container(self, container_id: str) -> str:
        self.containers.append(container_id)
        return container_id

    def add_network(self, network_id: str) -> str:
        self.networks.append(network_id)
        return network_id

    def add_volume(self, volume_name: str) -> str:
        self.volumes.append(volume_name)
        return volume_name

    def remove_container(self, container_id: str) -> None:
        _docker(["rm", "-f", container_id], check=False, timeout=30)
        if container_id in self.containers:
            self.containers.remove(container_id)

    def remove_network(self, network_id: str) -> None:
        _docker(["network", "rm", network_id], check=False, timeout=30)
        if network_id in self.networks:
            self.networks.remove(network_id)

    def remove_volume(self, volume_name: str) -> None:
        _docker(["volume", "rm", "-f", volume_name], check=False, timeout=30)
        if volume_name in self.volumes:
            self.volumes.remove(volume_name)

    def close(self) -> None:
        for container_id in reversed(self.containers):
            _docker(["rm", "-f", container_id], check=False, timeout=30)
        for network_id in reversed(self.networks):
            _docker(["network", "rm", network_id], check=False, timeout=30)
        for volume_name in reversed(self.volumes):
            _docker(["volume", "rm", "-f", volume_name], check=False, timeout=30)
        shutil.rmtree(self.root, ignore_errors=True)


def _short_root() -> Path:
    parent = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path("/tmp")
    root = Path(tempfile.mkdtemp(prefix="olrelay-", dir=parent))
    root.chmod(0o700)
    return root


def _new_network(resources: _Resources, name: str) -> str:
    result = _docker(["network", "create", "--internal", "--label", _LABEL, name])
    return resources.add_network(result.stdout.strip())


def _controller_probe_image(*, version: str | None = None) -> str:
    fingerprint = hashlib.sha256()
    for path in (
        _PROBE_DOCKERFILE,
        _SOURCE_ROOT / "pyproject.toml",
        _SOURCE_ROOT / "uv.lock",
    ):
        fingerprint.update(path.read_bytes())
    fingerprint.update((version or "locked").encode("ascii"))
    image = f"openloop-openhands-relay-probe:{fingerprint.hexdigest()[:16]}"
    if _docker(["image", "inspect", image], check=False).returncode == 0:
        return image
    build_args = [
        "build",
        "--file",
        str(_PROBE_DOCKERFILE),
        "--label",
        _LABEL,
        "--tag",
        image,
    ]
    if version is not None:
        build_args.extend(("--build-arg", f"OPENHANDS_PROBE_VERSION={version}"))
    build_args.append(str(_SOURCE_ROOT))
    _docker(
        build_args,
        timeout=1800,
    )
    return image


def _new_socket_volume(
    resources: _Resources,
    name: str,
    socket_path: Path,
) -> str:
    result = _docker(["volume", "create", "--label", _LABEL, name])
    volume = resources.add_volume(result.stdout.strip())
    socket_parent = socket_path.parent
    if not socket_parent.is_relative_to(Path("/run/openloop/jobs")):
        raise AssertionError("test relay socket escaped the fixed jobs root")
    _docker(
        [
            "run",
            "--rm",
            "--user",
            "root",
            "--entrypoint",
            "sh",
            "-v",
            f"{volume}:/run/openloop/jobs",
            DEFAULT_HAPROXY_RELAY_IMAGE,
            "-c",
            f"mkdir -p {socket_parent} && "
            f"chown -R {os.getuid()}:{os.getgid()} /run/openloop/jobs && "
            "chmod -R 0700 /run/openloop/jobs",
        ]
    )
    return volume


def _remove_socket(volume: str, socket_path: Path) -> None:
    _docker(
        [
            "run",
            "--rm",
            "--user",
            "root",
            "--entrypoint",
            "rm",
            "-v",
            f"{volume}:/run/openloop/jobs",
            DEFAULT_HAPROXY_RELAY_IMAGE,
            "-f",
            str(socket_path),
        ]
    )


def _compiled_profile(
    job_id: uuid.UUID,
    generation: int,
    conversation_id: uuid.UUID,
    session_key: str,
    *,
    mode: RelayMode,
) -> CompiledOpenHandsRelay:
    return compile_openhands_relay(
        job_id=job_id,
        generation=generation,
        conversation_id=conversation_id,
        relay_capability=secrets.token_urlsafe(32),
        session_api_key=session_key,
        mode=mode,
    )


def _install_profile(config_dir: Path, compiled: CompiledOpenHandsRelay) -> None:
    config_dir.mkdir(mode=0o700)
    descriptor = os.open(config_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        install_relay_artifacts(descriptor, compiled)
    finally:
        os.close(descriptor)


def _validate_config(config_dir: Path) -> None:
    _docker(
        [
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--read-only",
            "-v",
            f"{config_dir}:/run/openloop/config:ro",
            "-v",
            f"{config_dir}:/run/openloop/secrets:ro",
            DEFAULT_HAPROXY_RELAY_IMAGE,
            "haproxy",
            "-c",
            "-f",
            CONTAINER_RELAY_CONFIG_FILE,
        ]
    )


def _start_relay(
    resources: _Resources,
    *,
    config_dir: Path,
    socket_volume: str,
    network: str,
    name: str,
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
            "/tmp:rw,nosuid,nodev,noexec,size=8m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "64m",
            "--pids-limit",
            "64",
            "-v",
            f"{config_dir}:/run/openloop/config:ro",
            "-v",
            f"{config_dir}:/run/openloop/secrets:ro",
            "-v",
            f"{socket_volume}:/run/openloop/jobs:rw",
            DEFAULT_HAPROXY_RELAY_IMAGE,
            "haproxy",
            "-W",
            "-db",
            "-f",
            CONTAINER_RELAY_CONFIG_FILE,
        ]
    )
    return resources.add_container(result.stdout.strip())


def _start_controller_probe(
    resources: _Resources,
    *,
    image: str,
    socket_volume: str,
    name: str,
) -> str:
    pythonpath = "/openloop-src"
    sdk_mount: list[str] = []
    sdk_source_value = os.environ.get("OPENHANDS_RELAY_SDK_SOURCE")
    if sdk_source_value:
        sdk_source = Path(sdk_source_value).resolve(strict=True)
        if not (sdk_source / "openhands" / "sdk" / "__init__.py").is_file():
            raise ValueError(
                "OPENHANDS_RELAY_SDK_SOURCE must contain openhands/sdk/__init__.py"
            )
        pythonpath = f"/openloop-sdk:{pythonpath}"
        sdk_mount = ["-v", f"{sdk_source}:/openloop-sdk:ro"]

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
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "256m",
            "--pids-limit",
            "128",
            "-e",
            "HOME=/tmp",
            "-e",
            f"PYTHONPATH={pythonpath}",
            "-e",
            "OPENHANDS_SUPPRESS_BANNER=1",
            "-v",
            f"{socket_volume}:/run/openloop/jobs:rw",
            *sdk_mount,
            "-v",
            f"{_SOURCE_ROOT / 'src'}:/openloop-src:ro",
            "-v",
            f"{_PROBE_FILE}:/probe.py:ro",
            "--entrypoint",
            "python",
            image,
            "-c",
            "import time; time.sleep(600)",
        ]
    )
    return resources.add_container(result.stdout.strip())


def _probe_payload(endpoint: RelayClientEndpoint, action: str, **values) -> dict:
    return {
        "action": action,
        "socket_path": str(endpoint.socket_path),
        "conversation_id": str(endpoint.conversation_id),
        "relay_capability": endpoint.relay_capability,
        "session_api_key": endpoint.session_api_key,
        "mode": endpoint.mode.value,
        **values,
    }


def _probe(client_id: str, payload: dict, *, timeout: float = 60.0) -> dict:
    result = _docker(
        ["exec", "-i", client_id, "python", "/probe.py"],
        input_text=json.dumps(payload, separators=(",", ":")),
        timeout=timeout,
    )
    output = result.stdout.strip().splitlines()
    if not output:
        raise AssertionError("relay controller probe returned no output")
    return json.loads(output[-1])


def _wait_for_relay(
    client_id: str,
    endpoint: RelayClientEndpoint,
    relay_id: str,
    *,
    timeout: float = 60.0,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    payload = _probe_payload(
        endpoint,
        "http",
        request_relay_capability=endpoint.relay_capability,
        request_session_api_key=endpoint.session_api_key,
        path="/health",
    )
    while time.monotonic() < deadline:
        try:
            if _probe(client_id, payload, timeout=5)["status_code"] == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    state = _docker(["inspect", "--format", "{{json .State}}", relay_id], check=False)
    logs = _docker(["logs", relay_id], check=False)
    detail = (state.stdout + logs.stdout + logs.stderr).strip()
    raise AssertionError(
        f"relay UDS did not become healthy: {last_error}; {detail[:4000]}"
    )


def _wait_for_probe_marker(
    client_id: str,
    marker: Path,
    *,
    timeout: float = 60.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (
            _docker(
                ["exec", client_id, "test", "-f", str(marker)],
                check=False,
                timeout=5,
            ).returncode
            == 0
        ):
            return
        time.sleep(0.1)
    raise AssertionError("relay reconnect probe did not become ready")


def _start_header_sink(
    resources: _Resources,
    *,
    image: str,
    network: str,
    name: str,
) -> str:
    result = _docker(
        [
            "run",
            "-d",
            "--name",
            name,
            "--label",
            _LABEL,
            "--network",
            network,
            "--network-alias",
            "agent",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=8m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--entrypoint",
            "python",
            image,
            "-c",
            _HEADER_SINK,
        ]
    )
    return resources.add_container(result.stdout.strip())


def _start_agent(
    resources: _Resources,
    *,
    image: str,
    platform: str,
    network: str,
    name: str,
    workspace: Path,
    state_dir: Path,
    session_key: str,
    conversation_secret: str,
) -> str:
    environment = dict(os.environ)
    environment.update(
        {
            "OH_SESSION_API_KEYS_0": session_key,
            "OH_SECRET_KEY": conversation_secret,
        }
    )
    args = [
        "run",
        "-d",
        "--platform",
        platform,
        "--name",
        name,
        "--label",
        _LABEL,
        "--network",
        network,
        "--network-alias",
        "agent",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{workspace}:/workspace:rw",
        "-v",
        f"{state_dir}:/openhands-state:rw",
        "-e",
        "OH_SESSION_API_KEYS_0",
        "-e",
        "OH_SECRET_KEY",
        "-e",
        "OH_CONVERSATIONS_PATH=/openhands-state/conversations",
        "-e",
        f"OH_LEASE_TTL_SECONDS={CONVERSATION_LEASE_TTL_SECONDS}",
        "-e",
        "HOME=/tmp",
        "-e",
        "GIT_CONFIG_COUNT=1",
        "-e",
        "GIT_CONFIG_KEY_0=safe.directory",
        "-e",
        "GIT_CONFIG_VALUE_0=/workspace",
    ]
    args.extend((image, "--host", "0.0.0.0", "--port", "8000"))
    result = _docker(args, environment=environment)
    return resources.add_container(result.stdout.strip())


def _assert_no_published_ports(container_id: str) -> None:
    assert _docker(["port", container_id]).stdout.strip() == ""


def _assert_logs_redacted(container_id: str, *credentials: str) -> None:
    result = _docker(["logs", container_id], check=False)
    logs = result.stdout + result.stderr
    for credential in credentials:
        assert credential not in logs


def _git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


@pytest.mark.skipif(not _docker_usable(), reason="no usable docker daemon")
def test_real_haproxy_validates_file_backed_relay_profiles() -> None:
    root = _short_root()
    job_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    session_key = secrets.token_urlsafe(32)
    try:
        _docker(["pull", DEFAULT_HAPROXY_RELAY_IMAGE], timeout=300)
        for mode in RelayMode:
            compiled = _compiled_profile(
                job_id,
                1,
                conversation_id,
                session_key,
                mode=mode,
            )
            config_dir = root / mode.value
            _install_profile(config_dir, compiled)
            _validate_config(config_dir)
            config = (config_dir / "haproxy.cfg").read_text(encoding="utf-8")
            assert compiled.endpoint.relay_capability not in config
            assert session_key not in config
            assert (config_dir / Path(CONTAINER_RELAY_CAPABILITY_FILE).name).read_text(
                encoding="ascii"
            ) == (f"{compiled.endpoint.relay_capability}\n")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.skipif(not _docker_usable(), reason="no usable docker daemon")
def test_real_haproxy_file_capability_is_stripped_before_forwarding() -> None:
    root = _short_root()
    resources = _Resources(root)
    suffix = uuid.uuid4().hex[:10]
    platform = native_docker_platform()
    agent_image = runtime_server_image(DEFAULT_OPENHANDS_SERVER_IMAGE, platform)
    try:
        _docker(["pull", DEFAULT_HAPROXY_RELAY_IMAGE], timeout=300)
        _docker(["pull", "--platform", platform, agent_image], timeout=900)
        compiled = _compiled_profile(
            uuid.uuid4(),
            1,
            uuid.uuid4(),
            secrets.token_urlsafe(32),
            mode=RelayMode.RUNNING,
        )
        endpoint = compiled.endpoint
        config_dir = root / "preflight"
        _install_profile(config_dir, compiled)
        _validate_config(config_dir)
        network = _new_network(resources, f"olrelay-pre-{suffix}")
        socket_volume = _new_socket_volume(
            resources,
            f"olrelay-pre-{suffix}",
            endpoint.socket_path,
        )
        sink_id = _start_header_sink(
            resources,
            image=agent_image,
            network=network,
            name=f"olrelay-sink-{suffix}",
        )
        client_id = _start_controller_probe(
            resources,
            image=_controller_probe_image(),
            socket_volume=socket_volume,
            name=f"olrelay-pre-client-{suffix}",
        )
        relay_id = _start_relay(
            resources,
            config_dir=config_dir,
            socket_volume=socket_volume,
            network=network,
            name=f"olrelay-pre-relay-{suffix}",
        )
        _wait_for_relay(client_id, endpoint, relay_id)

        missing = _probe(
            client_id,
            _probe_payload(
                endpoint,
                "http",
                request_session_api_key=endpoint.session_api_key,
                path="/health",
            ),
        )
        assert missing["status_code"] == 403
        wrong = _probe(
            client_id,
            _probe_payload(
                endpoint,
                "http",
                request_relay_capability="w" * 43,
                request_session_api_key=endpoint.session_api_key,
                path="/health",
            ),
        )
        assert wrong["status_code"] == 403
        observed = _probe(
            client_id,
            _probe_payload(
                endpoint,
                "http",
                request_relay_capability=endpoint.relay_capability,
                request_session_api_key=endpoint.session_api_key,
                path="/health",
                include_json=True,
            ),
        )
        assert observed == {
            "status_code": 200,
            "json": {"relay_header_present": False, "host": "agent:8000"},
        }
        _assert_no_published_ports(sink_id)
        _assert_no_published_ports(client_id)
        _assert_no_published_ports(relay_id)
        _assert_logs_redacted(
            relay_id,
            endpoint.relay_capability,
            endpoint.session_api_key,
        )
    finally:
        resources.close()


@pytest.mark.skipif(not _docker_usable(), reason="no usable docker daemon")
def test_real_openhands_through_uds_relay_and_checkpoint_restart() -> None:
    root = _short_root()
    resources = _Resources(root)
    suffix = uuid.uuid4().hex[:10]
    platform = native_docker_platform()
    agent_image = runtime_server_image(DEFAULT_OPENHANDS_SERVER_IMAGE, platform)
    probe_image = _controller_probe_image(
        version=os.environ.get("OPENHANDS_RELAY_PROBE_VERSION")
    )
    job_id = uuid.uuid4()
    generation = 1
    conversation_id = uuid.uuid4()
    session_key = secrets.token_urlsafe(32)
    conversation_secret = secrets.token_urlsafe(32)

    try:
        _docker(["pull", DEFAULT_HAPROXY_RELAY_IMAGE], timeout=300)
        _docker(["pull", "--platform", platform, agent_image], timeout=900)

        # Real pinned OpenHands conversation setup over the same UDS topology.
        workspace_dir = root / "workspace"
        state_dir = root / "state"
        config_dir = root / "generation"
        workspace_dir.mkdir(mode=0o700)
        state_dir.mkdir(mode=0o700)
        _git(["init", "-q"], cwd=workspace_dir)
        _git(["config", "user.name", "OpenLoop relay spike"], cwd=workspace_dir)
        _git(
            ["config", "user.email", "relay-spike@openloop.invalid"], cwd=workspace_dir
        )
        (workspace_dir / "proof.txt").write_text("base\n", encoding="utf-8")
        _git(["add", "proof.txt"], cwd=workspace_dir)
        _git(["commit", "-qm", "base"], cwd=workspace_dir)
        base_commit = _git(["rev-parse", "HEAD"], cwd=workspace_dir)
        (workspace_dir / "proof.txt").write_text(
            "base\ncheckpoint-over-uds\n", encoding="utf-8"
        )

        network = _new_network(resources, f"olrelay-main-{suffix}")
        running_compiled = _compiled_profile(
            job_id,
            generation,
            conversation_id,
            session_key,
            mode=RelayMode.RUNNING,
        )
        running_endpoint = running_compiled.endpoint
        socket_volume = _new_socket_volume(
            resources,
            f"olrelay-main-{suffix}",
            running_endpoint.socket_path,
        )
        client_id = _start_controller_probe(
            resources,
            image=probe_image,
            socket_volume=socket_volume,
            name=f"olrelay-client-{suffix}",
        )
        expected_versions = {
            distribution: PINNED_OPENHANDS_VERSION
            for distribution in (
                "openhands-agent-server",
                "openhands-sdk",
                "openhands-tools",
                "openhands-workspace",
            )
        }
        assert _probe(
            client_id,
            _probe_payload(running_endpoint, "versions"),
        ) == {"versions": expected_versions}
        assert _probe(
            client_id,
            _probe_payload(running_endpoint, "compatibility"),
        ) == {"compatible": True}
        agent_id = _start_agent(
            resources,
            image=agent_image,
            platform=platform,
            network=network,
            name=f"olrelay-agent-{suffix}",
            workspace=workspace_dir,
            state_dir=state_dir,
            session_key=session_key,
            conversation_secret=conversation_secret,
        )
        _install_profile(config_dir, running_compiled)
        _validate_config(config_dir)
        running_relay_id = _start_relay(
            resources,
            config_dir=config_dir,
            socket_volume=socket_volume,
            network=network,
            name=f"olrelay-running-{suffix}",
        )
        _wait_for_relay(client_id, running_endpoint, running_relay_id)

        def http_status(**values) -> int:
            return _probe(
                client_id,
                _probe_payload(running_endpoint, "http", **values),
            )["status_code"]

        assert http_status(path="/health") == 403
        assert (
            http_status(
                path="/health",
                request_relay_capability="w" * 43,
                request_session_api_key=session_key,
            )
            == 403
        )
        assert (
            http_status(
                path="/api/settings",
                request_relay_capability=running_endpoint.relay_capability,
                request_session_api_key=session_key,
            )
            == 403
        )
        assert (
            http_status(
                path=f"/api/conversations/{conversation_id}",
                request_relay_capability=running_endpoint.relay_capability,
                request_session_api_key="x" * 43,
            )
            == 401
        )

        conversation_result = _probe(
            client_id,
            _probe_payload(running_endpoint, "conversation"),
            timeout=60,
        )
        assert conversation_result == {
            "conversation_id": str(conversation_id),
            "ready": True,
        }
        assert (
            _docker(
                ["inspect", "--format", "{{.State.Running}}", agent_id]
            ).stdout.strip()
            == "true"
        )
        _assert_no_published_ports(agent_id)
        _assert_no_published_ports(running_relay_id)
        _assert_no_published_ports(client_id)
        _assert_logs_redacted(
            running_relay_id,
            running_endpoint.relay_capability,
            session_key,
        )

        # Keep the SDK callback client alive while replacing only HAProxy.
        # A successful replacement state event proves re-upgrade, first-frame
        # authentication, reconciliation, callback delivery, and REST recovery.
        reconnect_ready_path = running_endpoint.socket_path.parent / (
            f".probe-ready-{suffix}"
        )
        reconnect_payload = _probe_payload(
            running_endpoint,
            "conversation_reconnect",
            ready_path=str(reconnect_ready_path),
            timeout=90.0,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            reconnect_result = executor.submit(
                _probe,
                client_id,
                reconnect_payload,
                timeout=120.0,
            )
            _wait_for_probe_marker(client_id, reconnect_ready_path)
            resources.remove_container(running_relay_id)
            _remove_socket(socket_volume, running_endpoint.socket_path)
            running_relay_id = _start_relay(
                resources,
                config_dir=config_dir,
                socket_volume=socket_volume,
                network=network,
                name=f"olrelay-restored-{suffix}",
            )
            _wait_for_relay(client_id, running_endpoint, running_relay_id)
            assert reconnect_result.result(timeout=120) == {
                "conversation_id": str(conversation_id),
                "order": ["reconcile", "event"],
                "rest_status_code": 200,
            }
        assert (
            _docker(
                ["inspect", "--format", "{{.State.Running}}", agent_id]
            ).stdout.strip()
            == "true"
        )

        # Quiescence: replace only HAProxy, rotate capability, reuse volume/path.
        resources.remove_container(running_relay_id)
        _remove_socket(socket_volume, running_endpoint.socket_path)
        checkpoint_compiled = _compiled_profile(
            job_id,
            generation,
            conversation_id,
            session_key,
            mode=RelayMode.CHECKPOINT,
        )
        checkpoint_endpoint = checkpoint_compiled.endpoint
        assert checkpoint_endpoint.socket_path == running_endpoint.socket_path
        assert checkpoint_endpoint.relay_capability != (
            running_endpoint.relay_capability
        )
        checkpoint_config_dir = root / "checkpoint-generation"
        _install_profile(checkpoint_config_dir, checkpoint_compiled)
        _validate_config(checkpoint_config_dir)
        checkpoint_relay_id = _start_relay(
            resources,
            config_dir=checkpoint_config_dir,
            socket_volume=socket_volume,
            network=network,
            name=f"olrelay-checkpoint-{suffix}",
        )
        _wait_for_relay(client_id, checkpoint_endpoint, checkpoint_relay_id)

        stale = _probe(
            client_id,
            _probe_payload(
                checkpoint_endpoint,
                "http",
                path="/health",
                request_relay_capability=running_endpoint.relay_capability,
                request_session_api_key=session_key,
            ),
        )
        assert stale["status_code"] == 403
        mutation = _probe(
            client_id,
            _probe_payload(
                checkpoint_endpoint,
                "http",
                method="POST",
                path="/api/conversations",
                json={},
                request_relay_capability=checkpoint_endpoint.relay_capability,
                request_session_api_key=session_key,
            ),
        )
        assert mutation["status_code"] == 403
        websocket = _probe(
            client_id,
            _probe_payload(checkpoint_endpoint, "websocket_status"),
        )
        assert websocket["status_code"] == 403

        archived = _probe(
            client_id,
            _probe_payload(
                checkpoint_endpoint,
                "archive",
                base_ref=base_commit,
            ),
            timeout=60,
        )
        archive = base64.b64decode(archived["archive_base64"], validate=True)
        assert archived["base_commit"] == base_commit
        assert archived["written"] == len(archive) > 0

        verification = root / "verification"
        _git(["clone", "-q", str(workspace_dir), str(verification)], cwd=root)
        patch_file = root / "workspace.patch"
        patch_file.write_bytes(archive)
        _git(["apply", str(patch_file)], cwd=verification)
        assert (verification / "proof.txt").read_text(encoding="utf-8") == (
            "base\ncheckpoint-over-uds\n"
        )

        assert (
            _docker(
                ["inspect", "--format", "{{.State.Running}}", agent_id]
            ).stdout.strip()
            == "true"
        )
        _assert_no_published_ports(checkpoint_relay_id)
        _assert_logs_redacted(
            checkpoint_relay_id,
            running_endpoint.relay_capability,
            checkpoint_endpoint.relay_capability,
            session_key,
        )
        _assert_logs_redacted(agent_id, session_key, conversation_secret)
    finally:
        resources.close()
