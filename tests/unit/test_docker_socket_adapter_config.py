"""Static contract for the fixed HAProxy Docker-socket adapter."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "ops/docker-socket-adapter/haproxy.cfg"


def _config() -> str:
    return CONFIG.read_text()


def _section(document: str, kind: str, name: str) -> str:
    header = kind if not name else f"{kind} {name}"
    match = re.search(
        rf"(?m)^{re.escape(header)}\n(?P<body>(?:^[ \t].*\n?)*)",
        document,
    )
    assert match is not None
    return match.group("body")


def test_adapter_configuration_has_no_rootfs_runtime_files() -> None:
    document = _config()

    assert "log stdout format raw local0" in document
    assert re.search(r"(?m)^  maxconn 32$", document)
    assert re.search(r"(?m)^  hard-stop-after 10s$", document)
    assert not re.search(r"(?im)^\s*(pidfile|chroot)\b", document)


def test_adapter_configuration_has_bounded_transport_timeouts() -> None:
    document = _config()
    defaults = _section(document, "defaults", "")

    assert re.search(r"(?m)^  mode tcp$", defaults)
    assert re.search(r"(?m)^  timeout connect 3s$", defaults)
    assert re.search(r"(?m)^  timeout client 1h$", defaults)
    assert re.search(r"(?m)^  timeout server 1h$", defaults)


def test_data_frontend_is_group_writable_unix_only_tcp() -> None:
    document = _config()
    frontend = _section(document, "frontend", "docker_socket")

    assert re.search(r"(?m)^  mode tcp$", frontend)
    assert re.findall(r"(?m)^  bind (.+)$", frontend) == [
        "unix@/run/openloop-docker/docker.sock mode 0660"
    ]
    assert re.search(r"(?m)^  default_backend docker_daemon$", frontend)


def test_health_frontend_is_root_only_and_reflects_backend_verdict() -> None:
    document = _config()
    frontend = _section(document, "frontend", "adapter_health")

    assert re.search(r"(?m)^  mode http$", frontend)
    assert re.findall(r"(?m)^  bind (.+)$", frontend) == [
        "unix@/run/openloop-health/health.sock mode 0600"
    ]
    assert re.search(r"(?m)^  monitor-uri /healthz$", frontend)
    assert re.search(
        r"(?m)^  monitor fail if \{ nbsrv\(docker_daemon\) eq 0 \}$",
        frontend,
    )


def test_backend_checks_docker_ping_over_the_raw_unix_socket() -> None:
    document = _config()
    backend = _section(document, "backend", "docker_daemon")

    assert re.search(r"(?m)^  mode tcp$", backend)
    assert re.search(r"(?m)^  option httpchk$", backend)
    assert re.search(
        r"(?m)^  http-check send meth GET uri /_ping ver HTTP/1\.1 "
        r"hdr Host docker$",
        backend,
    )
    assert re.search(r"(?m)^  http-check expect status 200$", backend)
    assert re.findall(r"(?m)^  server (.+)$", backend) == [
        "daemon unix@/var/run/docker.sock check inter 2s fall 2 rise 1"
    ]


def test_every_listener_and_backend_endpoint_is_unix_only() -> None:
    endpoints = re.findall(r"(?m)^  (?:bind|server \S+) (\S+)", _config())

    assert endpoints
    assert all(endpoint.startswith("unix@/") for endpoint in endpoints)
