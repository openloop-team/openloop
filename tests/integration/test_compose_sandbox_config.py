"""Static Docker-authority contract for Compose deployment files."""

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
COMPOSE_FILES = tuple(sorted(ROOT.glob("docker-compose*.yml")))
SOCKET = "/var/run/docker.sock"


def _compose(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _mounts(service: dict) -> list:
    return service.get("volumes", []) or []


def _targets(service: dict) -> set[str]:
    targets = set()
    for mount in _mounts(service):
        if isinstance(mount, dict):
            targets.add(str(mount.get("target", "")))
        elif isinstance(mount, str):
            targets.add(mount.split(":", 1)[-1])
    return targets


def test_legacy_runtime_socket_override_is_absent() -> None:
    assert not (ROOT / "docker-compose.sandbox.yml").exists()


def test_only_broker_service_receives_docker_socket() -> None:
    socket_owners = []
    for path in COMPOSE_FILES:
        for service_name, service in (_compose(path).get("services") or {}).items():
            if SOCKET in _targets(service):
                socket_owners.append((path.name, service_name))

    assert socket_owners == [("docker-compose.broker.yml", "broker")]


def test_runtime_services_never_receive_docker_socket() -> None:
    for path in COMPOSE_FILES:
        runtime = (_compose(path).get("services") or {}).get("runtime", {})
        assert SOCKET not in _targets(runtime), path.name
