"""Static contract for the privileged external-broker Compose override."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
OVERRIDE = ROOT / "docker-compose.broker.yml"
RUNTIME_COMPOSITIONS = (
    ROOT / "docker-compose.yml",
    ROOT / "docker-compose.deploy.yml",
)
COMPOSITIONS = (*RUNTIME_COMPOSITIONS, OVERRIDE)
BROKER_ROOT = "${OPENLOOP_BROKER_ROOT:?}"
COMPOSE_DATABASE_URL = (
    "postgresql://${POSTGRES_USER:-openloop}"
    "@postgres:5432/${POSTGRES_DB:-openloop}"
)
COMPOSE_PGPASSWORD = "${POSTGRES_PASSWORD:-change-me}"
BUILD = {
    "context": ".",
    "args": {
        "OPENLOOP_BROKER_UID": "${OPENLOOP_BROKER_UID:-10002}",
        "OPENLOOP_DATA_GID": "${OPENLOOP_DATA_GID:-10777}",
    },
}


def _compose() -> dict:
    return yaml.safe_load(OVERRIDE.read_text())


def _mounts(service: dict) -> dict[str, dict]:
    return {mount["target"]: mount for mount in service["volumes"]}


def test_init_service_runs_as_root_and_provisions_the_same_host_root():
    init = _compose()["services"]["broker-init"]

    assert init["user"] == "0:0"
    assert init["command"] == "python -m openloop.broker_provision"
    assert init["environment"] == {
        "OPENLOOP_BROKER_ROOT": BROKER_ROOT,
        "OPENLOOP_APP_UID": "1000",
        "OPENLOOP_BROKER_UID": "${OPENLOOP_BROKER_UID:-10002}",
        "OPENLOOP_DATA_GID": "${OPENLOOP_DATA_GID:-10777}",
    }
    root_mount = _mounts(init)[BROKER_ROOT]
    assert root_mount["source"] == BROKER_ROOT
    assert root_mount["bind"] == {"create_host_path": True}


def test_every_service_builds_the_same_configurable_numeric_identities():
    services = _compose()["services"]

    for service in ("broker-init", "broker", "runtime"):
        assert services[service]["build"] == BUILD


def test_broker_alone_gets_docker_authority_and_receipts_are_read_only():
    services = _compose()["services"]
    broker = services["broker"]
    runtime = services["runtime"]
    broker_mounts = _mounts(broker)
    runtime_mounts = _mounts(runtime)

    assert broker["user"] == (
        "${OPENLOOP_BROKER_UID:-10002}:${OPENLOOP_DATA_GID:-10777}"
    )
    assert broker["group_add"] == [
        "${DOCKER_GID:?}",
        "${OPENLOOP_DATA_GID:-10777}",
    ]
    assert broker_mounts["/var/run/docker.sock"]["source"] == (
        "${DOCKER_SOCKET:-/var/run/docker.sock}"
    )
    assert "/var/run/docker.sock" not in runtime_mounts
    assert broker_mounts[f"{BROKER_ROOT}/receipts"]["read_only"] is True
    for suffix in ("control", "state", "runtime", "ingress", "receipts"):
        assert broker_mounts[f"{BROKER_ROOT}/{suffix}"]["bind"] == {
            "create_host_path": True
        }
    assert "user" not in runtime
    assert runtime["group_add"] == ["${OPENLOOP_DATA_GID:-10777}"]


def test_broker_has_explicit_external_environment_health_and_ordering():
    services = _compose()["services"]
    broker = services["broker"]
    runtime = services["runtime"]

    assert broker["env_file"] == ".env.broker"
    assert broker["environment"]["DATABASE_URL"] == COMPOSE_DATABASE_URL
    assert broker["environment"]["PGPASSWORD"] == COMPOSE_PGPASSWORD
    assert broker["environment"]["BROKER_MODE"] == "external"
    assert runtime["environment"]["BROKER_MODE"] == "external"
    assert broker["depends_on"] == {
        "broker-init": {"condition": "service_completed_successfully"},
        "postgres": {"condition": "service_healthy"},
    }
    assert runtime["depends_on"]["broker"] == {"condition": "service_healthy"}
    assert broker["healthcheck"]["test"] == [
        "CMD",
        "openloop-broker",
        "--healthcheck",
    ]


def test_runtime_compositions_use_runtime_environment_file():
    for path in RUNTIME_COMPOSITIONS:
        runtime = yaml.safe_load(path.read_text())["services"]["runtime"]
        assert runtime["env_file"] == ".env.runtime"
        assert runtime["environment"]["DATABASE_URL"] == COMPOSE_DATABASE_URL
        assert runtime["environment"]["PGPASSWORD"] == COMPOSE_PGPASSWORD
        assert runtime["environment"]["LOG_LEVEL"] == "${LOG_LEVEL:-info}"


def test_example_files_document_and_preserve_the_secret_partition():
    compose = (ROOT / ".env.example").read_text()
    app = (ROOT / ".env.runtime.example").read_text()
    broker = (ROOT / ".env.broker.example").read_text()

    def assigned_names(document: str) -> set[str]:
        return {
            line.split("=", 1)[0]
            for line in document.splitlines()
            if line and not line.startswith("#") and "=" in line
        }

    def documented_names(document: str) -> set[str]:
        return {
            line.lstrip("#").split("=", 1)[0]
            for line in document.splitlines()
            if re.match(r"^#?[A-Z][A-Z0-9_]*=", line)
        }

    compose_names = assigned_names(compose)
    app_names = assigned_names(app)
    broker_names = assigned_names(broker)
    interpolated_names = {
        match
        for path in COMPOSITIONS
        for match in re.findall(r"\$\{([A-Z][A-Z0-9_]*)", path.read_text())
    }

    assert interpolated_names <= documented_names(compose)
    assert interpolated_names.isdisjoint(documented_names(app))
    assert interpolated_names.isdisjoint(documented_names(broker))

    compose_owned = {
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "LOG_LEVEL",
        "BROKER_IDENTITY_ISSUER",
        "BROKER_IDENTITY_AUDIENCE",
        "BROKER_EXECUTION_LEASE_SECONDS",
        "BROKER_GENERATION_DEADLINE_SECONDS",
        "BROKER_RECONCILE_INTERVAL_SECONDS",
    }
    assert compose_owned <= compose_names
    assert compose_owned.isdisjoint(app_names)
    assert compose_owned.isdisjoint(broker_names)

    assert "OPENAI_API_KEY" not in compose
    assert "BROKER_IDENTITY_PRIVATE_KEY" not in compose
    assert "BROKER_CAPABILITY_ROOTS" not in compose

    assert "BROKER_IDENTITY_PRIVATE_KEY" in app
    assert "BROKER_RECEIPT_ROOTS" in app
    assert "PGPASSWORD" not in app_names
    assert "BROKER_CAPABILITY_ROOTS" not in app_names
    assert "BROKER_RUNTIME_ROOTS" not in app_names
    assert "BROKER_CAPABILITY_ROOTS" in broker
    assert "BROKER_RUNTIME_ROOTS" in broker
    assert "BROKER_IDENTITY_PRIVATE_KEY" not in broker_names
    assert "BROKER_RECEIPT_ROOTS" not in broker_names
    assert "PGPASSWORD" not in broker_names
    assert "DATABASE_URL" in app_names
    assert "DATABASE_URL" not in broker_names
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignored
    assert ".env.runtime" in ignored
    assert ".env.broker" in ignored


def test_operator_commands_use_compose_default_environment_discovery():
    guidance = "\n".join(
        path.read_text()
        for path in (
            ROOT / "README.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "mise.toml",
            *COMPOSITIONS,
        )
    )

    assert "--env-file .env.runtime" not in guidance
