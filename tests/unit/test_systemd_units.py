"""Static contract for the production systemd Compose units."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
SYSTEMD = ROOT / "ops" / "systemd"
DEPLOY = ROOT / "docker-compose.deploy.yml"
BROKER = ROOT / "docker-compose.broker.yml"
SECRET_ENVIRONMENT_FILE = "-/etc/openloop/openloop-secrets.env"
COMPOSE = (
    "/usr/bin/docker compose --project-name openloop "
    "--file docker-compose.deploy.yml --file docker-compose.broker.yml"
)
PREFLIGHT = (
    "ExecStartPre=/usr/bin/test -r /opt/openloop/docker-compose.deploy.yml",
    "ExecStartPre=/usr/bin/test -r /opt/openloop/docker-compose.broker.yml",
    "ExecStartPre=/usr/bin/test -r /opt/openloop/.env",
    (
        "ExecStartPre=/usr/bin/test -r "
        "/opt/openloop/configs/prd/runtime.env"
    ),
    (
        "ExecStartPre=/usr/bin/test -r "
        "/opt/openloop/configs/prd/broker.env"
    ),
    (
        "ExecStartPre=/bin/bash "
        "/opt/openloop/ops/systemd/validate-secret-environment"
    ),
    f"ExecStartPre={COMPOSE} config --quiet",
)
LONG_RUNNING = (
    "postgres",
    "docker-socket-adapter",
    "broker",
    "runtime",
)
PREDECESSORS = {
    "postgres": None,
    "docker-socket-adapter": "postgres",
    "broker-init": "docker-socket-adapter",
    "broker": "broker-init",
    "runtime": "broker",
}


def _unit(service: str) -> str:
    return (SYSTEMD / f"openloop-{service}.service").read_text()


def _environment_secret_names() -> set[str]:
    names: set[str] = set()
    for path in (DEPLOY, BROKER):
        document = yaml.safe_load(path.read_text())
        names.update(
            definition["environment"]
            for definition in document.get("secrets", {}).values()
            if "environment" in definition
        )
    return names


def test_expected_systemd_artifacts_exist() -> None:
    assert {path.name for path in SYSTEMD.iterdir()} == {
        "README.md",
        "openloop-secrets.env.example",
        "openloop.target",
        "openloop-broker-init.service",
        "openloop-broker.service",
        "openloop-docker-socket-adapter.service",
        "openloop-postgres.service",
        "openloop-runtime.service",
        "validate-secret-environment",
    }


def test_every_service_uses_the_fixed_project_and_validates_configuration() -> None:
    for service in PREDECESSORS:
        unit = _unit(service)

        assert "WorkingDirectory=/opt/openloop" in unit
        assert "RequiresMountsFor=/opt/openloop" in unit
        assert "PartOf=openloop.target" in unit
        assert "Wants=network-online.target" in unit
        assert "Requires=docker.service" in unit
        assert "After=network-online.target docker.service" in unit
        assert all(command in unit for command in PREFLIGHT)
        assert f"EnvironmentFile={SECRET_ENVIRONMENT_FILE}" in unit
        assert ".env.runtime" not in unit
        assert ".env.broker" not in unit


def test_units_serialize_the_compose_services() -> None:
    for service, predecessor in PREDECESSORS.items():
        if predecessor is None:
            continue

        unit = _unit(service)
        assert (
            f"Requires=docker.service openloop-{predecessor}.service" in unit
        )
        assert (
            "After=network-online.target docker.service "
            f"openloop-{predecessor}.service"
        ) in unit


def test_long_running_services_wait_without_building_or_pulling() -> None:
    for service in LONG_RUNNING:
        unit = _unit(service)

        assert "Type=oneshot" in unit
        assert "RemainAfterExit=yes" in unit
        assert (
            f"ExecStart={COMPOSE} up --detach --no-deps --no-build "
            f"--pull never --wait --wait-timeout 120 {service}"
        ) in unit
        assert (
            f"ExecStop={COMPOSE} stop --timeout 30 {service}"
        ) in unit
        assert "TimeoutStartSec=180" in unit
        assert "TimeoutStopSec=60" in unit


def test_broker_init_is_blocking_and_propagates_its_exit_code() -> None:
    unit = _unit("broker-init")

    assert "Type=oneshot" in unit
    assert "RemainAfterExit=yes" in unit
    assert (
        f"ExecStart={COMPOSE} up --no-deps --no-build --pull never "
        "--abort-on-container-exit --exit-code-from broker-init broker-init"
    ) in unit
    assert "ExecStop=" not in unit
    assert "TimeoutStartSec=180" in unit


def test_no_unit_uses_destructive_compose_lifecycle_commands() -> None:
    for service in PREDECESSORS:
        unit = _unit(service)

        assert " compose down" not in unit
        assert " compose rm" not in unit
        assert "--volumes" not in unit
        assert "--remove-orphans" not in unit


def test_target_activates_runtime_and_is_boot_installable() -> None:
    target = (SYSTEMD / "openloop.target").read_text()

    assert "Requires=openloop-runtime.service" in target
    assert "After=openloop-runtime.service" in target
    assert "WantedBy=multi-user.target" in target


def test_secret_template_and_validator_track_compose_environment_sources() -> None:
    expected = _environment_secret_names()
    template = (SYSTEMD / "openloop-secrets.env.example").read_text()
    validator = (SYSTEMD / "validate-secret-environment").read_text()

    documented = set(
        re.findall(r"^([A-Z][A-Z0-9_]*)=", template, flags=re.MULTILINE)
    )
    required_block = validator.split("required=(", 1)[1].split(")", 1)[0]
    validated = set(
        re.findall(r"^\s+([A-Z][A-Z0-9_]*)$", required_block, flags=re.MULTILINE)
    )

    assert documented == expected
    assert validated == expected


def test_secret_validator_rejects_missing_values_without_printing_values() -> None:
    expected = _environment_secret_names()
    environment = {**os.environ, **dict.fromkeys(expected, "test-secret")}
    missing = "POSTGRES_PASSWORD"
    environment.pop(missing)

    result = subprocess.run(
        [
            "/bin/bash",
            str(SYSTEMD / "validate-secret-environment"),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert missing in result.stderr
    assert "test-secret" not in result.stderr


def test_operator_docs_prepare_all_images_and_refresh_units() -> None:
    readme = (SYSTEMD / "README.md").read_text()
    normalized = re.sub(r"\s+", " ", readme)
    update = readme.split("## Deploy an update", 1)[1]

    assert "pull postgres docker-socket-adapter" in normalized
    assert "build runtime" in normalized
    assert "build broker-init broker runtime" not in normalized
    assert "/etc/openloop/openloop-secrets.env" in readme
    assert "systemctl daemon-reload" in update
    assert "/etc/systemd/system/" in update
