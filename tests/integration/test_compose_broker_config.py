"""Static contract for the privileged external-broker Compose override."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from openloop.tools.openhands_relay_profile import DEFAULT_HAPROXY_RELAY_IMAGE


ROOT = Path(__file__).parents[2]
OVERRIDE = ROOT / "docker-compose.broker.yml"
DEPLOY = ROOT / "docker-compose.deploy.yml"
RUNTIME_COMPOSITIONS = (
    ROOT / "docker-compose.yml",
    DEPLOY,
)
COMPOSITIONS = (*RUNTIME_COMPOSITIONS, OVERRIDE)
BROKER_ROOT = "${OPENLOOP_BROKER_ROOT:?}"
COMPOSE_DATABASE_URL = (
    "postgresql://${POSTGRES_USER:-openloop}"
    "@postgres:5432/${POSTGRES_DB:-openloop}"
)
COMPOSE_PGPASSWORD = "${POSTGRES_PASSWORD:-change-me}"
SECRETS_ROOT = "${OPENLOOP_SECRETS_ROOT:-./secrets}"
OPENLOOP_IMAGE = "${OPENLOOP_IMAGE:-openloop:local}"
BUILD = {
    "context": ".",
    "args": {
        "OPENLOOP_BROKER_UID": "${OPENLOOP_BROKER_UID:-10002}",
        "OPENLOOP_DATA_GID": "${OPENLOOP_DATA_GID:-10777}",
    },
}
ADAPTER_CONFIG = "/usr/local/etc/haproxy/haproxy.cfg"
ADAPTER_DATA = "/run/openloop-docker"
ADAPTER_HEALTH = "/run/openloop-health"
FORWARDED_SOCKET = f"{ADAPTER_DATA}/docker.sock"
RAW_SOCKET = "/var/run/docker.sock"
DOTENV_TARGET = "/app/.env.runtime"


def _compose() -> dict:
    return yaml.safe_load(OVERRIDE.read_text())


def _mounts(service: dict) -> dict[str, dict]:
    return {
        mount["target"]: mount
        for mount in service["volumes"]
        if isinstance(mount, dict)
    }


def _secret_grants(service: dict) -> dict[str, str]:
    grants: dict[str, str] = {}
    for grant in service.get("secrets", ()):
        if isinstance(grant, str):
            grants[grant] = grant
        else:
            grants[grant["target"]] = grant["source"]
    return grants


def _secret_files(document: dict) -> dict[str, str]:
    return {
        name: definition["file"]
        for name, definition in document.get("secrets", {}).items()
    }


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


def test_services_share_one_image_and_only_runtime_builds_it():
    services = _compose()["services"]

    for service in ("broker-init", "broker", "runtime"):
        assert services[service]["image"] == OPENLOOP_IMAGE

    assert services["runtime"]["build"] == BUILD
    assert "build" not in services["broker"]
    assert "build" not in services["broker-init"]

    for path in RUNTIME_COMPOSITIONS:
        runtime = yaml.safe_load(path.read_text())["services"]["runtime"]
        assert runtime["image"] == OPENLOOP_IMAGE


def test_adapter_is_the_only_raw_socket_owner_and_is_stripped_down():
    services = _compose()["services"]
    adapter = services["docker-socket-adapter"]
    broker = services["broker"]
    runtime = services["runtime"]
    adapter_mounts = _mounts(adapter)
    broker_mounts = _mounts(broker)
    runtime_mounts = _mounts(runtime)

    assert adapter["image"] == DEFAULT_HAPROXY_RELAY_IMAGE
    assert adapter["user"] == "0:${OPENLOOP_DATA_GID:-10777}"
    assert adapter["network_mode"] == "none"
    assert "networks" not in adapter
    assert "ports" not in adapter
    assert "env_file" not in adapter
    assert "environment" not in adapter
    assert adapter["read_only"] is True
    assert adapter["cap_drop"] == ["ALL"]
    assert adapter["security_opt"] == ["no-new-privileges:true"]
    assert adapter_mounts[RAW_SOCKET]["source"] == (
        "${DOCKER_SOCKET:-/var/run/docker.sock}"
    )
    assert adapter_mounts[ADAPTER_CONFIG] == {
        "type": "bind",
        "source": "./ops/docker-socket-adapter/haproxy.cfg",
        "target": ADAPTER_CONFIG,
        "read_only": True,
    }
    assert adapter_mounts[ADAPTER_DATA] == {
        "type": "volume",
        "source": "docker-socket-adapter-data",
        "target": ADAPTER_DATA,
    }
    assert adapter["tmpfs"] == [
        f"{ADAPTER_HEALTH}:rw,nosuid,nodev,noexec,size=64k,mode=0700"
    ]
    assert adapter["healthcheck"]["test"] == [
        "CMD-SHELL",
        (
            "printf 'GET /healthz HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n' "
            "| socat -t 3 - "
            "UNIX-CONNECT:/run/openloop-health/health.sock "
            "| grep -q '^HTTP/1\\.[01] 200'"
        ),
    ]

    assert broker["user"] == (
        "${OPENLOOP_BROKER_UID:-10002}:${OPENLOOP_DATA_GID:-10777}"
    )
    assert broker["group_add"] == ["${OPENLOOP_DATA_GID:-10777}"]
    assert RAW_SOCKET not in broker_mounts
    assert broker_mounts[ADAPTER_DATA] == {
        "type": "volume",
        "source": "docker-socket-adapter-data",
        "target": ADAPTER_DATA,
        "read_only": True,
    }
    assert broker["environment"]["DOCKER_HOST"] == f"unix://{FORWARDED_SOCKET}"
    assert RAW_SOCKET not in runtime_mounts
    assert ADAPTER_DATA not in runtime_mounts
    assert "DOCKER_HOST" not in runtime["environment"]
    assert broker_mounts[f"{BROKER_ROOT}/receipts"]["read_only"] is True
    for suffix in ("control", "state", "runtime", "ingress", "receipts"):
        assert broker_mounts[f"{BROKER_ROOT}/{suffix}"]["bind"] == {
            "create_host_path": True
        }
    assert "user" not in runtime
    assert runtime["group_add"] == ["${OPENLOOP_DATA_GID:-10777}"]
    assert _compose()["volumes"] == {"docker-socket-adapter-data": None}


def test_broker_has_explicit_external_environment_health_and_ordering():
    services = _compose()["services"]
    broker = services["broker"]
    runtime = services["runtime"]

    assert "env_file" not in broker
    assert _mounts(broker)[DOTENV_TARGET] == {
        "type": "bind",
        "source": "./.env.broker",
        "target": DOTENV_TARGET,
        "read_only": True,
        "bind": {"create_host_path": False},
    }
    assert broker["environment"]["DATABASE_URL"] == COMPOSE_DATABASE_URL
    assert "PGPASSWORD" not in broker["environment"]
    assert "POSTGRES_PASSWORD" not in broker["environment"]
    assert broker["environment"]["BROKER_MODE"] == "external"
    assert runtime["environment"]["BROKER_MODE"] == "external"
    assert broker["depends_on"] == {
        "docker-socket-adapter": {"condition": "service_healthy"},
        "broker-init": {"condition": "service_completed_successfully"},
        "postgres": {"condition": "service_healthy"},
    }
    assert runtime["depends_on"]["broker"] == {"condition": "service_healthy"}
    assert broker["healthcheck"]["test"] == [
        "CMD",
        "openloop-broker",
        "--healthcheck",
    ]


def test_runtime_compositions_bind_mount_runtime_dotenv():
    for path in RUNTIME_COMPOSITIONS:
        runtime = yaml.safe_load(path.read_text())["services"]["runtime"]
        assert "env_file" not in runtime
        assert _mounts(runtime)[DOTENV_TARGET] == {
            "type": "bind",
            "source": "./.env.runtime",
            "target": DOTENV_TARGET,
            "read_only": True,
            "bind": {"create_host_path": False},
        }
        assert runtime["environment"]["DATABASE_URL"] == COMPOSE_DATABASE_URL
        assert runtime["environment"]["LOG_LEVEL"] == "${LOG_LEVEL:-info}"

    development = yaml.safe_load(RUNTIME_COMPOSITIONS[0].read_text())
    production = yaml.safe_load(DEPLOY.read_text())
    assert (
        development["services"]["runtime"]["environment"]["PGPASSWORD"]
        == COMPOSE_PGPASSWORD
    )
    assert "PGPASSWORD" not in production["services"]["runtime"]["environment"]


def test_production_deploy_mounts_only_the_provided_secret_inventory():
    document = yaml.safe_load(DEPLOY.read_text())
    services = document["services"]
    postgres = services["postgres"]
    runtime = services["runtime"]

    expected_files = {
        "postgres_password": f"{SECRETS_ROOT}/postgres_password",
        "openai_api_key": f"{SECRETS_ROOT}/openai_api_key",
        "anthropic_api_key": f"{SECRETS_ROOT}/anthropic_api_key",
        "groq_api_key": f"{SECRETS_ROOT}/groq_api_key",
        "openrouter_api_key": f"{SECRETS_ROOT}/openrouter_api_key",
        "slack_bot_token": f"{SECRETS_ROOT}/slack_bot_token",
        "slack_app_token": f"{SECRETS_ROOT}/slack_app_token",
        "github_token": f"{SECRETS_ROOT}/github_token",
        "github_app_private_key": f"{SECRETS_ROOT}/github_app_private_key",
        "claude_code_oauth_token": (
            f"{SECRETS_ROOT}/claude_code_oauth_token"
        ),
        "coding_worker_openhands_state_master_key": (
            f"{SECRETS_ROOT}/coding_worker_openhands_state_master_key"
        ),
        "broker_identity_private_key": (
            f"{SECRETS_ROOT}/broker_identity_private_key"
        ),
        "broker_receipt_roots": f"{SECRETS_ROOT}/broker_receipt_roots",
    }
    assert _secret_files(document) == expected_files
    assert _secret_grants(postgres) == {
        "postgres_password": "postgres_password"
    }
    assert postgres["environment"]["POSTGRES_PASSWORD_FILE"] == (
        "/run/secrets/postgres_password"
    )
    assert "POSTGRES_PASSWORD" not in postgres["environment"]
    assert _secret_grants(runtime) == {
        "postgres_password": "postgres_password",
        "openai_api_key": "openai_api_key",
        "anthropic_api_key": "anthropic_api_key",
        "groq_api_key": "groq_api_key",
        "openrouter_api_key": "openrouter_api_key",
        "slack_bot_token": "slack_bot_token",
        "slack_app_token": "slack_app_token",
        "github_token": "github_token",
        "github_app_private_key": "github_app_private_key",
        "claude_code_oauth_token": "claude_code_oauth_token",
        "coding_worker_openhands_state_master_key": (
            "coding_worker_openhands_state_master_key"
        ),
        "broker_identity_private_key": "broker_identity_private_key",
        "broker_receipt_roots": "broker_receipt_roots",
    }
    assert runtime["environment"]["GITHUB_APP_PRIVATE_KEY_PATH"] == (
        "/run/secrets/github_app_private_key"
    )
    raw = DEPLOY.read_text()
    assert "# - gemini_api_key" in raw
    assert "# gemini_api_key:" in raw
    assert "# - slack_signing_secret" in raw
    assert "# slack_signing_secret:" in raw


def test_broker_override_mounts_only_granted_flat_secrets():
    document = _compose()
    services = document["services"]

    assert _secret_files(document) == {
        "postgres_password": f"{SECRETS_ROOT}/postgres_password",
        "broker_identity_private_key": (
            f"{SECRETS_ROOT}/broker_identity_private_key"
        ),
        "broker_receipt_roots": f"{SECRETS_ROOT}/broker_receipt_roots",
        "broker_capability_roots": f"{SECRETS_ROOT}/broker_capability_roots",
        "broker_runtime_roots": f"{SECRETS_ROOT}/broker_runtime_roots",
    }
    assert _secret_grants(services["broker"]) == {
        "postgres_password": "postgres_password",
        "broker_capability_roots": "broker_capability_roots",
        "broker_runtime_roots": "broker_runtime_roots",
    }
    assert _secret_grants(services["runtime"]) == {
        "postgres_password": "postgres_password",
        "broker_identity_private_key": "broker_identity_private_key",
        "broker_receipt_roots": "broker_receipt_roots",
    }
    assert "secrets" not in services["docker-socket-adapter"]
    assert "secrets" not in services["broker-init"]


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
        "OPENLOOP_SECRETS_ROOT",
        "LOG_LEVEL",
        "BROKER_IDENTITY_ISSUER",
        "BROKER_IDENTITY_AUDIENCE",
        "BROKER_EXECUTION_LEASE_SECONDS",
        "BROKER_GENERATION_DEADLINE_SECONDS",
        "BROKER_RECONCILE_INTERVAL_SECONDS",
    }
    # Documented, not necessarily assigned: `.env.example` carries these as
    # commented entries showing the default Compose already applies, and an
    # operator uncomments one only to override it. Ownership is what matters
    # here — the partition below is what keeps them out of the other two files.
    assert compose_owned <= documented_names(compose)
    assert compose_owned.isdisjoint(app_names)
    assert compose_owned.isdisjoint(broker_names)

    assert "OPENAI_API_KEY" not in compose
    assert "BROKER_IDENTITY_PRIVATE_KEY" not in compose
    assert "BROKER_CAPABILITY_ROOTS" not in compose
    assert "DOCKER_GID" not in compose

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
    assert ".env.e2e" in ignored


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


def test_compositions_do_not_receive_secret_manager_tokens():
    for path in COMPOSITIONS:
        assert "DOPPLER_TOKEN" not in path.read_text()
