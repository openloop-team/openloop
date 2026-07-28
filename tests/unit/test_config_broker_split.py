"""Runtime and broker processes load disjoint settings schemas."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError, field_validator

from openloop.broker_config import (
    BrokerClientConfig,
    BrokerServiceConfig,
    DEFAULT_EXTERNAL_BROKER_CHECKPOINT_RECEIPT_ROOT,
    DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR,
    DEFAULT_EXTERNAL_BROKER_INGRESS_ROOT,
    DEFAULT_EXTERNAL_BROKER_RUNTIME_ROOT,
    DEFAULT_EXTERNAL_BROKER_STATE_ROOT,
)
from openloop.config import BrokerSettings, RuntimeSettings
from tests.support.settings import (
    IsolatedBrokerSettings,
    IsolatedCoprocessBrokerSettings,
    IsolatedSettings,
)


def test_runtime_loads_runtime_dotenv_without_legacy_fallback(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    (tmp_path / ".env").write_text("OLLAMA_BASE_URL=http://legacy.example\n")
    runtime_file = tmp_path / ".runtime.env"
    runtime_file.write_text("OLLAMA_BASE_URL=http://runtime.example\n")

    assert RuntimeSettings().ollama_base_url == "http://runtime.example"

    runtime_file.unlink()
    assert RuntimeSettings().ollama_base_url == "http://localhost:11434"


def test_runtime_defaults_only_external_client_mount_targets():
    settings = IsolatedSettings()

    assert settings.broker_mode == "coprocess"
    assert Path(settings.broker_control_socket_dir) == (
        DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR
    )
    assert Path(settings.broker_ingress_root) == DEFAULT_EXTERNAL_BROKER_INGRESS_ROOT
    assert Path(settings.broker_checkpoint_receipt_root) == (
        DEFAULT_EXTERNAL_BROKER_CHECKPOINT_RECEIPT_ROOT
    )
    assert settings.broker_identity_private_key is None
    assert settings.broker_shared_data_gid is None
    assert "broker_state_root" not in RuntimeSettings.model_fields
    assert "broker_capability_roots" not in RuntimeSettings.model_fields
    assert "broker_identity_public_keys" not in RuntimeSettings.model_fields
    assert "broker_reconcile_interval_seconds" not in RuntimeSettings.model_fields


def test_broker_defaults_all_fixed_container_targets():
    settings = IsolatedBrokerSettings()

    assert Path(settings.broker_control_socket_dir) == (
        DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR
    )
    assert Path(settings.broker_state_root) == DEFAULT_EXTERNAL_BROKER_STATE_ROOT
    assert Path(settings.broker_runtime_root) == DEFAULT_EXTERNAL_BROKER_RUNTIME_ROOT
    assert Path(settings.broker_ingress_root) == DEFAULT_EXTERNAL_BROKER_INGRESS_ROOT
    assert Path(settings.broker_checkpoint_receipt_root) == (
        DEFAULT_EXTERNAL_BROKER_CHECKPOINT_RECEIPT_ROOT
    )


def test_external_role_configs_preserve_process_owned_settings():
    runtime = IsolatedSettings(
        broker_mode="external",
        broker_control_socket_dir="/custom/control",
        broker_ingress_root="/custom/ingress",
        broker_checkpoint_receipt_root="/custom/receipts",
        broker_identity_private_key=SecretStr("client-seed"),
        broker_shared_data_gid=2000,
    )
    broker = IsolatedBrokerSettings(
        broker_control_socket_dir="/custom/control",
        broker_state_root="/custom/state",
        broker_runtime_root="/custom/runtime",
        broker_ingress_root="/custom/ingress",
        broker_checkpoint_receipt_root="/custom/receipts",
        broker_shared_data_gid=2000,
        broker_expected_app_uid=1000,
    )

    client = BrokerClientConfig.from_runtime_settings(runtime)
    service = BrokerServiceConfig.from_broker_settings(broker)

    for config in (client, service):
        assert config.control_socket_dir == Path("/custom/control")
        assert config.ingress_root == Path("/custom/ingress")
        assert config.checkpoint_receipt_root == Path("/custom/receipts")
    assert service.state_root == Path("/custom/state")
    assert service.runtime_root == Path("/custom/runtime")


def test_role_configs_expose_only_their_process_trust_material():
    runtime = IsolatedSettings(
        broker_mode="external",
        broker_identity_private_key=SecretStr("client-seed"),
        broker_shared_data_gid=2000,
    )
    broker = IsolatedBrokerSettings(
        broker_shared_data_gid=2000,
        broker_expected_app_uid=1000,
    )

    client = BrokerClientConfig.from_runtime_settings(runtime)
    service = BrokerServiceConfig.from_broker_settings(broker)

    assert not hasattr(client, "state_root")
    assert not hasattr(client, "capability_roots")
    assert not hasattr(client, "identity_public_keys")
    assert not hasattr(service, "identity_private_key")
    assert not hasattr(service, "receipt_roots")


def test_coprocess_service_authority_is_explicit_and_separate(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime = IsolatedSettings(
        broker_receipt_roots={"receipt-key-v1": SecretStr("receipt-root")}
    )
    broker = IsolatedCoprocessBrokerSettings(
        broker_control_socket_dir=str(tmp_path / "control"),
        broker_state_root=str(tmp_path / "state"),
        broker_runtime_root=str(runtime_root),
        broker_capability_roots={"cap-key-v1": SecretStr("cap-root")},
        broker_runtime_roots={"runtime-key-v1": SecretStr("runtime-root")},
    )

    client = BrokerClientConfig.from_runtime_settings(
        runtime,
        coprocess_settings=broker,
    )
    service = BrokerServiceConfig.from_coprocess_settings(broker)

    assert client.control_socket_dir == tmp_path / "control"
    assert client.ingress_root == runtime_root / ".workspace-ingress"
    assert client.checkpoint_receipt_root is None
    assert service.mode == "coprocess"
    assert service.state_root == tmp_path / "state"


def test_coprocess_client_requires_coprocess_settings():
    with pytest.raises(ValueError, match="CoprocessBrokerSettings"):
        BrokerClientConfig.from_runtime_settings(IsolatedSettings())


def test_broker_mode_rejects_unknown_value():
    with pytest.raises(ValidationError):
        IsolatedSettings(broker_mode="banana")


def test_settings_validation_errors_hide_all_input_values():
    database_secret = "postgresql://operator:database-secret@db/openloop"
    provider_secret = "provider-secret-value"

    class RuntimeSettingsWithValidation(IsolatedSettings):
        @field_validator("database_url")
        @classmethod
        def reject_database_url(cls, _value):
            raise ValueError("database URL rejected")

    with pytest.raises(ValidationError) as captured:
        RuntimeSettingsWithValidation(
            broker_mode="banana",
            database_url=database_secret,
            gemini_api_key=provider_secret,
        )

    rendered = str(captured.value)
    assert database_secret not in rendered
    assert provider_secret not in rendered
    assert "input_value" not in rendered
    assert "broker_mode" in rendered


@pytest.mark.parametrize("value", [0, -5])
def test_broker_reconcile_interval_seconds_rejects_non_positive(value):
    with pytest.raises(ValidationError):
        IsolatedBrokerSettings(broker_reconcile_interval_seconds=value)


def test_broker_public_keys_round_trip_through_env_json(monkeypatch):
    monkeypatch.setenv(
        "BROKER_IDENTITY_PUBLIC_KEYS", '{"identity-v1":"cHVibGljLWtleQ=="}'
    )
    monkeypatch.setenv(
        "BROKER_RECEIPT_PUBLIC_KEYS", '{"receipt-key-v1":"cmVjZWlwdC1rZXk="}'
    )

    settings = BrokerSettings(_env_file=None)

    assert settings.broker_identity_public_keys == {
        "identity-v1": "cHVibGljLWtleQ=="
    }
    assert settings.broker_receipt_public_keys == {
        "receipt-key-v1": "cmVjZWlwdC1rZXk="
    }


def test_uid_settings_are_owned_by_the_process_that_uses_them(monkeypatch):
    monkeypatch.setenv("BROKER_SHARED_DATA_GID", "2000")
    monkeypatch.setenv("BROKER_EXPECTED_APP_UID", "1000")

    runtime = RuntimeSettings(_env_file=None)
    broker = BrokerSettings(_env_file=None)

    assert runtime.broker_shared_data_gid == 2000
    assert "broker_expected_app_uid" not in RuntimeSettings.model_fields
    assert broker.broker_shared_data_gid == 2000
    assert broker.broker_expected_app_uid == 1000


def test_broker_identity_private_key_is_runtime_only_and_masked(monkeypatch):
    seed = "c" * 43 + "="
    monkeypatch.setenv("BROKER_IDENTITY_PRIVATE_KEY", seed)

    runtime = RuntimeSettings(_env_file=None)
    broker = BrokerSettings(_env_file=None)

    assert runtime.broker_identity_private_key is not None
    assert runtime.broker_identity_private_key.get_secret_value() == seed
    assert seed not in repr(runtime)
    assert "broker_identity_private_key" not in BrokerSettings.model_fields
    assert seed not in repr(broker)
