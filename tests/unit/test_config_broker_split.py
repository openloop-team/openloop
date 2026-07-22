"""Config surface for the broker-process-split (Phase A).

Phase A only adds `Settings` fields plus `broker_mode` validation; nothing
consumes them yet, so this covers construction only (defaults, rejection of
bad values, and env-var round-trips) — zero behavior change to the rest of
the app. See `.superpowers/sdd/phase-A-brief.md`.
"""

import pytest
from pydantic import ValidationError, field_validator

from openloop.config import Settings


def test_defaults_are_coprocess_with_all_split_fields_unset():
    settings = Settings(_env_file=None)

    assert settings.broker_mode == "coprocess"
    assert settings.broker_identity_private_key is None
    assert settings.broker_identity_key_id == "identity-v1"
    assert settings.broker_identity_public_keys == {}
    assert settings.broker_receipt_public_keys == {}
    assert settings.broker_checkpoint_receipt_root is None
    assert settings.broker_ingress_root is None
    assert settings.broker_shared_data_gid is None
    assert settings.broker_expected_app_uid is None
    assert settings.broker_reconcile_interval_seconds == 300
    assert settings.broker_dev_in_memory is False


def test_broker_mode_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, broker_mode="banana")


def test_settings_validation_errors_hide_all_input_values():
    database_secret = "postgresql://operator:database-secret@db/openloop"
    provider_secret = "provider-secret-value"

    class SettingsWithDatabaseValidation(Settings):
        @field_validator("database_url")
        @classmethod
        def reject_database_url(cls, _value):
            raise ValueError("database URL rejected")

    with pytest.raises(ValidationError) as captured:
        SettingsWithDatabaseValidation(
            _env_file=None,
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
        Settings(_env_file=None, broker_reconcile_interval_seconds=value)


def test_broker_identity_public_keys_round_trips_through_env_json(monkeypatch):
    monkeypatch.setenv(
        "BROKER_IDENTITY_PUBLIC_KEYS", '{"identity-v1": "cHVibGljLWtleQ=="}'
    )

    settings = Settings(_env_file=None)

    assert settings.broker_identity_public_keys == {
        "identity-v1": "cHVibGljLWtleQ=="
    }


def test_broker_receipt_public_keys_round_trips_through_env_json(monkeypatch):
    monkeypatch.setenv(
        "BROKER_RECEIPT_PUBLIC_KEYS", '{"receipt-key-v1": "cmVjZWlwdC1rZXk="}'
    )

    settings = Settings(_env_file=None)

    assert settings.broker_receipt_public_keys == {
        "receipt-key-v1": "cmVjZWlwdC1rZXk="
    }


def test_shared_data_gid_and_expected_app_uid_round_trip_through_env(monkeypatch):
    monkeypatch.setenv("BROKER_SHARED_DATA_GID", "2000")
    monkeypatch.setenv("BROKER_EXPECTED_APP_UID", "1000")

    settings = Settings(_env_file=None)

    assert settings.broker_shared_data_gid == 2000
    assert settings.broker_expected_app_uid == 1000


def test_broker_identity_private_key_is_masked_in_repr(monkeypatch):
    seed = "c" * 43 + "="
    monkeypatch.setenv("BROKER_IDENTITY_PRIVATE_KEY", seed)

    settings = Settings(_env_file=None)

    assert settings.broker_identity_private_key is not None
    assert settings.broker_identity_private_key.get_secret_value() == seed
    assert seed not in repr(settings)
    assert seed not in repr(settings.broker_identity_private_key)
