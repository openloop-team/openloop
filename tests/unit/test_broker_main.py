"""Secret-safe diagnostics and exit semantics for the broker CLI shell."""

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, field_validator

import openloop.broker_main as broker_main


@pytest.fixture(autouse=True)
def _isolate_logging_configuration(monkeypatch):
    """Keep in-process CLI tests from mutating the root logger."""
    monkeypatch.setattr(
        broker_main.logging,
        "basicConfig",
        lambda **_kwargs: None,
    )


def test_main_logs_only_validation_location_and_type(monkeypatch, caplog):
    database_secret = "postgresql://operator:database-secret@db/openloop"
    provider_secret = "provider-secret-value"
    monkeypatch.setenv("BROKER_MODE", "banana")
    monkeypatch.setenv("DATABASE_URL", database_secret)
    monkeypatch.setenv("GEMINI_API_KEY", provider_secret)

    with caplog.at_level(logging.ERROR, logger="openloop.broker"):
        code = broker_main.main([])

    assert code == 1
    assert 'loc=["broker_mode"]' in caplog.text
    assert 'type="value_error"' in caplog.text
    assert database_secret not in caplog.text
    assert provider_secret not in caplog.text
    assert "broker_mode must be" not in caplog.text
    assert "Traceback" not in caplog.text


def test_main_logs_only_unexpected_settings_exception_class(monkeypatch, caplog):
    unexpected_secret = "unexpected-secret-value"

    def fail_settings():
        raise RuntimeError(unexpected_secret)

    monkeypatch.setattr(broker_main, "Settings", fail_settings)

    with caplog.at_level(logging.ERROR, logger="openloop.broker"):
        code = broker_main.main([])

    assert code == 1
    assert "error_type=RuntimeError" in caplog.text
    assert unexpected_secret not in caplog.text
    assert "Traceback" not in caplog.text


def test_main_omits_validator_controlled_message_text(monkeypatch, caplog):
    secret = "validator-message-secret"

    class InvalidSettings(BaseModel):
        database_url: str

        @field_validator("database_url")
        @classmethod
        def reject_database_url(cls, value):
            raise ValueError(f"rejected database URL {value}")

    monkeypatch.setattr(
        broker_main,
        "Settings",
        lambda: InvalidSettings(database_url=secret),
    )

    with caplog.at_level(logging.ERROR, logger="openloop.broker"):
        code = broker_main.main([])

    assert code == 1
    assert 'loc=["database_url"]' in caplog.text
    assert 'type="value_error"' in caplog.text
    assert secret not in caplog.text
    assert "rejected database URL" not in caplog.text


def test_main_keyboard_interrupt_before_serving_returns_130(monkeypatch):
    monkeypatch.setattr(
        broker_main,
        "Settings",
        lambda: SimpleNamespace(log_level="info"),
    )

    def interrupt(awaitable):
        awaitable.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(broker_main.asyncio, "run", interrupt)

    assert broker_main.main([]) == 130
