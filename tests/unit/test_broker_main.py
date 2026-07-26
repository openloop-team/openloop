"""Secret-safe diagnostics and exit semantics for the broker CLI shell.

Each case composes the loader it needs into ``main`` rather than swapping the
module's ``Settings`` out from under it — the failure paths are the point, so
they are supplied directly.
"""

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, field_validator

import openloop.broker_main as broker_main
from tests.support.settings import IsolatedSettings as Settings


@pytest.fixture(autouse=True)
def _isolate_logging_configuration(monkeypatch):
    """Keep in-process CLI tests from mutating the root logger."""
    monkeypatch.setattr(
        broker_main.logging,
        "basicConfig",
        lambda **_kwargs: None,
    )


def test_main_logs_only_validation_location_and_type(caplog):
    database_secret = "postgresql://operator:database-secret@db/openloop"
    provider_secret = "provider-secret-value"

    def load_invalid_mode() -> Settings:
        return Settings(
            broker_mode="banana",
            database_url=database_secret,
            gemini_api_key=provider_secret,
        )

    with caplog.at_level(logging.ERROR, logger="openloop.broker"):
        code = broker_main.main([], load_settings=load_invalid_mode)

    assert code == 1
    assert 'loc=["broker_mode"]' in caplog.text
    assert 'type="value_error"' in caplog.text
    assert database_secret not in caplog.text
    assert provider_secret not in caplog.text
    assert "broker_mode must be" not in caplog.text
    assert "Traceback" not in caplog.text


def test_main_logs_only_unexpected_settings_exception_class(caplog):
    unexpected_secret = "unexpected-secret-value"

    def load_raising() -> Settings:
        raise RuntimeError(unexpected_secret)

    with caplog.at_level(logging.ERROR, logger="openloop.broker"):
        code = broker_main.main([], load_settings=load_raising)

    assert code == 1
    assert "error_type=RuntimeError" in caplog.text
    assert unexpected_secret not in caplog.text
    assert "Traceback" not in caplog.text


def test_main_omits_validator_controlled_message_text(caplog):
    secret = "validator-message-secret"

    class InvalidSettings(BaseModel):
        database_url: str

        @field_validator("database_url")
        @classmethod
        def reject_database_url(cls, value):
            raise ValueError(f"rejected database URL {value}")

    def load_validator_failure():
        return InvalidSettings(database_url=secret)

    with caplog.at_level(logging.ERROR, logger="openloop.broker"):
        code = broker_main.main([], load_settings=load_validator_failure)

    assert code == 1
    assert 'loc=["database_url"]' in caplog.text
    assert 'type="value_error"' in caplog.text
    assert secret not in caplog.text
    assert "rejected database URL" not in caplog.text


def test_main_keyboard_interrupt_before_serving_returns_130(monkeypatch):
    def interrupt(awaitable):
        awaitable.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(broker_main.asyncio, "run", interrupt)

    assert (
        broker_main.main(
            [], load_settings=lambda: SimpleNamespace(log_level="info")
        )
        == 130
    )
