"""Secret-safe diagnostics and exit semantics for the broker CLI shell.

Each case composes the loader it needs into ``main`` rather than swapping the
module's ``BrokerSettings`` out from under it — the failure paths are the point, so
they are supplied directly.
"""

import logging
import os

import pytest
from pydantic import BaseModel, field_validator

import openloop.broker_main as broker_main
from tests.support.settings import IsolatedBrokerSettings as Settings


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

    def load_invalid_settings() -> Settings:
        return Settings(
            broker_reconcile_interval_seconds=0,
            database_url=database_secret,
        )

    with caplog.at_level(logging.ERROR, logger="openloop.broker"):
        code = broker_main.main([], load_settings=load_invalid_settings)

    assert code == 1
    assert 'loc=["broker_reconcile_interval_seconds"]' in caplog.text
    assert 'type="greater_than"' in caplog.text
    assert database_secret not in caplog.text
    assert "greater than 0" not in caplog.text
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


def test_healthcheck_does_not_construct_full_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("BROKER_CONTROL_SOCKET_DIR", str(tmp_path))

    def fail_if_loaded() -> Settings:
        raise AssertionError("healthcheck consumed mounted settings")

    assert broker_main.main(["--healthcheck"], load_settings=fail_if_loaded) == 1


def test_healthcheck_uses_default_control_path_without_full_settings(monkeypatch):
    monkeypatch.delenv("BROKER_CONTROL_SOCKET_DIR", raising=False)
    observed = []
    monkeypatch.setattr(
        broker_main,
        "_healthcheck_socket",
        lambda path: observed.append(path) or 0,
    )

    def fail_if_loaded() -> Settings:
        raise AssertionError("healthcheck consumed mounted settings")

    assert broker_main.main(["--healthcheck"], load_settings=fail_if_loaded) == 0
    assert observed == [
        os.fspath(broker_main.DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR)
    ]


def test_main_keyboard_interrupt_before_serving_returns_130(monkeypatch):
    def interrupt(awaitable):
        awaitable.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(broker_main.asyncio, "run", interrupt)

    assert (
        broker_main.main(
            [],
            load_settings=lambda: Settings(
                broker_state_root="/host/state",
                broker_runtime_root="/host/runtime",
                broker_shared_data_gid=2000,
                broker_expected_app_uid=1000,
            ),
        )
        == 130
    )
