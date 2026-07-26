"""Where configuration values come from, and which source wins.

`Settings` loads from the environment and `.env.runtime`. One is constructed at
each process entrypoint and passed down; consumers take it as an argument
rather than reaching for a global.

These tests preserve the production precedence contract. The rest of the
unit/in-process suite uses ``IsolatedSettings``, which accepts constructor
arguments and declared defaults but ignores ambient sources entirely.
"""

import pytest

from openloop.config import Settings
from tests.support.settings import IsolatedSettings


@pytest.fixture
def ambient(monkeypatch, tmp_path):
    """A working copy and environment that both carry non-default config."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.runtime").write_text(
        "OLLAMA_BASE_URL=http://from-dotenv:1111\n"
    )
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("STORAGE_MODE", "postgres")
    return tmp_path


def test_reads_the_dotenv_file(ambient):
    assert Settings().ollama_base_url == "http://from-dotenv:1111"


def test_reads_the_process_environment(ambient):
    assert Settings().storage_mode == "postgres"


def test_the_environment_outranks_the_dotenv_file(ambient, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-environ:2222")

    assert Settings().ollama_base_url == "http://from-environ:2222"


def test_an_explicit_argument_outranks_both_ambient_sources(ambient, monkeypatch):
    """What tests rely on: naming a field pins it regardless of the machine."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-environ:2222")

    settings = Settings(ollama_base_url="http://explicit:3333", storage_mode="memory")

    assert settings.ollama_base_url == "http://explicit:3333"
    assert settings.storage_mode == "memory"


def test_env_file_none_still_reads_the_environment(ambient):
    """`_env_file=None` silences the dotenv file only — a common misreading,
    since the environment remains a source."""
    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.storage_mode == "postgres"


def test_isolated_settings_ignore_both_ambient_sources(ambient):
    settings = IsolatedSettings()

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.storage_mode is None


def test_settings_factory_selects_named_profiles(settings_factory):
    memory = settings_factory()
    postgres = settings_factory("postgres", postgres_pool_max_size=3)

    assert memory.storage_mode == "memory"
    assert postgres.storage_mode == "postgres"
    assert postgres.database_url.endswith("/openloop_test")
    assert postgres.postgres_pool_max_size == 3
