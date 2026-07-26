"""Hermetic settings helpers for unit and in-process integration tests.

``IsolatedSettings`` keeps the production model and validators, but accepts
values only from explicit constructor arguments. It never reads the developer's
environment, ``.env.runtime``, or file-secret directories.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from openloop.config import Settings


TEST_AGENTS_DIR = Path(__file__).parents[1] / "integration" / "data"

_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "memory": {
        "agents_dir": str(TEST_AGENTS_DIR),
        "storage_mode": "memory",
    },
    "postgres": {
        "agents_dir": str(TEST_AGENTS_DIR),
        "storage_mode": "postgres",
        "database_url": "postgresql://test:test@localhost:5432/openloop_test",
    },
}


class IsolatedSettings(Settings):
    """A ``Settings`` variant whose only external source is ``__init__``."""

    # Test composition always uses fixture agents, never whichever files happen
    # to be present in the working copy's ``agents/`` directory.
    agents_dir: str = str(TEST_AGENTS_DIR)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls, env_settings, dotenv_settings, file_secret_settings
        return (init_settings,)


def build_test_settings(
    profile: str = "memory", **overrides: Any
) -> IsolatedSettings:
    """Build a validated test configuration from a named safe profile."""
    try:
        values = dict(_PROFILES[profile])
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"unknown test settings profile {profile!r}; expected one of: {choices}"
        ) from exc
    values.update(overrides)
    return IsolatedSettings(**values)
