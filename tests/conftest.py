"""Shared fixtures for hermetic tests."""

import shutil

import pytest

from tests.support.settings import IsolatedSettings, build_test_settings
from tests.support.socket_paths import create_short_socket_root


@pytest.fixture
def settings_factory():
    """Return a factory for validated, environment-independent settings."""
    return build_test_settings


@pytest.fixture
def settings(settings_factory) -> IsolatedSettings:
    """Default process-local configuration for composition-root tests."""
    return settings_factory()


@pytest.fixture
def short_socket_root():
    """Owner-private, realpath-resolved root with room for nested broker UDSes."""
    root = create_short_socket_root()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
