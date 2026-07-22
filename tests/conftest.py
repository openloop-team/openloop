"""Suite-wide test environment.

``openloop.app`` still loads route-shaping agent metadata at import time. Point
it at the test fixture before any test module imports the ASGI shell, keeping
tests decoupled from ``agents/dev-platform.yaml``.
"""

import os
from pathlib import Path
import shutil

import pytest

from tests.support.socket_paths import create_short_socket_root

os.environ["AGENTS_DIR"] = str(Path(__file__).parent / "integration" / "data")


@pytest.fixture
def short_socket_root():
    """Owner-private, realpath-resolved root with room for nested broker UDSes."""
    root = create_short_socket_root()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
