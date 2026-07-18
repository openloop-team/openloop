"""Suite-wide test environment.

``openloop.app`` still loads route-shaping agent metadata at import time. Point
it at the test fixture before any test module imports the ASGI shell, keeping
tests decoupled from ``agents/dev-platform.yaml``.
"""

import os
from pathlib import Path

os.environ["AGENTS_DIR"] = str(Path(__file__).parent / "integration" / "data")
