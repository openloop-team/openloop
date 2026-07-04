"""Suite-wide test environment.

``openloop.app`` builds its module-level ``app = create_app()`` at import time,
loading agents from ``settings.agents_dir`` (default: the shipped ``agents/``
examples). Point it at the test fixture instead — this file is imported before
any test module, so the env var is set before that import can happen — keeping
tests decoupled from ``agents/dev-platform.yaml``.
"""

import os
from pathlib import Path

os.environ["AGENTS_DIR"] = str(Path(__file__).parent / "integration" / "data")
