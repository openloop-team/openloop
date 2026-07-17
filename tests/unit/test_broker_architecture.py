"""Architecture boundaries that keep the broker ledger unprivileged."""

import ast
from pathlib import Path


BROKER_ROOT = Path(__file__).parents[2] / "src" / "openloop" / "broker"

BANNED_IMPORT_PREFIXES = (
    "docker",
    "openloop.app",
    "openloop.runtime",
    "openloop.sandbox",
    "openloop.tools",
    "openloop.workers",
    "openhands",
)

BANNED_IMPORT_PARTS = ("haproxy", "kms", "relay", "rpc", "worker")


def test_broker_modules_do_not_import_privileged_or_runtime_layers():
    violations = []
    for path in sorted(BROKER_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                lowered = name.lower()
                if lowered.startswith(BANNED_IMPORT_PREFIXES) or any(
                    part in lowered.split(".") for part in BANNED_IMPORT_PARTS
                ):
                    violations.append((path.name, node.lineno, name))
    assert violations == []


def test_broker_package_has_no_generic_public_mutation_escape_hatch():
    banned_names = {"save", "update", "upsert", "execute", "set_state"}
    violations = []
    for path in sorted(BROKER_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in banned_names:
                    violations.append((path.name, node.lineno, node.name))
    assert violations == []

