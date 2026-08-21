"""ADR 0007's one mechanically checkable completion condition.

Only SQLAlchemy's dialect may reach the database driver. This is the whole of
what tooling can settle; that every surviving SQL string carries a stated reason
is settled in review.

The check is AST-based, not textual. A grep for "asyncpg" would flag the dialect
name itself — `postgresql+asyncpg://` in a URL, the `_DRIVER` constant in
`openloop/db/engine.py` — which is the one place the name legitimately appears.
Naming the driver is not calling it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROOTS = (ROOT / "src", ROOT / "tests")
DRIVER = "asyncpg"
# Metadata describes a schema and never defines one.
DEFINING_METHODS = frozenset({"create_all", "drop_all", "reflect"})
DEFINING_KEYWORD = "autoload_with"


def _modules():
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path, ast.parse(path.read_text(), filename=str(path))


def _where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(ROOT)}:{node.lineno}"


@pytest.mark.unit
def test_no_module_reaches_the_driver():
    offenders: list[str] = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == DRIVER for a in node.names):
                    offenders.append(_where(path, node))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == DRIVER:
                    offenders.append(_where(path, node))
            elif isinstance(node, ast.Attribute):
                # asyncpg.create_pool(...), asyncpg.connect(...)
                if isinstance(node.value, ast.Name) and node.value.id == DRIVER:
                    offenders.append(_where(path, node))
    assert offenders == [], (
        "ADR 0007: only SQLAlchemy's dialect may reach the driver; found "
        + ", ".join(offenders)
    )


@pytest.mark.unit
def test_no_metadata_defines_a_schema():
    offenders: list[str] = []
    for path, tree in _modules():
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in DEFINING_METHODS:
                offenders.append(f"{_where(path, node)} ({func.attr})")
            if any(kw.arg == DEFINING_KEYWORD for kw in node.keywords):
                offenders.append(f"{_where(path, node)} ({DEFINING_KEYWORD})")
    assert offenders == [], (
        "ADR 0007: metadata describes a schema and never defines one; found "
        + ", ".join(offenders)
    )
