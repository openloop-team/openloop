"""Column types SQLAlchemy does not ship (ADR 0007).

Describe-only, like every declaration in this project: `get_col_spec` exists so
a CAST renders, never so a table is created.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType[str]):
    """pgvector's `vector(n)`. Values cross the wire as bracketed text."""

    cache_ok = True

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_col_spec(self, **kw: Any) -> str:
        return f"vector({self.dim})"
