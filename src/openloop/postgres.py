"""Shared PostgreSQL pool lifecycle primitives.

Durable stores borrow a caller-owned pool.  Pool creation and shutdown belong
to the application (or another top-level caller), never to an individual store.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class BorrowedPostgresStore:
    """Base class for a store that borrows, but never owns, an asyncpg pool."""

    def __init__(self) -> None:
        self._pool: Any | None = None

    @asynccontextmanager
    async def _setup_connection(self, pool: Any) -> AsyncIterator[Any]:
        """Bind *pool* while schema setup runs, clearing it on failure."""
        if pool is None:
            raise TypeError("setup() requires a caller-owned Postgres pool")
        self._pool = pool
        try:
            async with pool.acquire() as conn:
                yield conn
        except BaseException:
            self._pool = None
            raise

    async def close(self) -> None:
        """Detach from the borrowed pool without closing the caller's resource."""
        self._pool = None

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called first")
        return self._pool


async def create_pool(
    dsn: str, *, min_size: int, max_size: int
) -> Any:
    """Create an asyncpg pool without importing asyncpg at module import time."""
    import asyncpg

    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
