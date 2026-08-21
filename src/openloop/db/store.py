"""Store base class — durable stores borrow, never own, an engine (ADR 0007)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


class BorrowedEngineStore:
    """Base for a store that borrows, but never disposes, an `AsyncEngine`."""

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None

    async def setup(self, engine: AsyncEngine) -> None:
        """Bind *engine* and ensure this store's schema.

        Declared here because the composition root calls it on every borrowed
        store it settles, and knows them only by this base.
        """
        raise NotImplementedError

    @asynccontextmanager
    async def _setup_connection(
        self, engine: AsyncEngine
    ) -> AsyncIterator[AsyncConnection]:
        """Bind *engine* while schema setup runs in one transaction.

        Clears the binding on failure so a store that failed setup cannot be
        mistaken for a usable one.
        """
        if engine is None:
            raise TypeError("setup() requires a caller-owned SQLAlchemy engine")
        self._engine = engine
        try:
            async with engine.begin() as connection:
                yield connection
        except BaseException:
            self._engine = None
            raise

    async def close(self) -> None:
        """Detach from the borrowed engine without disposing the caller's."""
        self._engine = None

    def _require_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called first")
        return self._engine
