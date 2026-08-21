"""Postgres + pgvector memory backend.

Stores records in a `memories` table; recall ranks by vector distance when a
query embedding is supplied (`embedding <=> $query`), otherwise by recency.
Requires the `vector` extension (the `pgvector/pgvector` image ships it).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import cast, literal, select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from openloop.db import BorrowedEngineStore
from openloop.db.types import Vector
from openloop.memory.schema import DEFAULT_EMBEDDING_DIM, memories
from openloop.memory.store import MemoryRecord

__all__ = ["DEFAULT_EMBEDDING_DIM", "PostgresMemoryStore"]


def _vec_literal(embedding: list[float]) -> str:
    """pgvector accepts a bracketed, comma-separated text literal."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


class PostgresMemoryStore(BorrowedEngineStore):
    """pgvector-backed :class:`~openloop.memory.store.MemoryStore`."""

    def __init__(self, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    async def setup(self, engine: AsyncEngine) -> None:
        """Bind a caller-owned engine and create the extension, table, and index."""
        async with self._setup_connection(engine) as conn:
            # sql-text: schema evolution moves to Alembic (ADR 0009); DDL is not
            # restated as metadata, and the vector column's width is a runtime
            # value the expression language has no place for.
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS memories (
                        id          BIGSERIAL PRIMARY KEY,
                        scope_key   TEXT NOT NULL,
                        kind        TEXT NOT NULL DEFAULT 'message',
                        text        TEXT NOT NULL,
                        embedding   vector({self.embedding_dim}),
                        metadata    JSONB NOT NULL DEFAULT '{{}}',
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS memories_scope_idx "
                    "ON memories (scope_key, created_at DESC)"
                )
            )

    async def remember(self, record: MemoryRecord) -> None:
        engine = self._require_engine()
        embedding = _vec_literal(record.embedding) if record.embedding else None
        statement = memories.insert().values(
            scope_key=record.scope_key,
            kind=record.kind,
            text=record.text,
            # An explicit cast: a bare text parameter has no implicit coercion
            # to `vector` on the extended query protocol.
            embedding=cast(literal(embedding), Vector(self.embedding_dim)),
            metadata=record.metadata,
            created_at=record.created_at,
        )
        async with engine.begin() as conn:
            await conn.execute(statement)

    async def recall(
        self,
        scope_key: str,
        query_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        columns = (
            memories.c.scope_key,
            memories.c.kind,
            memories.c.text,
            memories.c.metadata,
            memories.c.created_at,
        )
        if query_embedding is not None:
            distance = memories.c.embedding.op("<=>")(
                cast(literal(_vec_literal(query_embedding)), Vector(self.embedding_dim))
            )
            statement = (
                select(*columns)
                .where(
                    memories.c.scope_key == scope_key,
                    memories.c.embedding.is_not(None),
                )
                .order_by(distance)
                .limit(limit)
            )
        else:
            statement = (
                select(*columns)
                .where(memories.c.scope_key == scope_key)
                .order_by(memories.c.created_at.desc())
                .limit(limit)
            )

        engine = self._require_engine()
        async with engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()

        return [
            MemoryRecord(
                scope_key=row["scope_key"],
                text=row["text"],
                kind=row["kind"],
                # JSONB arrives decoded — the dialect registers a jsonb codec.
                metadata=row["metadata"] or {},
                created_at=row["created_at"] or datetime.now(UTC),
            )
            for row in rows
        ]
