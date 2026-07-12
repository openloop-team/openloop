"""Postgres + pgvector memory backend.

Stores records in a `memories` table; recall ranks by vector distance when a
query embedding is supplied (`embedding <=> $query`), otherwise by recency.
Requires the `vector` extension (the `pgvector/pgvector` image ships it).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openloop.memory.store import MemoryRecord
from openloop.postgres import BorrowedPostgresStore

# text-embedding-3-small is 1536-dim; change alongside the embedder.
DEFAULT_EMBEDDING_DIM = 1536


def _vec_literal(embedding: list[float]) -> str:
    """pgvector accepts a bracketed, comma-separated text literal."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


class PostgresMemoryStore(BorrowedPostgresStore):
    """pgvector-backed :class:`~openloop.memory.store.MemoryStore`."""

    def __init__(self, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    async def setup(self, pool) -> None:
        """Bind a caller-owned pool and create the extension, table, and index."""
        async with self._setup_connection(pool) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
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
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memories_scope_idx "
                "ON memories (scope_key, created_at DESC)"
            )

    async def remember(self, record: MemoryRecord) -> None:
        pool = self._require_pool()
        embedding = _vec_literal(record.embedding) if record.embedding else None
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (scope_key, kind, text, embedding, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                record.scope_key,
                record.kind,
                record.text,
                embedding,
                json.dumps(record.metadata),
                record.created_at,
            )

    async def recall(
        self,
        scope_key: str,
        query_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if query_embedding is not None:
                rows = await conn.fetch(
                    """
                    SELECT scope_key, kind, text, metadata, created_at
                    FROM memories
                    WHERE scope_key = $1 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $2
                    LIMIT $3
                    """,
                    scope_key,
                    _vec_literal(query_embedding),
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT scope_key, kind, text, metadata, created_at
                    FROM memories
                    WHERE scope_key = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    scope_key,
                    limit,
                )

        return [
            MemoryRecord(
                scope_key=row["scope_key"],
                text=row["text"],
                kind=row["kind"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"] or datetime.now(timezone.utc),
            )
            for row in rows
        ]
