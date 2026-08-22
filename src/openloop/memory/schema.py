"""Describes the `memories` table for statement construction only.

This is not the schema's definition — `PostgresMemoryStore.setup` still owns
that until schema evolution moves to a migration tool. Nothing here emits DDL.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

from openloop.db.types import Vector

# text-embedding-3-small is 1536-dim; change alongside the embedder.
DEFAULT_EMBEDDING_DIM = 1536

metadata = MetaData()

memories = Table(
    "memories",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("scope_key", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("embedding", Vector(DEFAULT_EMBEDDING_DIM)),
    Column("metadata", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
