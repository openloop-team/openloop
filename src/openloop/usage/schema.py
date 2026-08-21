"""Describes the `usage` table for statement construction only (ADR 0007)."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Table,
    Text,
)

metadata = MetaData()

usage = Table(
    "usage",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("scope_key", Text, nullable=False),
    Column("workspace", Text, nullable=False),
    Column("agent", Text, nullable=False),
    Column("channel", Text),
    Column("surface", Text),
    # Quoted in SQL because `user` is reserved; SQLAlchemy quotes it for us.
    Column("user", Text),
    Column("task_kind", Text),
    Column("idempotency_key", Text),
    Column("model", Text, nullable=False),
    Column("prompt_tokens", Integer, nullable=False),
    Column("completion_tokens", Integer, nullable=False),
    Column("cost_usd", Float, nullable=False),
    Column("outcome", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("job_id", Text),
    Column("broker_job_id", Text),
    Column("broker_generation", BigInteger),
    Column("approval_id", Text),
    Column("approver", Text),
    Column("session_id", Text),
)
