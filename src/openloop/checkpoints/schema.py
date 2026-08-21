"""Describes `worker_checkpoints` for statement construction only (ADR 0007)."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

worker_checkpoints = Table(
    "worker_checkpoints",
    metadata,
    Column("job_id", Text, primary_key=True),
    Column("repo", Text, nullable=False),
    Column("instruction", Text, nullable=False),
    Column("base", Text, nullable=False),
    Column("branch", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("completed_steps", JSONB, nullable=False),
    Column("state_json", JSONB, nullable=False),
    Column("title", Text),
    Column("body", Text),
    Column("pr_number", Integer),
    Column("pr_url", Text),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
