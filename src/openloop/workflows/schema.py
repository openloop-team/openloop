"""Describes `workflow_instances` for statement construction only (ADR 0007)."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

workflow_instances = Table(
    "workflow_instances",
    metadata,
    Column("id", Text, primary_key=True),
    Column("workflow", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("completed_steps", JSONB, nullable=False),
    Column("state", JSONB, nullable=False),
    Column("waiting_on", Text),
    Column("result", JSONB),
    Column("error", Text),
    # Store-owned: never written from an instance payload.
    Column("leased_until", DateTime(timezone=True)),
    Column("drive_gen", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
