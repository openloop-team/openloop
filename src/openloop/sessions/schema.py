"""Describes the surface-session and thread tables for statement construction
only (ADR 0007).

Not the schema's definition — the stores' `setup()` still owns that until ADR
0009 moves it to Alembic. Nothing here emits DDL.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

surface_sessions = Table(
    "surface_sessions",
    metadata,
    Column("id", Text, primary_key=True),
    Column("surface", Text, nullable=False),
    Column("workspace", Text, nullable=False),
    Column("agent", Text, nullable=False),
    Column("channel", Text),
    Column("thread", Text),
    # The idempotency key for an inbound surface event; a partial unique index
    # in the database makes a concurrent redelivery fail the insert.
    Column("event_id", Text),
    Column("status", Text, nullable=False),
    Column("workflow_instance_id", Text),
    Column("progress_message_id", Text),
    Column("final_message_id", Text),
    Column("approval_ids", JSONB, nullable=False),
    Column("request_text", Text),
    Column("result_summary", Text),
    Column("result_artifact_ref", Text),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
