"""Describes the surface-session and thread tables for statement construction
only.

Not the schema's definition — the stores' `setup()` still owns that until schema
evolution moves to a migration tool. Nothing here emits DDL.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, MetaData, Table, Text
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

surface_threads = Table(
    "surface_threads",
    metadata,
    Column("scope_key", Text, primary_key=True),
    Column("surface", Text, nullable=False),
    Column("workspace", Text, nullable=False),
    Column("agent", Text, nullable=False),
    Column("channel", Text),
    Column("thread", Text),
    Column("active_turn_id", Text),
    Column("context_ref", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

surface_thread_transcript = Table(
    "surface_thread_transcript",
    metadata,
    Column("scope_key", Text, primary_key=True),
    Column("turn_id", Text, primary_key=True),
    Column("seq", BigInteger),
    Column("request", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

surface_thread_inbox = Table(
    "surface_thread_inbox",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("scope_key", Text, nullable=False),
    Column("event_id", Text, nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
