"""Describes the `approvals` table for statement construction only."""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Integer, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

approvals = Table(
    "approvals",
    metadata,
    Column("id", Text, primary_key=True),
    Column("agent", Text, nullable=False),
    Column("action", Text, nullable=False),
    Column("tool", Text, nullable=False),
    Column("permission", Text, nullable=False),
    Column("args", JSONB, nullable=False),
    Column("approvers", JSONB, nullable=False),
    Column("summary", Text, nullable=False),
    Column("requested_by", Text),
    Column("status", Text, nullable=False),
    Column("decided_by", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # NULL doubles as the pre-version sentinel version-checking consumers refuse.
    Column("args_schema", Integer),
    # NULL workflow_backed marks a legacy row the resolver classifies by
    # registry; NULL effect_at keeps a decided row in the reconciler's sweep.
    Column("workflow_backed", Boolean),
    Column("workflow_instance_id", Text),
    Column("effect_at", DateTime(timezone=True)),
)
