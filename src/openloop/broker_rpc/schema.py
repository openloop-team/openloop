"""Describes `broker_rpc_audit` for statement construction only (ADR 0007).

Not the table's definition: broker migration `0002_rpc_authorization.sql`
creates it and the sink only appends. Nothing here emits DDL.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import UUID

metadata = MetaData()

broker_rpc_audit = Table(
    "broker_rpc_audit",
    metadata,
    Column("sequence", BigInteger, primary_key=True),
    Column("request_id", UUID(as_uuid=True), nullable=False),
    Column("method", Text, nullable=False),
    Column("decision", Text, nullable=False),
    Column("reason_code", Text, nullable=False),
    Column("peer_pid", BigInteger, nullable=False),
    Column("peer_uid", BigInteger, nullable=False),
    Column("peer_gid", BigInteger, nullable=False),
    Column("tenant_id", Text, nullable=False),
    Column("workload_subject", Text, nullable=False),
    Column("worker_instance_id", UUID(as_uuid=True), nullable=False),
    Column("assignment_id", UUID(as_uuid=True), nullable=False),
    Column("isolation_mode", Text, nullable=False),
    Column("required_isolation", Text, nullable=False),
    Column("jwt_key_id", Text, nullable=False),
    Column("jwt_id", UUID(as_uuid=True), nullable=False),
    Column("job_id", UUID(as_uuid=True)),
    Column("operation_id", UUID(as_uuid=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
