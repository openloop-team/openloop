"""Describes the broker's tables for statement construction only (ADR 0007).

The migration files under `migrations/` define this schema; this module only
describes it, and where the two disagree the database is right. Nothing here is
consulted to create or alter anything, and none of the CHECK constraints those
files carry are restated — they are the database's to enforce.
"""

from __future__ import annotations

from sqlalchemy import (
    CHAR,
    BigInteger,
    Column,
    DateTime,
    Integer,
    MetaData,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

broker_schema_migrations = Table(
    "broker_schema_migrations",
    metadata,
    Column("version", Integer, primary_key=True),
    Column("name", Text, nullable=False),
    Column("checksum", CHAR(64), nullable=False),
    Column("applied_at", DateTime(timezone=True), nullable=False),
)

broker_jobs = Table(
    "broker_jobs",
    metadata,
    Column("job_id", UUID(as_uuid=True), primary_key=True),
    Column("conversation_id", UUID(as_uuid=True), nullable=False),
    Column("tenant_id", Text, nullable=False),
    Column("workload_subject", Text, nullable=False),
    Column("profile", Text, nullable=False),
    Column("runtime_driver", Text, nullable=False),
    Column("durable_state_driver", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("revision", BigInteger, nullable=False),
    Column("generation", BigInteger, nullable=False),
    Column("current_generation", BigInteger),
    Column("pending_operation_id", UUID(as_uuid=True)),
    Column("durable_state_ref", Text),
    Column("durable_key_version", Text),
    Column("durable_digest", CHAR(64)),
    Column("terminal_outcome", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    # Added by 0002_rpc_authorization.sql; all four are set together or not
    # at all (the database enforces that with num_nonnulls).
    Column("minimum_isolation", Text),
    Column("control_key_version", Text),
    Column("control_epoch", BigInteger),
    Column("control_capability_digest", CHAR(64)),
)

broker_generations = Table(
    "broker_generations",
    metadata,
    Column("job_id", UUID(as_uuid=True), primary_key=True),
    Column("generation", BigInteger, primary_key=True),
    Column("state", Text, nullable=False),
    Column("revision", BigInteger, nullable=False),
    Column("previous_job_state", Text, nullable=False),
    Column("start_operation_id", UUID(as_uuid=True), nullable=False),
    Column("pending_operation_id", UUID(as_uuid=True)),
    Column("runtime_ref", Text),
    Column("durable_state_ref", Text),
    Column("runtime_key_version", Text),
    Column("durable_key_version", Text),
    Column("capability_digest", CHAR(64)),
    Column("durable_digest", CHAR(64)),
    Column("execution_lease_deadline", DateTime(timezone=True), nullable=False),
    Column("barrier_id", Text),
    # The verified-receipt block: all sixteen NULL together, or all present.
    Column("receipt_issuer", Text),
    Column("receipt_id", Text),
    Column("receipt_tenant_id", Text),
    Column("receipt_job_id", UUID(as_uuid=True)),
    Column("receipt_conversation_id", UUID(as_uuid=True)),
    Column("receipt_generation", BigInteger),
    Column("receipt_barrier_id", Text),
    Column("receipt_artifact_id", Text),
    Column("receipt_base_commit", Text),
    Column("receipt_ciphertext_sha256", CHAR(64)),
    Column("receipt_plaintext_sha256", CHAR(64)),
    Column("receipt_byte_count", BigInteger),
    Column("receipt_store_version", Text),
    Column("receipt_envelope_version", Text),
    Column("receipt_key_version", Text),
    Column("receipt_durable_write_sequence", BigInteger),
    Column("release_target", Text),
    Column("release_terminal_outcome", Text),
    Column("failure_reason_code", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

broker_operations = Table(
    "broker_operations",
    metadata,
    Column("operation_id", UUID(as_uuid=True), primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("workload_subject", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("idempotency_key", Text),
    Column("command_kind", Text, nullable=False),
    Column("request_digest", CHAR(64), nullable=False),
    Column("job_id", UUID(as_uuid=True)),
    Column("generation", BigInteger),
    Column("status", Text, nullable=False),
    Column("intent_ticket", JSONB, nullable=False),
    Column("completion_result", JSONB),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

broker_audit = Table(
    "broker_audit",
    metadata,
    Column("audit_id", BigInteger, primary_key=True),
    Column("command_kind", Text, nullable=False),
    Column("tenant_id", Text, nullable=False),
    Column("workload_subject", Text, nullable=False),
    Column("job_id", UUID(as_uuid=True), nullable=False),
    Column("generation", BigInteger),
    Column("operation_id", UUID(as_uuid=True), nullable=False),
    Column("before_job_state", Text),
    Column("after_job_state", Text, nullable=False),
    Column("before_generation_state", Text),
    Column("after_generation_state", Text),
    Column("reason_code", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
