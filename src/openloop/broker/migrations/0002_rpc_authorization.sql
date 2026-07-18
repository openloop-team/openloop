ALTER TABLE broker_jobs
    ADD COLUMN minimum_isolation TEXT CHECK (
        minimum_isolation IS NULL
        OR minimum_isolation IN ('shared', 'dedicated')
    ),
    ADD COLUMN control_key_version TEXT CHECK (
        control_key_version IS NULL
        OR octet_length(control_key_version) BETWEEN 1 AND 256
    ),
    ADD COLUMN control_epoch BIGINT CHECK (
        control_epoch IS NULL OR control_epoch > 0
    ),
    ADD COLUMN control_capability_digest CHAR(64) CHECK (
        control_capability_digest IS NULL
        OR control_capability_digest ~ '^[0-9a-f]{64}$'
    );

ALTER TABLE broker_jobs
    ADD CONSTRAINT broker_jobs_rpc_authorization_all_or_none CHECK (
        num_nonnulls(
            minimum_isolation,
            control_key_version,
            control_epoch,
            control_capability_digest
        ) IN (0, 4)
    );

CREATE TABLE broker_rpc_audit (
    sequence BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('CREATE_JOB', 'INSPECT_JOB')),
    decision TEXT NOT NULL CHECK (decision IN ('allowed', 'denied', 'error')),
    reason_code TEXT NOT NULL CHECK (
        octet_length(reason_code) BETWEEN 1 AND 64
    ),
    peer_pid BIGINT NOT NULL CHECK (peer_pid >= 0),
    peer_uid BIGINT NOT NULL CHECK (peer_uid >= 0),
    peer_gid BIGINT NOT NULL CHECK (peer_gid >= 0),
    tenant_id TEXT NOT NULL CHECK (
        octet_length(tenant_id) BETWEEN 1 AND 128
    ),
    workload_subject TEXT NOT NULL CHECK (
        octet_length(workload_subject) BETWEEN 1 AND 256
    ),
    worker_instance_id UUID NOT NULL,
    assignment_id UUID NOT NULL,
    isolation_mode TEXT NOT NULL CHECK (
        isolation_mode IN ('shared', 'dedicated')
    ),
    required_isolation TEXT NOT NULL CHECK (
        required_isolation IN ('shared', 'dedicated')
    ),
    jwt_key_id TEXT NOT NULL CHECK (
        octet_length(jwt_key_id) BETWEEN 1 AND 256
    ),
    jwt_id UUID NOT NULL,
    job_id UUID,
    operation_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX broker_rpc_audit_created_at
ON broker_rpc_audit (created_at);

CREATE INDEX broker_rpc_audit_owner
ON broker_rpc_audit (tenant_id, workload_subject, created_at);
