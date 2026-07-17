CREATE TABLE IF NOT EXISTS broker_schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    name TEXT NOT NULL,
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE broker_jobs (
    job_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL CHECK (octet_length(tenant_id) BETWEEN 1 AND 128),
    workload_subject TEXT NOT NULL CHECK (
        octet_length(workload_subject) BETWEEN 1 AND 256
    ),
    profile TEXT NOT NULL CHECK (octet_length(profile) BETWEEN 1 AND 64),
    runtime_driver TEXT NOT NULL CHECK (
        octet_length(runtime_driver) BETWEEN 1 AND 64
    ),
    durable_state_driver TEXT NOT NULL CHECK (
        octet_length(durable_state_driver) BETWEEN 1 AND 64
    ),
    state TEXT NOT NULL CHECK (
        state IN ('created', 'active', 'parked', 'finalizing', 'terminal')
    ),
    revision BIGINT NOT NULL CHECK (revision > 0),
    generation BIGINT NOT NULL CHECK (generation >= 0),
    current_generation BIGINT,
    pending_operation_id UUID,
    durable_state_ref TEXT CHECK (
        durable_state_ref IS NULL
        OR octet_length(durable_state_ref) BETWEEN 1 AND 1024
    ),
    durable_key_version TEXT CHECK (
        durable_key_version IS NULL
        OR octet_length(durable_key_version) BETWEEN 1 AND 256
    ),
    durable_digest CHAR(64) CHECK (
        durable_digest IS NULL OR durable_digest ~ '^[0-9a-f]{64}$'
    ),
    terminal_outcome TEXT CHECK (
        terminal_outcome IS NULL
        OR terminal_outcome IN ('success', 'cancelled', 'failed')
    ),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CHECK (current_generation IS NULL OR current_generation > 0),
    CHECK (current_generation IS NULL OR current_generation <= generation),
    CHECK (
        (state IN ('finalizing', 'terminal') AND terminal_outcome IS NOT NULL)
        OR (state NOT IN ('finalizing', 'terminal') AND terminal_outcome IS NULL)
    )
);

CREATE TABLE broker_generations (
    job_id UUID NOT NULL REFERENCES broker_jobs(job_id),
    generation BIGINT NOT NULL CHECK (generation > 0),
    state TEXT NOT NULL CHECK (
        state IN (
            'starting',
            'running',
            'quiescing',
            'quiesced',
            'releasing',
            'released',
            'abandoned'
        )
    ),
    revision BIGINT NOT NULL CHECK (revision > 0),
    previous_job_state TEXT NOT NULL CHECK (
        previous_job_state IN ('created', 'parked')
    ),
    start_operation_id UUID NOT NULL,
    pending_operation_id UUID,
    runtime_ref TEXT CHECK (
        runtime_ref IS NULL OR octet_length(runtime_ref) BETWEEN 1 AND 1024
    ),
    durable_state_ref TEXT CHECK (
        durable_state_ref IS NULL
        OR octet_length(durable_state_ref) BETWEEN 1 AND 1024
    ),
    runtime_key_version TEXT CHECK (
        runtime_key_version IS NULL
        OR octet_length(runtime_key_version) BETWEEN 1 AND 256
    ),
    durable_key_version TEXT CHECK (
        durable_key_version IS NULL
        OR octet_length(durable_key_version) BETWEEN 1 AND 256
    ),
    capability_digest CHAR(64) CHECK (
        capability_digest IS NULL OR capability_digest ~ '^[0-9a-f]{64}$'
    ),
    durable_digest CHAR(64) CHECK (
        durable_digest IS NULL OR durable_digest ~ '^[0-9a-f]{64}$'
    ),
    execution_lease_deadline TIMESTAMPTZ NOT NULL,
    barrier_id TEXT CHECK (
        barrier_id IS NULL OR octet_length(barrier_id) BETWEEN 1 AND 256
    ),
    receipt_issuer TEXT,
    receipt_id TEXT,
    receipt_tenant_id TEXT,
    receipt_job_id UUID,
    receipt_conversation_id UUID,
    receipt_generation BIGINT,
    receipt_barrier_id TEXT,
    receipt_artifact_id TEXT,
    receipt_base_commit TEXT,
    receipt_ciphertext_sha256 CHAR(64),
    receipt_plaintext_sha256 CHAR(64),
    receipt_byte_count BIGINT,
    receipt_store_version TEXT,
    receipt_envelope_version TEXT,
    receipt_key_version TEXT,
    receipt_durable_write_sequence BIGINT,
    release_target TEXT CHECK (
        release_target IS NULL OR release_target IN ('parked', 'finalizing')
    ),
    release_terminal_outcome TEXT CHECK (
        release_terminal_outcome IS NULL
        OR release_terminal_outcome IN ('success', 'cancelled', 'failed')
    ),
    failure_reason_code TEXT CHECK (
        failure_reason_code IS NULL
        OR octet_length(failure_reason_code) BETWEEN 1 AND 64
    ),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (job_id, generation),
    CHECK (
        (state IN ('starting', 'quiescing', 'releasing')
            AND pending_operation_id IS NOT NULL)
        OR (state NOT IN ('starting', 'quiescing', 'releasing')
            AND pending_operation_id IS NULL)
    ),
    CHECK (
        state NOT IN ('quiescing', 'quiesced', 'releasing', 'released')
        OR barrier_id IS NOT NULL
    ),
    CHECK (
        (state IN ('releasing', 'released')
            AND receipt_id IS NOT NULL
            AND release_target IS NOT NULL)
        OR (state NOT IN ('releasing', 'released')
            AND receipt_id IS NULL
            AND release_target IS NULL)
    ),
    CHECK (
        (
            receipt_id IS NULL
            AND receipt_issuer IS NULL
            AND receipt_tenant_id IS NULL
            AND receipt_job_id IS NULL
            AND receipt_conversation_id IS NULL
            AND receipt_generation IS NULL
            AND receipt_barrier_id IS NULL
            AND receipt_artifact_id IS NULL
            AND receipt_base_commit IS NULL
            AND receipt_ciphertext_sha256 IS NULL
            AND receipt_plaintext_sha256 IS NULL
            AND receipt_byte_count IS NULL
            AND receipt_store_version IS NULL
            AND receipt_envelope_version IS NULL
            AND receipt_key_version IS NULL
            AND receipt_durable_write_sequence IS NULL
        )
        OR (
            receipt_id IS NOT NULL
            AND receipt_issuer IS NOT NULL
            AND receipt_tenant_id IS NOT NULL
            AND receipt_job_id IS NOT NULL
            AND receipt_conversation_id IS NOT NULL
            AND receipt_generation > 0
            AND receipt_barrier_id IS NOT NULL
            AND receipt_artifact_id IS NOT NULL
            AND receipt_base_commit ~ '^(?:[0-9a-f]{40}|[0-9a-f]{64})$'
            AND receipt_ciphertext_sha256 ~ '^[0-9a-f]{64}$'
            AND receipt_plaintext_sha256 ~ '^[0-9a-f]{64}$'
            AND receipt_byte_count >= 0
            AND receipt_store_version IS NOT NULL
            AND receipt_envelope_version IS NOT NULL
            AND receipt_key_version IS NOT NULL
            AND receipt_durable_write_sequence >= 0
            AND octet_length(receipt_issuer) BETWEEN 1 AND 256
            AND octet_length(receipt_id) BETWEEN 1 AND 256
            AND octet_length(receipt_tenant_id) BETWEEN 1 AND 128
            AND octet_length(receipt_barrier_id) BETWEEN 1 AND 256
            AND octet_length(receipt_artifact_id) BETWEEN 1 AND 256
            AND octet_length(receipt_store_version) BETWEEN 1 AND 256
            AND octet_length(receipt_envelope_version) BETWEEN 1 AND 256
            AND octet_length(receipt_key_version) BETWEEN 1 AND 256
            AND receipt_job_id = job_id
            AND receipt_generation = generation
            AND receipt_barrier_id = barrier_id
        )
    ),
    CHECK (
        (release_target = 'finalizing' AND release_terminal_outcome IS NOT NULL)
        OR (release_target IS DISTINCT FROM 'finalizing'
            AND release_terminal_outcome IS NULL)
    ),
    CHECK (
        (state = 'abandoned' AND failure_reason_code IS NOT NULL)
        OR (state <> 'abandoned' AND failure_reason_code IS NULL)
    )
);

CREATE UNIQUE INDEX broker_one_live_generation_per_job
ON broker_generations (job_id)
WHERE state IN ('starting', 'running', 'quiescing', 'quiesced', 'releasing');

CREATE TABLE broker_operations (
    operation_id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL CHECK (octet_length(tenant_id) BETWEEN 1 AND 128),
    workload_subject TEXT NOT NULL CHECK (
        octet_length(workload_subject) BETWEEN 1 AND 256
    ),
    source TEXT NOT NULL CHECK (source IN ('caller', 'internal')),
    idempotency_key TEXT CHECK (
        idempotency_key IS NULL
        OR octet_length(idempotency_key) BETWEEN 16 AND 128
    ),
    command_kind TEXT NOT NULL CHECK (
        command_kind IN (
            'create_job',
            'begin_start',
            'mark_running',
            'abandon_generation',
            'begin_quiesce',
            'mark_quiesced',
            'begin_release',
            'mark_released',
            'begin_finalize',
            'mark_terminal'
        )
    ),
    request_digest CHAR(64) NOT NULL CHECK (
        request_digest ~ '^[0-9a-f]{64}$'
    ),
    job_id UUID,
    generation BIGINT CHECK (generation IS NULL OR generation > 0),
    status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
    intent_ticket JSONB NOT NULL,
    completion_result JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (source = 'caller' AND idempotency_key IS NOT NULL)
        OR (source = 'internal' AND idempotency_key IS NULL)
    ),
    CHECK (jsonb_typeof(intent_ticket) = 'object'),
    CHECK (octet_length(intent_ticket::text) <= 16384),
    CHECK (
        completion_result IS NULL
        OR (
            jsonb_typeof(completion_result) = 'object'
            AND octet_length(completion_result::text) <= 16384
        )
    ),
    FOREIGN KEY (job_id) REFERENCES broker_jobs(job_id)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (job_id, generation)
        REFERENCES broker_generations(job_id, generation)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE UNIQUE INDEX broker_caller_idempotency
ON broker_operations (tenant_id, workload_subject, idempotency_key)
WHERE idempotency_key IS NOT NULL;

CREATE TABLE broker_audit (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    command_kind TEXT NOT NULL CHECK (
        command_kind IN (
            'create_job',
            'begin_start',
            'mark_running',
            'abandon_generation',
            'begin_quiesce',
            'mark_quiesced',
            'begin_release',
            'mark_released',
            'begin_finalize',
            'mark_terminal'
        )
    ),
    tenant_id TEXT NOT NULL CHECK (octet_length(tenant_id) BETWEEN 1 AND 128),
    workload_subject TEXT NOT NULL CHECK (
        octet_length(workload_subject) BETWEEN 1 AND 256
    ),
    job_id UUID NOT NULL REFERENCES broker_jobs(job_id),
    generation BIGINT,
    operation_id UUID NOT NULL REFERENCES broker_operations(operation_id),
    before_job_state TEXT CHECK (
        before_job_state IS NULL
        OR before_job_state IN ('created', 'active', 'parked', 'finalizing', 'terminal')
    ),
    after_job_state TEXT NOT NULL CHECK (
        after_job_state IN ('created', 'active', 'parked', 'finalizing', 'terminal')
    ),
    before_generation_state TEXT CHECK (
        before_generation_state IS NULL
        OR before_generation_state IN (
            'starting', 'running', 'quiescing', 'quiesced',
            'releasing', 'released', 'abandoned'
        )
    ),
    after_generation_state TEXT CHECK (
        after_generation_state IS NULL
        OR after_generation_state IN (
            'starting', 'running', 'quiescing', 'quiesced',
            'releasing', 'released', 'abandoned'
        )
    ),
    reason_code TEXT CHECK (
        reason_code IS NULL OR octet_length(reason_code) BETWEEN 1 AND 64
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (operation_id, command_kind),
    FOREIGN KEY (job_id, generation)
        REFERENCES broker_generations(job_id, generation)
        DEFERRABLE INITIALLY DEFERRED
);

ALTER TABLE broker_jobs
ADD CONSTRAINT broker_jobs_pending_operation_fk
FOREIGN KEY (pending_operation_id) REFERENCES broker_operations(operation_id)
DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE broker_generations
ADD CONSTRAINT broker_generations_start_operation_fk
FOREIGN KEY (start_operation_id) REFERENCES broker_operations(operation_id)
DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE broker_generations
ADD CONSTRAINT broker_generations_pending_operation_fk
FOREIGN KEY (pending_operation_id) REFERENCES broker_operations(operation_id)
DEFERRABLE INITIALLY DEFERRED;
