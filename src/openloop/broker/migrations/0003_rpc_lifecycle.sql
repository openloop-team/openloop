ALTER TABLE broker_rpc_audit
    DROP CONSTRAINT broker_rpc_audit_method_check;

ALTER TABLE broker_rpc_audit
    ADD CONSTRAINT broker_rpc_audit_method_check CHECK (
        method IN (
            'CREATE_JOB',
            'INSPECT_JOB',
            'START_SEGMENT',
            'QUIESCE_SEGMENT',
            'RELEASE_SEGMENT',
            'FINALIZE_JOB'
        )
    );
