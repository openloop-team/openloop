-- Destructive schema retirement for the removed dedicated analysis worker.
--
-- REQUIRED PRECONDITIONS (stop if any item is unverified):
--   1. Select and record the exact database target. Run psql with
--      ON_ERROR_STOP=1, and take and verify the normal PostgreSQL backup.
--   2. Deploy the release that removes the analysis action, stores, upload
--      recording, staging command, and analysis artifact interpretation.
--   3. Confirm the worker is disabled, agent/deployment configuration contains
--      no analysis tool or action, and there are no known callers.
--   4. Remove ANALYSIS_WORKER_* configuration and the retired sandbox overlay.
--      Containerized OpenHands must use the external broker, and only that
--      broker may own the Docker socket.
--   5. Drain every old application replica. Restart one replacement replica
--      and confirm it does not recreate any dedicated analysis table.
--   6. Run 2026-07-22-audit-analysis-worker.sql after the drain, save its
--      output, and require every dedicated and shared category count to be
--      zero. A missing shared table is a stop condition.
--   7. Confirm no residual analysis container exists and the exact configured
--      analysis workspace is empty. Do not delete unexplained residual state.
--
-- ROLLOUT ORDER:
--   deploy removal -> drain old replicas -> restart/prove no recreation ->
--   run and record the read-only audit -> apply this guarded transaction ->
--   rerun the audit -> restart once more and rerun the audit.
--
-- Never edit this transaction to bypass a nonzero or incomplete audit. Before
-- the DROP, roll back through the normal application deployment mechanism.
-- After the DROP, rollback requires both the verified backup and the previous
-- application release.
--
-- This transaction refuses to delete any dedicated or shared analysis state.
-- Old application replicas must be drained first because their composition
-- root recreates the dedicated tables even while the worker is disabled.

BEGIN;

DO $analysis_retirement$
DECLARE
    schema_name TEXT := current_schema();
    table_name TEXT;
    category TEXT;
    row_count BIGINT;
    missing_shared TEXT[] := ARRAY[]::TEXT[];
    blocked JSONB := '{}'::JSONB;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'workflow_instances',
        'approvals',
        'usage',
        'surface_sessions'
    ]
    LOOP
        IF to_regclass(format('%I.%I', schema_name, table_name)) IS NULL THEN
            missing_shared := array_append(missing_shared, table_name);
        END IF;
    END LOOP;

    IF cardinality(missing_shared) > 0 THEN
        RAISE EXCEPTION 'analysis retirement audit is incomplete; missing required shared tables: %',
            array_to_string(missing_shared, ', ');
    END IF;

    -- Freeze every inspected shared surface before counting. SHARE conflicts
    -- with inserts/updates/deletes but leaves ordinary readers available.
    FOREACH table_name IN ARRAY ARRAY[
        'workflow_instances',
        'approvals',
        'usage',
        'surface_sessions'
    ]
    LOOP
        EXECUTE format('LOCK TABLE %I.%I IN SHARE MODE', schema_name, table_name);
    END LOOP;

    -- Lock existing dedicated tables in the same deterministic order. The
    -- subsequent DROP upgrades these locks inside this transaction.
    FOREACH table_name IN ARRAY ARRAY[
        'analysis_staged_inputs',
        'analysis_uploads',
        'analysis_artifacts',
        'analysis_attempts',
        'analysis_inputs'
    ]
    LOOP
        IF to_regclass(format('%I.%I', schema_name, table_name)) IS NOT NULL THEN
            EXECUTE format('LOCK TABLE %I.%I IN SHARE MODE', schema_name, table_name);
            EXECUTE format('SELECT count(*) FROM %I.%I', schema_name, table_name)
                INTO row_count;
            IF row_count > 0 THEN
                blocked := blocked || jsonb_build_object(table_name, row_count);
            END IF;
        END IF;
    END LOOP;

    FOR category, row_count IN
        SELECT 'workflow_instances.analysis_worker', count(*)
        FROM workflow_instances
        WHERE workflow = 'analysis_worker'
        UNION ALL
        SELECT 'approvals.analysis_action', count(*)
        FROM approvals
        WHERE action = 'analysis.report:write'
        UNION ALL
        SELECT 'approvals.analysis_tool', count(*)
        FROM approvals
        WHERE tool = 'analysis'
        UNION ALL
        SELECT 'usage.analysis_worker', count(*)
        FROM usage
        WHERE task_kind = 'analysis_worker'
        UNION ALL
        SELECT 'surface_sessions.analysis_artifact', count(*)
        FROM surface_sessions
        WHERE result_artifact_ref LIKE 'analysis://%'
    LOOP
        IF row_count > 0 THEN
            blocked := blocked || jsonb_build_object(category, row_count);
        END IF;
    END LOOP;

    IF blocked <> '{}'::JSONB THEN
        RAISE EXCEPTION 'analysis retirement blocked by nonempty categories: %',
            blocked;
    END IF;

    FOREACH table_name IN ARRAY ARRAY[
        'analysis_staged_inputs',
        'analysis_uploads',
        'analysis_artifacts',
        'analysis_attempts',
        'analysis_inputs'
    ]
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %I.%I', schema_name, table_name);
    END LOOP;
END
$analysis_retirement$;

COMMIT;
