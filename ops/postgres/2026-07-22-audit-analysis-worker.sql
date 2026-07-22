-- Read-only production audit for the dedicated analysis-worker retirement.
-- Run with psql ON_ERROR_STOP. Every emitted count must be zero before rollout
-- and again before applying 2026-07-22-retire-analysis-worker.sql.

DO $analysis_retirement_audit$
DECLARE
    schema_name TEXT := current_schema();
    table_name TEXT;
    row_count BIGINT;
    missing_shared TEXT[] := ARRAY[]::TEXT[];
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

    FOREACH table_name IN ARRAY ARRAY[
        'analysis_staged_inputs',
        'analysis_uploads',
        'analysis_artifacts',
        'analysis_attempts',
        'analysis_inputs'
    ]
    LOOP
        IF to_regclass(format('%I.%I', schema_name, table_name)) IS NULL THEN
            row_count := 0;
        ELSE
            EXECUTE format('SELECT count(*) FROM %I.%I', schema_name, table_name)
                INTO row_count;
        END IF;
        RAISE NOTICE 'analysis retirement count: % = %', table_name, row_count;
    END LOOP;

    EXECUTE format(
        'SELECT count(*) FROM %I.workflow_instances WHERE workflow = $1',
        schema_name
    ) INTO row_count USING 'analysis_worker';
    RAISE NOTICE 'analysis retirement count: workflow_instances.analysis_worker = %',
        row_count;

    EXECUTE format(
        'SELECT count(*) FROM %I.approvals WHERE action = $1',
        schema_name
    ) INTO row_count USING 'analysis.report:write';
    RAISE NOTICE 'analysis retirement count: approvals.analysis_action = %', row_count;

    EXECUTE format(
        'SELECT count(*) FROM %I.approvals WHERE tool = $1',
        schema_name
    ) INTO row_count USING 'analysis';
    RAISE NOTICE 'analysis retirement count: approvals.analysis_tool = %', row_count;

    EXECUTE format(
        'SELECT count(*) FROM %I.usage WHERE task_kind = $1',
        schema_name
    ) INTO row_count USING 'analysis_worker';
    RAISE NOTICE 'analysis retirement count: usage.analysis_worker = %', row_count;

    EXECUTE format(
        'SELECT count(*) FROM %I.surface_sessions '
        'WHERE result_artifact_ref LIKE $1',
        schema_name
    ) INTO row_count USING 'analysis://%';
    RAISE NOTICE 'analysis retirement count: surface_sessions.analysis_artifact = %',
        row_count;
END
$analysis_retirement_audit$;
