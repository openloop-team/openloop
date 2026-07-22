# Retire the dedicated analysis worker

This runbook removes the disabled dedicated analysis worker It is intentionally
fail-closed: any retained analysis data stops the procedure.

Do not apply the destructive SQL before every application replica runs the
retirement release. The previous composition root recreates analysis tables on
startup even when `ANALYSIS_WORKER_ENABLED` is false.

## Preconditions

- Confirm the deployed worker is disabled.
- Confirm deployed agent configuration contains neither native tool `analysis`
  nor approval action `analysis.report:write`.
- Confirm there are no known internal or external callers.
- Select and record the exact deployment, database service, and analysis
  workspace path. Do not use a wildcard or an unresolved path for cleanup.
- Take and verify the deployment's normal PostgreSQL backup.

Use a configured PostgreSQL service entry so credentials do not enter shell
history or the process list. The examples below assume `PGSERVICE` names the
reviewed target:

```bash
PGSERVICE=openloop-production psql --set ON_ERROR_STOP=1 \
  --file ops/postgres/2026-07-22-audit-analysis-worker.sql
```

The audit prints one count per dedicated and shared category. Every count must
be zero. Save the output with the deployment record. A missing shared table is
an error because it makes the audit incomplete.

Inspect Docker state without deleting anything:

```bash
docker ps -a --filter label=openloop.sandbox=analysis
```

The command must return no analysis container. Inspect the exact configured
analysis workspace directory and stop if it contains any file. Do not remove a
residual container or workspace until its owner and contents are understood.

## Roll out application removal

1. Deploy the release that no longer registers the action, initializes analysis
   stores, records Slack uploads for analysis, or interprets `analysis://`
   artifacts.
2. Remove all `ANALYSIS_WORKER_*` variables and the native analysis tool from
   deployment configuration.
3. Remove `docker-compose.sandbox.yml` from deployment composition.
4. Start containerized OpenHands only with the external broker configuration.
5. Drain every old application replica.
6. Verify the runtime service has no `/var/run/docker.sock` mount and the broker
   is the only service that does.
7. Restart one new runtime replica and confirm it does not recreate an analysis
   table.

## Recheck and retire the schema

Run the read-only audit again after all old replicas are gone:

```bash
PGSERVICE=openloop-production psql --set ON_ERROR_STOP=1 \
  --file ops/postgres/2026-07-22-audit-analysis-worker.sql
```

If every count remains zero, apply the guarded transaction:

```bash
PGSERVICE=openloop-production psql --set ON_ERROR_STOP=1 \
  --file ops/postgres/2026-07-22-retire-analysis-worker.sql
```

The transaction locks all inspected tables, recomputes every count, and aborts
without dropping anything if a late write or retained row exists. It drops only
the five dedicated analysis tables when the locked counts are zero.

## Postflight

1. Run the read-only audit a third time. Dedicated tables report zero by
   absence; every shared reference count remains zero.
2. Restart a runtime replica and repeat the audit to prove application startup
   does not recreate the tables.
3. Invoke a non-production request for `analysis.report:write` and confirm it
   receives the ordinary unknown-action response.
4. Recheck the effective Compose configuration and running mounts: only the
   standalone broker owns the Docker socket.
5. Record SQL output, application version, replica drain confirmation, backup
   identifier, and postflight results with the deployment change.

## Failure and rollback

- A nonzero or missing-table audit is a stop condition. Do not edit the SQL to
  bypass it. Return to architecture review for a retention decision.
- Before the schema drop, roll back through the ordinary application deployment
  mechanism.
- After the schema drop, rollback requires both the previous application
  version and the verified pre-deployment database backup.
- The migration is safe to rerun after success, but rerunning it is not a
  substitute for the postflight checks.
