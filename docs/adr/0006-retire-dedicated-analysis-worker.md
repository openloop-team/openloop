# 0006 — Retire the dedicated analysis worker

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

[ADR 0005](0005-workspace-agent-as-thread-task-execution-context.md) makes one
capability-scoped workspace agent the execution context of a durable thread
task, with investigation, research, coding, artifact creation, and proposed
effects as stages of that task rather than separate user-visible worker
products.

The existing analysis worker is a separate product slice. It accepts approved
staged, Slack-upload, or GitHub-archive inputs; asks a model to author Python;
runs the program in a network-disabled container; reads one Markdown report;
and delivers it through analysis-specific persistence and session code. Keeping
it would require a second broker profile and continued specialized workflow,
provisioning, artifact, recovery, configuration, and testing machinery.

The worker is deployed but disabled and has no consumers. Its dedicated
PostgreSQL tables and references in shared stores can be checked before removal.
The model-facing runtime currently retains a legacy Compose path that grants it
the Docker socket for this worker and older coding modes, even though the
external broker now supplies the intended process boundary.

## Decision

Retire the dedicated analysis worker instead of migrating it to the broker.

Remove its native tool, durable workflow, input and artifact stores, CLI,
Slack-upload path, session-delivery branches, configuration, agent exposure,
container image, deployment wiring, active documentation, and tests. An old
`analysis.report:write` request receives the ordinary unknown-action response;
there is no compatibility shim.

Run a repository-owned, operator-invoked PostgreSQL retirement migration only
after a fail-closed preflight proves that all dedicated tables and every
analysis-linked row in shared workflow, approval, usage, and session tables are
empty. A nonzero check aborts the transaction without deleting data. When the
checks pass, drop `analysis_staged_inputs`, `analysis_uploads`,
`analysis_artifacts`, `analysis_attempts`, and the possible legacy
`analysis_inputs` table. Do not run destructive DDL automatically at application
startup.

Remove direct Docker authority from the model-facing runtime. Delete the legacy
Compose socket-mount path and application-side Docker execution choices.
Containerized OpenHands requires the enabled broker in `external` mode; a
missing broker, direct runtime selection, or coprocess selection fails closed.
The existing `CODING_WORKER_SANDBOX=docker` value may remain temporarily as the
marker for broker-hosted OpenHands until the general execution-profile contract
replaces it.

Retain shared control-plane mechanisms and broker-owned security properties,
not unused worker abstractions. Future sealed computation, if justified by a
real workflow, must be designed as a narrow capability of the thread-bound
workspace task.

## Consequences

- OpenLoop loses the standalone sealed-analysis tool and its direct report
  workflow. This is intentional because it has no consumer.
- Later work starts from fewer outcome-specific abstractions and can validate a
  general workspace-task boundary against real workflows.
- The model-facing runtime no longer needs or accepts Docker authority;
  container lifecycle belongs to the external broker.
- Production retirement requires an explicit backup, two empty-state checks
  around the application rollout, and an operator-run schema drop.
- Any discovered analysis data blocks removal and requires a new retention
  decision; shared audit history is never silently erased.
- Generic artifact references, approvals, workflows, usage accounting, thread
  persistence, broker limits, workspace ingress, receipts, and recovery remain.
- Historical sealed-analysis design documents may remain as design history but
  are no longer operational guidance.
- Reintroducing sealed computation requires a new decision record and security
  review rather than reverting this ADR implicitly.
