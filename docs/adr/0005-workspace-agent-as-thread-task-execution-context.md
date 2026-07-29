# 0005 — Make a capability-scoped workspace agent the execution context of a durable thread task

- **Status:** Accepted
- **Date:** 2026-07-22
- **Implementation:** direction only; each capability it admits requires its own
  decision record.

## Context

OpenLoop delivered work through product-specific workers: a coding worker whose
output is a pull request, and a sealed analysis worker whose output is a report.
Each carried a near-complete control plane of its own — workflow, durable state,
artifact handling, delivery branches, provisioning, configuration, and tests.

That structure imposes costs that grow with every capability added:

- The user must choose a worker before knowing what the work will require. A
  question that turns out to need a code change was addressed to the wrong
  product.
- A task that begins as investigation and ends as a fix has no representation.
  Evidence cannot cross from one worker to the next, because each worker owns
  its own results.
- Every new capability — another connector, another artifact type — implies
  another worker and another copy of the control plane.
- Security properties are re-derived per worker instead of held once. Each new
  worker is a fresh opportunity to get isolation, credential handling, or
  approval wrong.

The alternative considered was to keep specializing: retain each worker and give
it a narrow execution profile behind the broker. That multiplies control planes
rather than capabilities, and it leaves the user-visible taxonomy — pick a
worker — in place.

## Decision

Make one capability-scoped workspace agent the execution context of a durable
thread task. Investigation, research, coding, execution, and artifact production
are stages of a single user-visible task rather than separate worker products.

The following are invariants of that direction, not implementation detail:

- The durable control plane retains identity, policy, budget, approvals,
  persistence, recovery, and delivery authority. None of that authority moves
  into a sandbox.
- Sandboxes are replaceable execution environments, never sources of truth. No
  task invariant may depend on warm state — a live container or an existing
  checkout is a performance optimization only.
- A session receives only the capabilities its current work requires and cannot
  broaden its own grants. The model may request elevation; it cannot grant it.
- Credentials are resolved and injected outside the sandbox. A reusable
  credential never lands inside one.
- Sensitive effects stay explicit: high-risk writes pass through the durable
  approval and idempotency plane.
- Where data boundaries require isolation, one durable task may own several
  capability-scoped sessions. One unrestricted session with access to
  everything is not an acceptable substitute, transitionally or otherwise.
- Shared abstractions are adopted from evidence: after a second real profile
  uses one without a worker-specific escape hatch, not in anticipation of it.
- Scheduled and event-triggered work reuses this task, execution, approval, and
  delivery model rather than acquiring a parallel automation architecture.

## Consequences

- [0006](0006-retire-dedicated-analysis-worker.md) is a consequence of this
  record rather than an independent decision, and later records may cite these
  invariants instead of restating them.
- Reaching production data or production credentials from a workspace session
  requires its own decision record, preceded by a threat model and an
  adversarial test suite. This ADR does not authorize that exposure.
- Reintroducing a product-specific worker, or moving policy, approval, identity,
  or delivery authority into a sandbox, requires an ADR superseding this one.
- Because no invariant may rest on warm state, every execution capability must
  be reconstructable from durable records — which is a real constraint on how
  execution features may be built, not a preference.
- Sequencing, status, and staged rollout of this migration are planned outside
  the repository and are deliberately not recorded here. What is durable is the
  direction and its invariants; those change only by supersession. Records that
  depend on this one should cite the invariants, not a plan's phase names.
