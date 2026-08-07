# Architecture Decision Records

Numbered, immutable-once-accepted records of decisions (Nygard format:
Context / Decision / Consequences). New ADRs take the next number; a change
of mind is a new ADR that supersedes the old one, noted in both `Status:`
lines. Numbers are identities and freeze once committed.

## Writing one

Mechanism belongs in the design spec; a record holds the decision and the
constraints it imposes. 0005 is the reference shape.

- **Test each Decision sentence:** if the implementation changed shape but the
  decision held, would this need editing? If yes it is spec — cut it, keeping
  whatever would break if it were done differently.
- **Shape:** when Decision runs long, check that the extra lines are
  constraints and not mechanism — a lifecycle record legitimately needs its
  legal states. A modal (must, never, only, owns, requires) is the clearest
  form, and a plain invariant such as "credentials are resolved outside the
  sandbox" decides just as well; a line describing *how* does not.
- **Standalone:** no links to untracked documents, and no vocabulary borrowed
  from one — a plan's stage and gate names mean nothing to a reader here, so
  record the constraint, not the label.

## Records

- [0001 — Surface message standards](0001-surface-message-standards.md)
- [0002 — Authenticate container-runtime control behind an intent broker](0002-authenticated-container-intent-broker.md)
- [0003 — Carry OpenHands traffic through a capability-authenticated generation relay](0003-capability-authenticated-openhands-relay.md)
- [0004 — Gate OpenHands generation release on a durable checkpoint receipt](0004-receipt-gated-openhands-generation-lifecycle.md)
- [0005 — Make a capability-scoped workspace agent the execution context of a durable thread task](0005-workspace-agent-as-thread-task-execution-context.md)
- [0006 — Retire the dedicated analysis worker](0006-retire-dedicated-analysis-worker.md)
- [0007 — Address a deployment as an immutable release tuple](0007-immutable-release-and-configuration-versioning.md)
