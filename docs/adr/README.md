# Architecture Decision Records

Numbered, immutable-once-accepted records of decisions (Nygard format:
Context / Decision / Consequences). A change of mind is a new ADR that
supersedes the old one, noted in both `Status:` lines.

Numbers are identities: a record takes the next free one and it freezes once
committed. A number is claimed by writing the record, never reserved ahead of
it — a plan that names a digit for a decision nobody has written yet will find
it taken by whatever landed first, so plans should name the decision instead.

## Writing one

Mechanism belongs in the design spec; a record holds the decision and the
constraints it imposes. Start from [template.md](template.md).

- **Whether to write one:** the subject needs live alternatives a reasonable
  person could have chosen instead. A correctness fix nobody would argue the
  other side of is not a decision, and neither is a preference that no state
  of the repository can violate. Test the alternatives against the code before
  drafting: if they collapse on inspection, the record was not deciding
  anything and the work is elsewhere.
- **Each Decision sentence:** if the implementation changed shape but the
  decision held, would this need editing? If yes it is spec — cut it, keeping
  whatever would break if it were done differently.
- **Shape:** when Decision runs long, check that the extra lines are
  constraints and not mechanism — a lifecycle record legitimately needs its
  legal states. A modal (must, never, only, owns, requires) is the clearest
  form, and a plain invariant such as "credentials are resolved outside the
  sandbox" decides just as well; a line describing *how* does not.
- **Consequences:** carry costs — what gets harder, what is given up, what
  must now be done differently. A consequence that restates the Decision in
  other words is not one, and a list with nothing lost in it means the
  alternatives were never weighed.
- **Standalone:** no links to untracked documents, and no vocabulary borrowed
  from one — a plan's stage and gate names mean nothing to a reader here, so
  record the constraint, not the label.

## Header fields

Bullets between the title and `## Context`, in this order.

- **`Status:`** is one of `Proposed`, `Accepted`, `Rejected`, or `Superseded`.
  A rejected record stays in the log with its Context and alternatives intact:
  what was considered and declined is worth as much as what was chosen.
  Supersession names the other record on both sides — the old one reads
  `Superseded by 0010`, the new one `Accepted (supersedes 0006)`.
- **`Date:`** is the day the record was written, and it does not change when
  the status does.
- **`Implementation:` (optional):** bounds what the record authorizes
  touching, written as *what it covers; what it leaves otherwise unchanged*.
  Add it when the subject is broader than the licence, so the bound is read
  before the body; omit it when Decision already says where the record stops.
  It states scope, never progress — status and sequencing are planned outside
  the repository.

## Records

- [0001 — Surface message standards](0001-surface-message-standards.md)
- [0002 — Authenticate container-runtime control behind an intent broker](0002-authenticated-container-intent-broker.md)
- [0003 — Carry OpenHands traffic through a capability-authenticated generation relay](0003-capability-authenticated-openhands-relay.md)
- [0004 — Gate OpenHands generation release on a durable checkpoint receipt](0004-receipt-gated-openhands-generation-lifecycle.md)
- [0005 — Make a capability-scoped workspace agent the execution context of a durable thread task](0005-workspace-agent-as-thread-task-execution-context.md)
- [0006 — Retire the dedicated analysis worker](0006-retire-dedicated-analysis-worker.md)
