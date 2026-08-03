# 0007 — Bind one continuable workspace task to one durable thread

- **Status:** Accepted
- **Date:** 2026-08-03
- **Implementation:** thread↔task binding, continuation entry, and code-profile
  re-entry shipped with this record.

## Context

[ADR 0005](0005-workspace-agent-as-thread-task-execution-context.md) makes a
capability-scoped workspace agent the execution context of a durable thread
task, and requires that every execution capability be reconstructable from
durable records.

A workspace task had no identity of its own. It existed inside whatever ran it —
a workflow instance's state, a worker checkpoint — so it ended when that run
ended. A later reply in the same thread therefore had nothing to continue: the
model was asked again, delegated again, and a second task was born with its own
branch, its own pull request, and its own approval. The user's one piece of work
became several, and neither the conversation nor the workspace carried across
them.

The alternatives considered were to keep the model in the loop and have it
re-delegate against the earlier task's parameters, or to hold the earlier run's
execution context alive between replies. The first leaves task identity as
something a model must reconstruct correctly each turn — and pays for a model
round to decide it. The second makes continuation depend on warm state, which
0005 forbids.

## Decision

A thread scope binds to at most one workspace task, durably and outside any run,
and eligible replies in that thread are turns of that task.

- Task identity, its authorization envelope, and its serialized state are held in
  a durable record independent of any workflow instance, surface session, or
  execution context. A turn's records reference the task; the task never
  references a turn as its identity.
- A continuation is re-entry, not re-delegation: the task id, its workspace
  identity (for the code profile, its branch and pull request), the approval that
  authorized it, and the agent its spend attributes to are read from the durable
  record and carried forward unchanged. Only per-turn facts — the request, the
  session delivering it, the instance driving it — are new.
- A profile's gate is a property of the task, not of each turn. `Gate.START`
  therefore admits a continuation without a second approval; a profile that
  requires per-effect authorization expresses that as its own gate, and no
  continuation may bypass a gate a profile declares.
- Continuation is authorized, not merely addressed: a reply continues a task only
  if its author is the human the durable record names as the task's initiator. An
  unattributed task and an unattributed reply are both refused.
- Refusal is never a dead end. A reply that cannot continue the task is an
  ordinary turn, which may delegate new work through the ordinary gate.
- One task runs one turn at a time. A reply that arrives while its task is
  executing is durably queued and delivered to that task afterwards, never run
  beside it.
- Every one of these decisions is made from durable records alone, so a replica
  that has never seen the thread reaches the same one.

## Consequences

- Task identity now outlives the workflow instance and the surface session that
  created it, and a restart mid-task is recoverable rather than merely survivable:
  the next reply reconstructs the task cold.
- A claim orphaned by a crash must be repairable from the durable workflow
  instance, because a task that cannot be unclaimed wedges its thread forever.
- Binding a thread to a task changes what a reply means in that thread. Widening
  eligibility beyond the task's initiator, or admitting a continuation across a
  gate a profile declares, requires a new decision record.
- A per-task spend cap now bounds a single attempt, while a thread-bound task may
  span many. Making the cap cover a task's whole life is a change to what the cap
  means and is decided separately.
- Retiring a task — so that its thread is free to start another — becomes a real
  lifecycle event a surface must be able to reach. Denial at the start gate is
  the only retirement this record admits; the rest is later work.
- A second profile joining continuation must supply its own re-entry rule (what
  identity it preserves, what it resets); nothing about the code profile's branch
  reuse generalizes on its own.
