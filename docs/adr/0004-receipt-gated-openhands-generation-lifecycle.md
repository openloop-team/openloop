# 0004 — Gate OpenHands generation release on a durable checkpoint receipt

- **Status:** Proposed
- **Date:** 2026-07-16

## Context

Cold resume is correct only when both authorities needed for continuation are
durable: the OpenHands conversation store and an authenticated workspace
artifact. A stop operation that both establishes quiescence and destroys the
agent offers no safe interval in which OpenLoop can capture the workspace after
the barrier while the server is still alive.

The previous lifecycle also overloaded `TERMINAL` for both final jobs and failed
starts, and described returning a generation capability before committing the
generation `RUNNING`. Those orderings make crash recovery ambiguous: a caller
can hold a capability that inspection will not return, and recovery cannot tell
whether a terminal record means cleanup or retry.

Retaining a mutable workspace volume while parked is not a solution. It creates
a second resume authority, bypasses artifact verification, and prevents movement
to a different node. Host-local Docker volumes likewise do not satisfy the
cold-resume requirement for shared durable state.

## Decision

Model job state separately from generation state.

```text
job:        CREATED → ACTIVE ↔ PARKED
            any nonterminal state → FINALIZING → TERMINAL
generation: STARTING → RUNNING → QUIESCING → QUIESCED
                                           → RELEASING → RELEASED
             any state before RELEASING ───────────→ ABANDONED
```

`TERMINAL` is reserved for a job with an explicit success, cancellation, or
failure outcome. `ABANDONED` is reserved for a failed generation; a generation
that reaches `RELEASING` always converges to `RELEASED`. Generation numbers and
operation ids are monotonic and never reused.

`create_job` allocates the job `CREATED` with broker-minted identifiers, an
immutable owner, and a durable-state namespace. A generation starts only from a
stable `CREATED` or `PARKED` job; the persisted pending generation fences
concurrent starts. Every terminal path — releasing a final generation, explicit
cancellation, or a recorded terminal failure — passes through `FINALIZING`,
where `finalize_job` performs terminal cleanup before committing `TERMINAL`.

Starting a generation follows this ordering:

1. Transactionally allocate and persist the generation, operation id,
   `STARTING`, and pending intent.
2. Runtime and durable-state drivers idempotently ensure resources.
3. Health-check through the generation relay.
4. Atomically persist the opaque runtime handle, capability metadata,
   generation `RUNNING`, and job `ACTIVE`.
5. Only after that commit, return the generation capability.

If the response is lost, owner-authorized inspection re-derives and returns the
capability. A failed start marks only that generation `ABANDONED`, returns the
job to its previous stable state, and requires an explicit bounded retry that
allocates a new generation.

Stopping is split into two intents, `quiesce_segment` and `release_segment`,
with the checkpoint capture between them:

1. `quiesce_segment` records the application barrier, fences mutation and new
   WebSockets, drains in-flight work, switches the relay to a fixed read-only
   checkpoint surface, commits `QUIESCED`, and leaves the agent alive.
2. OpenLoop stores the encrypted workspace artifact. The trusted checkpoint
   store returns a signed durable receipt binding tenant, job, conversation,
   generation, barrier, base commit, artifact identity, hashes, and store/key
   versions.
3. `release_segment` verifies the receipt, then atomically persists its
   metadata, the target job state, and `RELEASING`. Teardown revokes traffic
   and idempotently deletes the relay, agent, network, socket, and generation
   workspace scratch; job durable state is detached, never deleted.
4. After teardown, it commits generation `RELEASED` and the persisted target,
   job `PARKED` or `FINALIZING`.

Conversation state uses a broker-managed durable-state driver attachable by
every compatible node. Workspace scratch is fresh per generation and never
durable authority. Resume reconstructs scratch from the authenticated artifact.
Loss of compute never deletes durable state; deletion requires a terminal job
outcome plus completed external finalization, explicit cancellation policy, or
an administrative retention deadline.

Production uses database time and fenced leases. Recovery completes persisted
intents by generation and operation id. On expiry in `QUIESCED`, a valid receipt
completes parking; no receipt fails the job closed rather than replaying
uncertain work. Explicit cancellation may bypass a receipt only while recording
a terminal cancellation/failure outcome and retaining required evidence.

The broker and relay decisions used by this lifecycle are recorded in
[ADR 0002](0002-authenticated-container-intent-broker.md) and
[ADR 0003](0003-capability-authenticated-openhands-relay.md).

## Consequences

- No externally usable generation exists before its durable `RUNNING` state.
- The broker can prove that workspace state is durable before normal teardown;
  caller assertion is not the transition boundary.
- Lost replies and crashes during resource creation, checkpointing, or teardown
  converge through persisted intent, idempotency, and deterministic receipt
  lookup.
- Failed generations no longer collide with terminal job semantics.
- Parked jobs can resume on another compatible node using shared conversation
  state plus the verified artifact; mutable workspace volumes are disposable.
- The agent and checkpoint-only relay remain live between quiescence and receipt
  persistence, consuming bounded resources and requiring an execution lease.
- The checkpoint store becomes a trusted signer with key rotation and
  deterministic receipt lookup requirements.
- Checkpoint-store availability gates parking: an outage during the quiesce
  window fails otherwise-parkable jobs closed instead of parking them.
- The shared state schema, reconciler, retention policy, and fault-injection
  matrix become more complex, but uncertain model/tool effects continue to fail
  closed rather than being silently replayed.
