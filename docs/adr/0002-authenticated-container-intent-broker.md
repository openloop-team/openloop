# 0002 — Authenticate container-runtime control behind an intent broker

- **Status:** Proposed
- **Date:** 2026-07-16

## Context

The OpenLoop process that parses untrusted model output currently launches
OpenHands containers through `HardenedDockerLaunch.command()`. It therefore has
access to Docker, whose control socket is root-on-host equivalent. The sealed
analysis worker has the same architectural problem through its own
`DockerSandbox` launch path.

Moving the socket into a helper is not sufficient. A helper that accepts
caller-selected images, commands, mounts, environment variables, networks, or
runtime handles merely recreates the Docker API as a confused deputy. Unix
socket modes are also insufficient authorization for the multi-tenant target:
unrelated controller workloads may share an OS uid.

The control plane must additionally survive broker restart, coordinate multiple
nodes, preserve immutable tenant ownership, reconcile provider operations that
succeeded immediately before a crash, and permit future runtime and storage
substrates without exposing their primitives to callers.

## Decision

Run a privileged **node broker** on every execution node. The broker owns only
that node's container-runtime socket and exposes reviewed workload intents, not
runtime primitives. OpenLoop and every model-influenced worker are forbidden
from holding a Docker/container-runtime socket in production.

The OpenHands profile exposes only these intents:

- create a tenant-owned job;
- start a generation;
- quiesce a running generation;
- release a durably checkpointed generation;
- inspect an owned job; and
- finalize a finished job to its terminal outcome.

The sealed analysis profile exposes a fixed batch intent with network `none` and
no interactive relay. New workload classes require a new reviewed profile;
existing profiles are never widened toward caller-selected runtime inputs.

Callers never supply images, commands, mounts, environment variables, networks,
storage locations, or runtime handles: images are pinned by digest, and
profiles and drivers compute every runtime primitive. The broker mints job and
conversation identifiers. Caller workspace bytes are data, never a mount
instruction; they enter as a bounded stream tied to the operation id, and an
unprivileged throwaway extractor materializes them into generation-scoped
scratch, keeping traversal, symlink, hardlink, device-node, and decompression
attacks away from the broker.

The control unix socket carries framed authenticated RPC. Every caller presents
a short-lived scheduler-issued workload identity. `create_job` immutably binds
the job to `(tenant_id, workload_subject)`, and every later request authorizes
that owner plus a job-scoped control capability. `SO_PEERCRED` and filesystem
permissions remain defense-in-depth, not primary authorization. Operator
inspection and cancellation use a separate audited role.

Every mutating request includes an idempotency key; mutations of an existing
job also include an expected generation. The broker persists ownership,
job/generation state, pending intent, operation id, key-version metadata, and
database-time leases in a shared transactional store, initially PostgreSQL.
Host-local SQLite implements the same repository contract only for a
single-node development profile; production startup rejects it.

The broker core resolves each profile against two independent implementation
seams:

- a runtime driver that can idempotently `ensure`, `inspect`, and `release`
  resources by broker operation id; and
- a durable-state driver that attaches a broker-managed, portable conversation
  namespace.

A remote runtime must support provider-native idempotency or deterministic
resource discovery through immutable, non-secret labels. An opaque handle
learned only after creation is not an adequate recovery contract.

The privileged broker is never on the OpenHands HTTP/WebSocket data path. That
decision is recorded separately in
[ADR 0003](0003-capability-authenticated-openhands-relay.md). The meanings and
ordering of the OpenHands lifecycle intents are recorded in
[ADR 0004](0004-receipt-gated-openhands-generation-lifecycle.md).

## Consequences

- Compromise or prompt injection in the OpenLoop/model-facing process no longer
  grants a root-equivalent runtime socket.
- Tenant ownership, expected-generation fencing, idempotency, and audit are
  enforced at the single component capable of creating runtime resources.
- Docker, rootless containers, gVisor/Kata, and remote execution can implement
  the same reviewed intents without changing callers.
- The broker becomes high-value privileged infrastructure and requires a small
  implementation, pinned dependencies, strict resource profiles, workload-
  identity verification, KMS-backed key derivation, and focused audit.
- Production now depends on a shared transactional control store, scheduler-
  issued workload identity, node fencing, and runtime-driver reconciliation.
- Same-UID controller support still requires OS isolation that prevents
  cross-process memory and credential-file inspection; application
  authentication cannot repair absent process isolation.
- Direct Docker launch and SQLite authority remain available only in explicitly
  non-production development profiles.
