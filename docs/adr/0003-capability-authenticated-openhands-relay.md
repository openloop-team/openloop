# 0003 — Carry OpenHands traffic through a capability-authenticated generation relay

- **Status:** Proposed
- **Date:** 2026-07-16

## Context

The pinned OpenHands agent server is an HTTP/WebSocket service. The current
Docker adapter publishes it to host loopback, which allows every process in that
network namespace to reach the listener. Replacing the publish with a
TCP-to-UDS shim does not restore unix-socket authorization: any process that can
reach the shim is forwarded through the shim's own UDS access, including into
OpenHands' WebSocket pre-authentication window.

Directly sharing a bridge network between OpenLoop and the agent is also unsafe.
Ordinary container networking is bidirectional, so the model-controlled agent
could originate a connection toward the controller. Putting the privileged
broker on the HTTP/WebSocket path would avoid that direction but expand the
broker's parser, availability, and credential surface.

Filesystem permissions on a UDS remain useful but cannot distinguish unrelated
workloads that share a uid. The data plane therefore needs its own
generation-bound authentication independent of the OpenHands session key.

## Decision

Create one unprivileged, disposable HAProxy **relay** per OpenHands generation.
It listens on exactly one generation UDS and opens outbound connections to one
fixed upstream, `agent:8000`, on an internal per-generation network. It exposes
no host-side TCP listener and binds no listener on the hostile job network.

OpenLoop uses an owned compatibility adapter rather than a TCP shim:

- httpx HTTP transport over the generation UDS;
- `websockets.unix_connect` for the event stream;
- relay-capability injection for REST and WebSocket handshakes;
- REST `X-Session-API-Key` injection; and
- first-message WebSocket session authentication.

The adapter is guarded by a boot-time compatibility probe against the pinned
OpenHands version. An incompatible SDK seam fails before a job starts.

Every generation has two independent credentials:

1. A **relay capability**, known only to the broker, relay, and owner-authorized
   adapter. It reaches the relay through a secret file or inherited descriptor,
   never command arguments or container labels. HAProxy rejects a missing or
   incorrect `X-OpenLoop-Relay-Capability` before contacting the agent and
   removes the header before forwarding.
2. An **agent session key**, validated end-to-end by OpenHands. It never grants
   relay access by itself.

The relay uses a broker-generated fixed configuration:

- exact method/path ACLs for the pinned SDK plus `/health`; `/api/` is not a
  wildcard;
- WebSocket upgrade only for the job's exact conversation UUID;
- fixed upstream and normalized `Host`;
- no `CONNECT`, absolute-form target, or caller-selected destination;
- bounded headers, connections, tunnel lifetime, memory, and timeouts;
- a checkpoint-only read ACL baked into the same immutable configuration and
  selected by the broker at generation quiescence — the relay exposes no
  runtime admin interface; and
- no credential-bearing request or query-string logs.

The agent and relay share a private internal network. The agent's only other
network leg reaches a per-job, deny-by-default egress proxy that performs DNS
and rejects non-allowlisted, private, loopback, link-local, metadata, and
control-plane destinations after resolution.

A loopback shim is permitted only in an explicitly single-tenant development
profile. A multi-tenant production process refuses to start with that profile or
with a host/controller-side compatibility TCP listener.

The privileged container broker provisions and revokes the relay but is never
on this traffic path; its decision is recorded in
[ADR 0002](0002-authenticated-container-intent-broker.md). The transition into
checkpoint-only mode and receipt-gated teardown are recorded in
[ADR 0004](0004-receipt-gated-openhands-generation-lifecycle.md).

## Consequences

- The agent cannot initiate traffic toward OpenLoop or the broker; it can only
  answer the relay's already-established upstream connections.
- Discovering the UDS path or sharing its Unix uid does not grant data-plane
  authority.
- Capability confidentiality still assumes OS-level process isolation: a
  same-UID workload that can read the adapter's memory or secret files obtains
  the relay capability. Deployment isolation must deny cross-workload
  `ptrace`, `/proc/<pid>/mem`, and credential-file access, as in
  [ADR 0002](0002-authenticated-container-intent-broker.md).
- The relay capability never enters the model-controlled agent, so compromise
  of the agent session key does not unlock the host-side data path.
- The production path has no loopback compatibility listener and avoids placing
  the WebSocket session key in a URI.
- OpenLoop must own and test a small transport adapter across HTTP streaming,
  WebSocket authentication, reconnect, and close behavior for every pinned SDK
  upgrade.
- HAProxy parses agent-controlled response bytes and therefore remains pinned,
  unprivileged, resource-limited, and disposable. It is defense-in-depth, not a
  payload sanitizer.
- The egress proxy and DNS/firewall probes become mandatory production
  infrastructure rather than configuration guidance.
