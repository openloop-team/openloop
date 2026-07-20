"""Co-process broker composition — assemble the graph, own the UDS RPC seam.

`build_broker` constructs the full broker control-plane graph inside the app
process and returns a ready :class:`BrokerRpcClient` plus the resources the
coding-worker adapter needs (architecture step 4, first wiring slice). It is
gated behind ``coding_worker_openhands_broker_enabled`` and is **fail-closed**:
any missing or invalid configuration logs a specific error and returns ``None``
so the caller disables the coding worker loudly rather than falling back to the
direct in-process container launch path.

Decision-locked shape (see
``docs/superpowers/specs/2026-07-19-broker-app-integration-design.md``):

- **Co-process, real UDS only** — the graph lives in this process, but the
  client reaches it solely over the ``BrokerRpcServer`` Unix socket; there is no
  in-process shortcut into the application/coordinator.
- **Separated versioned key rings, one per trust domain.** Capability, runtime,
  and receipt roots are independent version→secret maps (rotatable); a root
  reused within or across domains is rejected (fake rotation / shared trust
  line). The identity keypair is generated ephemerally per-process (tokens live
  <=300s and never outlive the process).
- **Receipt trust split.** Per-version Ed25519 receipt keys are derived under a
  dedicated domain: the current version's PRIVATE key is handed to the
  checkpoint-store side (in the returned handle); the broker verifier holds only
  the PUBLIC keys of every version (overlapping verification).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import SecretStr

from openloop.broker.ledger import BrokerLedger
from openloop.broker.memory import InMemoryBrokerRepository
from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker.postgres import PostgresBrokerRepository
from openloop.broker_control.coordinator import BrokerSegmentCoordinator
from openloop.broker_control.development import local_durable_adapter_for_docker
from openloop.broker_control.receipts import (
    CheckpointReceiptIssuer,
    CheckpointReceiptVerifier,
)
from openloop.broker_control.secrets import (
    RuntimeSecretAuthority,
    RuntimeSecretRootRing,
)
from openloop.broker_rpc.application import BrokerRpcApplication
from openloop.broker_rpc.audit import (
    InMemoryRpcAuditSink,
    PeerCredentials,
    PostgresRpcAuditSink,
)
from openloop.broker_rpc.capability import (
    CapabilityRootRing,
    JobCapabilityAuthority,
)
from openloop.broker_rpc.client import BrokerRpcClient
from openloop.broker_rpc.coordinator import BrokerRpcPolicy
from openloop.broker_rpc.identity import (
    WorkloadIdentityIssuer,
    WorkloadIdentityToken,
    WorkloadIdentityVerifier,
    WorkloadIntent,
)
from openloop.broker_rpc.keys import VerificationKeySet
from openloop.broker_rpc.limits import BrokerRpcLimits
from openloop.broker_rpc.peer import (
    LinuxPeerCredentialProvider,
    PeerCredentialProblem,
    PeerCredentialProvider,
    StaticPeerCredentialProvider,
)
from openloop.broker_rpc.server import BrokerRpcServer, UnixSocketPolicy
from openloop.broker_runtime.contract import RuntimeDriver
from openloop.broker_runtime.docker import DockerOpenHandsRuntimeDriver
from openloop.broker_runtime.docker_policy import DockerRuntimeConfig
from openloop.config import Settings

log = logging.getLogger("openloop")

# Fixed control-plane identifiers for the single co-process workload principal.
_OWNER_TENANT = "openloop"
_OWNER_SUBJECT = "coding-worker"
_IDENTITY_KEY_ID = "identity-v1"
# The checkpoint store is the receipt signer; the broker only ever verifies.
_RECEIPT_ISSUER = "checkpoint-store"
# BrokerRpcPolicy tokens — the profile, runtime driver, and durable-state driver
# this slice pins (OpenHands profile × Docker driver × local durable state).
_POLICY_PROFILE = "default"
_RUNTIME_DRIVER = "docker"
_DURABLE_DRIVER = "local"


@dataclass(frozen=True, slots=True)
class BrokerClientHandle:
    """A live broker client plus the checkpoint-store-side signing material.

    ``receipt_issuer`` (current-version PRIVATE key) is deliberately kept out of
    the broker graph — it belongs to the checkpoint store the worker adapter
    signs with (phase 3). ``reconciler`` is filled in phase 3 when the local
    checkpoint store (its receipt locator) is wired. ``loop`` is the app event
    loop the broker client runs on, captured so the synchronous coding-worker
    thread can bridge into the async client via ``run_coroutine_threadsafe``.
    """

    client: BrokerRpcClient
    owner: BrokerOwner
    receipt_issuer: CheckpointReceiptIssuer
    receipt_verifier: CheckpointReceiptVerifier
    loop: asyncio.AbstractEventLoop
    reconciler: Any | None = None


def _decode_roots(
    name: str, roots: dict[str, SecretStr], current_version: str
) -> dict[str, bytes]:
    """Decode a version→base64-32-byte root map, rejecting malformed input."""
    if not roots:
        raise ValueError(f"broker {name} roots must be set when the broker is enabled")
    if current_version not in roots:
        raise ValueError(
            f"broker_{name}_current_version {current_version!r} is absent from "
            f"broker_{name}_roots"
        )
    decoded: dict[str, bytes] = {}
    for version, secret in roots.items():
        try:
            raw = base64.b64decode(secret.get_secret_value(), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"broker {name} root {version!r} is not valid base64"
            ) from exc
        if len(raw) != 32:
            raise ValueError(
                f"broker {name} root {version!r} must decode to 32 bytes"
            )
        decoded[version] = raw
    return decoded


def _reject_reused_roots(*domains: dict[str, bytes]) -> None:
    """A root reused within or across domains is fake rotation / a shared trust
    line — decision 5 forbids it."""
    seen: set[bytes] = set()
    for domain in domains:
        for raw in domain.values():
            if raw in seen:
                raise ValueError(
                    "broker root reused within or across domains (fake rotation "
                    "or a shared trust line)"
                )
            seen.add(raw)


def _derive_receipt_key(root: bytes, domain: str, version: str) -> Ed25519PrivateKey:
    """Domain-separated per-version Ed25519 receipt-signing key from a root."""
    seed = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"{domain}:{version}".encode(),
    ).derive(root)
    return Ed25519PrivateKey.from_private_bytes(seed)


def _peer_credential_provider() -> PeerCredentialProvider:
    """Real SO_PEERCRED peer credentials on Linux; static self-attribution only
    as a DEVELOPMENT fallback where the kernel primitive is unavailable."""
    try:
        return LinuxPeerCredentialProvider()
    except PeerCredentialProblem:
        # No SO_PEERCRED (e.g. macOS dev). Static self-attribution fabricates
        # peer identity and collapses per-peer isolation to a single bucket —
        # acceptable only for local development.
        log.warning(
            "broker peer credentials are STATIC (no SO_PEERCRED on %s) — "
            "DEVELOPMENT ONLY: per-peer isolation and audit peer identity are "
            "not enforced",
            sys.platform,
        )
        return StaticPeerCredentialProvider(
            PeerCredentials(os.getpid(), os.getuid(), os.getgid())
        )


async def build_broker(
    settings: Settings,
    stack: Any,
    *,
    pool: Any | None = None,
    runtime_driver: RuntimeDriver | None = None,
    clock: Callable[[], datetime] | None = None,
) -> BrokerClientHandle | None:
    """Assemble the co-process broker graph behind the flag; fail-closed.

    Returns ``None`` (with a specific log) on any missing/invalid setting so the
    caller disables the coding worker loudly. All owned resources register their
    teardown on ``stack`` before any client can reach them.
    """
    if not settings.coding_worker_openhands_broker_enabled:
        return None
    # Whole-second UTC: generation deadlines derived from this clock flow into
    # RunningGenerationAccess, which rejects any sub-second timestamp — a raw
    # datetime.now(UTC) would make every start_segment fail INTERNAL.
    now = clock or (lambda: datetime.now(UTC).replace(microsecond=0))

    # Everything below is fail-closed: any construction or setup failure logs a
    # specific reason and returns None so the caller disables the coding worker
    # loudly, never crashing app startup. Nothing that can fail runs after the
    # socket binds, so a returned None never leaves a listener bound.
    try:
        for field_name in (
            "broker_control_socket_dir",
            "broker_state_root",
            "broker_runtime_root",
        ):
            if not getattr(settings, field_name):
                raise ValueError(f"{field_name} is required when the broker is enabled")
        capability_roots = _decode_roots(
            "capability",
            settings.broker_capability_roots,
            settings.broker_capability_current_version,
        )
        runtime_roots = _decode_roots(
            "runtime",
            settings.broker_runtime_roots,
            settings.broker_runtime_current_version,
        )
        receipt_roots = _decode_roots(
            "receipt",
            settings.broker_receipt_roots,
            settings.broker_receipt_current_version,
        )
        _reject_reused_roots(capability_roots, runtime_roots, receipt_roots)
        # The configured generation deadline IS the runtime's absolute maximum
        # lifetime, so the coordinator enforces it and rejects a longer lease
        # (without this the cap silently defaulted to the driver's 86400s).
        runtime_config = DockerRuntimeConfig(
            runtime_root=Path(settings.broker_runtime_root),
            state_root=Path(settings.broker_state_root),
            maximum_lifetime_seconds=settings.broker_generation_deadline_seconds,
        )

        # --- key material (separated trust domains) --------------------------
        capability = JobCapabilityAuthority(
            CapabilityRootRing(
                capability_roots,
                current_version=settings.broker_capability_current_version,
            )
        )
        secret_authority = RuntimeSecretAuthority(
            RuntimeSecretRootRing(
                runtime_roots,
                current_version=settings.broker_runtime_current_version,
            )
        )
        receipt_current = settings.broker_receipt_current_version
        receipt_keys = {
            version: _derive_receipt_key(root, settings.broker_receipt_domain, version)
            for version, root in receipt_roots.items()
        }
        receipt_issuer = CheckpointReceiptIssuer(
            private_key=receipt_keys[receipt_current],
            key_id=receipt_current,
            issuer=_RECEIPT_ISSUER,
        )
        receipt_verifier = CheckpointReceiptVerifier(
            public_keys=VerificationKeySet(
                {version: key.public_key() for version, key in receipt_keys.items()}
            ),
            issuer=_RECEIPT_ISSUER,
        )

        # Ephemeral identity keypair: private -> issuer (client), public ->
        # verifier (broker). Nothing at rest; tokens live <=300s in-process.
        identity_key = Ed25519PrivateKey.generate()
        identity_issuer = WorkloadIdentityIssuer(
            private_key=identity_key,
            key_id=_IDENTITY_KEY_ID,
            issuer=settings.broker_identity_issuer,
            audience=settings.broker_identity_audience,
            clock=now,
        )
        identity_verifier = WorkloadIdentityVerifier(
            public_keys={_IDENTITY_KEY_ID: identity_key.public_key()},
            issuer=settings.broker_identity_issuer,
            audience=settings.broker_identity_audience,
            clock=now,
        )

        # --- ledger / durable audit / runtime / coordinator ------------------
        if pool is not None:
            repository: Any = PostgresBrokerRepository()
            await repository.setup(pool)
            # Durable broker state demands a durable RPC audit trail: the
            # in-memory sink would drop authenticated security decisions on
            # restart while their effects survive. The broker migrations (run by
            # the repository setup above) own the broker_rpc_audit table.
            audit_sink: Any = PostgresRpcAuditSink()
            await audit_sink.setup(pool)
        else:
            repository = InMemoryBrokerRepository(clock=now)
            audit_sink = InMemoryRpcAuditSink(clock=now)
        ledger = BrokerLedger(repository)

        durable = local_durable_adapter_for_docker(runtime_config)
        # Share the whole-second clock so the driver and ledger agree on time.
        runtime = runtime_driver or DockerOpenHandsRuntimeDriver(
            runtime_config, clock=now
        )
        policy = BrokerRpcPolicy(
            _POLICY_PROFILE,
            _RUNTIME_DRIVER,
            _DURABLE_DRIVER,
            settings.broker_execution_lease_seconds,
        )
        coordinator = BrokerSegmentCoordinator(
            ledger=ledger,
            policy=policy,
            runtime_driver=runtime,
            secret_authority=secret_authority,
            durable_state_adapter=durable,
            receipt_verifier=receipt_verifier,
            clock=now,
        )
        application = BrokerRpcApplication(
            ledger=ledger,
            identity_verifier=identity_verifier,
            capability_authority=capability,
            audit_sink=audit_sink,
            policy=policy,
            segment_coordinator=coordinator,
        )

        # --- server (not yet bound) + client ---------------------------------
        socket_path = Path(settings.broker_control_socket_dir) / "control.sock"
        server = BrokerRpcServer(
            application=application,
            socket_policy=UnixSocketPolicy(socket_path, mode=0o600),
            peer_provider=_peer_credential_provider(),
            limits=BrokerRpcLimits(),
        )
        owner = BrokerOwner(_OWNER_TENANT, _OWNER_SUBJECT)
        worker_instance_id: UUID = uuid4()
        assignment_id: UUID = uuid4()

        def identity_provider(intent: WorkloadIntent) -> WorkloadIdentityToken:
            return identity_issuer.issue(
                owner=owner,
                worker_instance_id=worker_instance_id,
                assignment_id=assignment_id,
                isolation_mode=IsolationMode.DEDICATED,
                required_isolation=IsolationMode.SHARED,
                intents={intent},
            )

        client = BrokerRpcClient(path=socket_path, identity_provider=identity_provider)
    except Exception as exc:  # noqa: BLE001 — boot gate: never crash app startup
        log.error("broker DISABLED: %s", exc)
        return None

    # Bind the socket last; a bind failure is still fail-closed and, because the
    # stop callback is only registered on success, cannot leak a listener.
    try:
        await server.start()
    except Exception as exc:  # noqa: BLE001 — boot gate: never crash app startup
        log.error("broker DISABLED: control socket bind failed: %s", exc)
        return None
    stack.push_async_callback(server.stop)

    log.info(
        "broker composed (repo=%s, runtime=%s, socket=%s)",
        type(repository).__name__,
        type(runtime).__name__,
        socket_path,
    )
    return BrokerClientHandle(
        client=client,
        owner=owner,
        receipt_issuer=receipt_issuer,
        receipt_verifier=receipt_verifier,
        loop=asyncio.get_running_loop(),
        reconciler=None,
    )
