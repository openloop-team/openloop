"""Broker composition — assemble the graph and own the UDS RPC seam.

`build_broker` is the app-side dispatcher. Behind
``coding_worker_openhands_broker_enabled`` it composes the broker for the
configured ``broker_mode`` and returns a ready :class:`BrokerClientHandle` (plus
the resources the coding-worker adapter needs), or ``None`` fail-closed:

- ``coprocess`` (default, unchanged): the whole broker graph lives in this
  process. The dispatcher generates the ephemeral identity keypair, splits it
  into the client's issuer (private) and the service's verifier (public), builds
  the service (unbound) and the client against its socket, then binds the socket
  **last** so a client-construction failure never leaves a listener bound.
- ``external``: only the client half is composed here — it talks to a separate
  ``openloop-broker`` process over the same UDS. The service half
  (:func:`build_broker_service`) is constructed **fail-fast** by that
  entrypoint, not by this dispatcher; in external mode ``_ledger``/
  ``_coordinator`` stay ``None`` (the broker owns recovery).

Both halves are fail-closed at the dispatcher: any missing or invalid setting
logs a specific error and returns ``None`` so the caller disables the coding
worker loudly rather than falling back to the direct in-process launch path.

Decision-locked shape (see
``docs/superpowers/specs/2026-07-19-broker-app-integration-design.md`` and the
process-split spec):

- **Real UDS only** — the client reaches the graph solely over the
  ``BrokerRpcServer`` Unix socket; there is no in-process shortcut into the
  application/coordinator, coprocess or external.
- **Separated versioned key rings, one per trust domain.** Capability, runtime,
  and receipt roots are independent version→secret maps (rotatable); a root
  reused within or across domains is rejected (fake rotation / shared trust
  line). Across the process boundary (external mode) the broker holds only
  capability/runtime roots plus the identity/receipt *publics*, and
  ``_reject_cross_boundary_reuse`` rejects a held root that reproduces a trusted
  public.
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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import SecretStr

from openloop.broker.ledger import BrokerLedger
from openloop.broker.memory import InMemoryBrokerRepository
from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker.postgres import PostgresBrokerRepository
from openloop.broker_control.coordinator import BrokerSegmentCoordinator
from openloop.broker_control.development import local_durable_adapter_for_docker
from openloop.broker_control.local_receipts import LocalCheckpointReceiptStore
from openloop.broker_control.recovery import BrokerLifecycleReconciler
from openloop.broker_control.receipts import (
    CheckpointReceiptIssuer,
    CheckpointReceiptVerifier,
)
from openloop.broker_control.secrets import (
    RuntimeSecretAuthority,
    RuntimeSecretRootRing,
)
from openloop.broker_control.workspace_ingress import LocalWorkspaceIngress
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
# The checkpoint store is the receipt signer; the broker only ever verifies.
_RECEIPT_ISSUER = "checkpoint-store"
# BrokerRpcPolicy tokens — the profile, runtime driver, and durable-state driver
# this slice pins (OpenHands profile × Docker driver × local durable state).
_POLICY_PROFILE = "default"
_RUNTIME_DRIVER = "docker"
_DURABLE_DRIVER = "local"
# START_SEGMENT performs the bounded Docker probe, creation, and 15-second
# readiness gate inside one authenticated RPC. The transport's generic 5-second
# application limit cannot represent that operation, so the co-process profile
# pins a longer but still finite envelope across server, client, and sync bridge.
BROKER_RPC_APPLICATION_TIMEOUT_SECONDS = 120.0
BROKER_RPC_TOTAL_TIMEOUT_SECONDS = 130.0


@dataclass(slots=True)
class BrokerClientHandle:
    """A live broker client plus the checkpoint-store-side signing material.

    ``receipt_issuer`` (current-version PRIVATE key) is deliberately kept out of
    the broker graph — it belongs to the checkpoint store the worker adapter
    signs with. ``reconciler`` is filled by :meth:`bind_checkpoint_store` when a
    local checkpoint store (its receipt locator) is wired, but only in coprocess
    mode: in external mode the broker owns lifecycle recovery, so ``_ledger``/
    ``_coordinator`` stay ``None`` and no reconciler is built. ``loop`` is the app
    event loop the broker client runs on, captured so the synchronous
    coding-worker thread can bridge into the async client via
    ``run_coroutine_threadsafe``.

    ``shared_data_gid``/``receipt_root`` describe the dedicated cross-boundary
    receipts tree (external mode); they are carried here so the phase-E pass can
    hand them to :class:`LocalCheckpointReceiptStore` without another settings
    read. In coprocess they stay ``None`` (nothing crosses a process boundary).
    """

    client: BrokerRpcClient
    owner: BrokerOwner
    receipt_issuer: CheckpointReceiptIssuer
    receipt_verifier: CheckpointReceiptVerifier
    loop: asyncio.AbstractEventLoop
    workspace_ingress: LocalWorkspaceIngress
    _ledger: BrokerLedger | None = field(default=None, repr=False)
    _coordinator: BrokerSegmentCoordinator | None = field(default=None, repr=False)
    shared_data_gid: int | None = None
    receipt_root: Path | None = None
    checkpoint_store: LocalCheckpointReceiptStore | None = None
    reconciler: Any | None = None

    def bind_checkpoint_store(self, artifact_store: Any) -> LocalCheckpointReceiptStore:
        """Bind the worker-side artifact store once and finish recovery wiring."""
        if self.checkpoint_store is not None:
            if getattr(self.checkpoint_store, "_artifacts", None) is not artifact_store:
                raise ValueError("broker checkpoint store is already bound")
            return self.checkpoint_store
        # expected_uid/expected_gid validate the *in-artifact* tree, which stays
        # owned by the app's primary gid at 0700; shared_data_gid governs only the
        # dedicated cross-boundary receipts tree (wired in a later phase).
        checkpoint_store = LocalCheckpointReceiptStore(
            artifact_store=artifact_store,
            issuer=self.receipt_issuer,
            historical_verifier=self.receipt_verifier,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )
        self.checkpoint_store = checkpoint_store
        # External mode: the broker process owns lifecycle recovery, so the app
        # holds no ledger/coordinator to reconcile against — leave reconciler None.
        if self._ledger is not None and self._coordinator is not None:
            self.reconciler = BrokerLifecycleReconciler(
                ledger=self._ledger,
                coordinator=self._coordinator,
                receipt_locator=checkpoint_store,
                receipt_verifier=self.receipt_verifier,
            )
        return checkpoint_store


@dataclass(slots=True)
class BrokerServiceHandle:
    """The unbound broker service graph plus its bindable listener.

    Everything the broker process needs is constructed but the control socket is
    **not yet bound**: :meth:`bind` is the last fallible act and the caller
    registers ``server.stop`` on its stack only after a successful bind, which
    keeps the no-leaked-listener property.
    """

    application: BrokerRpcApplication
    server: BrokerRpcServer
    ledger: BrokerLedger
    coordinator: BrokerSegmentCoordinator
    runtime: RuntimeDriver
    workspace_ingress: LocalWorkspaceIngress
    receipt_verifier: CheckpointReceiptVerifier
    repository: Any
    audit_sink: Any

    async def bind(self) -> None:
        """Bind the control socket; the caller registers ``server.stop`` after."""
        await self.server.start()


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
    _reject_reused_roots(decoded)
    return decoded


def _decode_public_keys(
    name: str, keys: dict[str, str]
) -> dict[str, Ed25519PublicKey]:
    """Decode a version→base64-32-byte Ed25519 public map (external mode).

    The broker holds only the PUBLIC halves of the app's identity and receipt
    keys; this turns the config surface into usable verification keys, rejecting
    an empty map or malformed/short encodings the same way roots are.
    """
    if not keys:
        raise ValueError(f"broker {name} public keys must be set in external mode")
    decoded: dict[str, Ed25519PublicKey] = {}
    for version, encoded in keys.items():
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"broker {name} public key {version!r} is not valid base64"
            ) from exc
        if len(raw) != 32:
            raise ValueError(
                f"broker {name} public key {version!r} must decode to 32 bytes"
            )
        try:
            decoded[version] = Ed25519PublicKey.from_public_bytes(raw)
        except Exception as exc:  # noqa: BLE001 - normalize crypto errors to config
            raise ValueError(
                f"broker {name} public key {version!r} is not a valid Ed25519 key"
            ) from exc
    return decoded


def _decode_identity_seed(secret: SecretStr) -> Ed25519PrivateKey:
    """Decode a base64 32-byte Ed25519 seed into the app's identity signer."""
    try:
        raw = base64.b64decode(secret.get_secret_value(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("broker_identity_private_key is not valid base64") from exc
    if len(raw) != 32:
        raise ValueError("broker_identity_private_key must decode to 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(raw)


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


def _reject_cross_boundary_reuse(
    held: dict[str, dict[str, bytes]],
    receipt_publics: dict[str, Ed25519PublicKey],
    identity_publics: dict[str, Ed25519PublicKey],
    *,
    receipt_domain: str,
) -> None:
    """Decision 11: a held broker root must not reproduce a trusted public.

    In external mode the broker holds capability/runtime *roots* and the app's
    identity/receipt *publics*. A held root that (directly or via the receipt
    HKDF) equals a configured public would be a hidden shared trust line across
    the process boundary. For every held root we check:

    - receipt reuse: the root HKDF-derived under ``receipt_domain``/version
      equals a configured receipt public;
    - identity reuse (direct): the root used verbatim as an Ed25519 seed — the
      realistic misuse, an operator pasting a root as the identity key — equals
      a configured identity public;
    - identity reuse (HKDF): the root HKDF-derived under the receipt domain
      (keyed by the identity key id) equals a configured identity public.

    Any match raises, naming the receipt domain.
    """
    receipt_raw = {v: p.public_bytes_raw() for v, p in receipt_publics.items()}
    identity_raw = {k: p.public_bytes_raw() for k, p in identity_publics.items()}
    message = (
        "broker root reuse across the process boundary "
        f"(receipt domain {receipt_domain!r})"
    )
    for roots in held.values():
        for root in roots.values():
            for version, pub_raw in receipt_raw.items():
                derived = _derive_receipt_key(root, receipt_domain, version)
                if derived.public_key().public_bytes_raw() == pub_raw:
                    raise ValueError(message)
            direct = Ed25519PrivateKey.from_private_bytes(root)
            direct_raw = direct.public_key().public_bytes_raw()
            for key_id, pub_raw in identity_raw.items():
                if direct_raw == pub_raw:
                    raise ValueError(message)
                derived = _derive_receipt_key(root, receipt_domain, key_id)
                if derived.public_key().public_bytes_raw() == pub_raw:
                    raise ValueError(message)


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


def _default_clock(clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
    """Whole-second UTC by default: generation deadlines derived from this clock
    flow into RunningGenerationAccess, which rejects any sub-second timestamp —
    a raw datetime.now(UTC) would make every start_segment fail INTERNAL."""
    return clock or (lambda: datetime.now(UTC).replace(microsecond=0))


async def build_broker_service(
    settings: Settings,
    stack: Any,
    *,
    pool: Any | None = None,
    runtime_driver: RuntimeDriver | None = None,
    clock: Callable[[], datetime] | None = None,
    identity_public_keys: dict[str, Ed25519PublicKey] | None = None,
    receipt_verifier: CheckpointReceiptVerifier | None = None,
) -> BrokerServiceHandle:
    """Assemble the broker service graph with an **unbound** listener.

    The identity verifier comes from ``identity_public_keys`` when injected
    (coprocess passes the ephemeral public) else from
    ``settings.broker_identity_public_keys`` (external). The receipt verifier is
    injected (coprocess) else built from ``settings.broker_receipt_public_keys``
    (external). External config additionally runs the decision-11 cross-boundary
    reuse gate. **Raises** on any invalid config — the entrypoint is the
    fail-fast caller; the coprocess dispatcher wraps this in its fail-closed try.
    """
    now = _default_clock(clock)
    for field_name in (
        "broker_control_socket_dir",
        "broker_state_root",
        "broker_runtime_root",
    ):
        if not getattr(settings, field_name):
            raise ValueError(f"{field_name} is required when the broker is enabled")

    external = identity_public_keys is None or receipt_verifier is None
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
    # Within-broker-domain reuse (cap vs runtime) is rejected in every mode; the
    # coprocess dispatcher additionally checks receipt roots (which live here in
    # one process), and external adds the cross-boundary gate below.
    _reject_reused_roots(capability_roots, runtime_roots)

    # --- verification keys (publics only) --------------------------------
    if identity_public_keys is None:
        identity_public_keys = _decode_public_keys(
            "identity", settings.broker_identity_public_keys
        )
    receipt_publics: dict[str, Ed25519PublicKey] | None = None
    if receipt_verifier is None:
        receipt_publics = _decode_public_keys(
            "receipt", settings.broker_receipt_public_keys
        )
        receipt_verifier = CheckpointReceiptVerifier(
            public_keys=VerificationKeySet(receipt_publics),
            issuer=_RECEIPT_ISSUER,
        )
    if external:
        # The broker holds capability/runtime roots; neither may reproduce an
        # identity/receipt public it trusts the app to sign with (decision 11).
        if receipt_publics is None:
            receipt_publics = _decode_public_keys(
                "receipt", settings.broker_receipt_public_keys
            )
        _reject_cross_boundary_reuse(
            {"capability": capability_roots, "runtime": runtime_roots},
            receipt_publics,
            identity_public_keys,
            receipt_domain=settings.broker_receipt_domain,
        )

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
    identity_verifier = WorkloadIdentityVerifier(
        public_keys=identity_public_keys,
        issuer=settings.broker_identity_issuer,
        audience=settings.broker_identity_audience,
        clock=now,
    )

    # --- ledger / durable audit / runtime / coordinator ------------------
    if pool is not None:
        repository: Any = PostgresBrokerRepository()
        await repository.setup(pool)
        # Durable broker state demands a durable RPC audit trail: the in-memory
        # sink would drop authenticated security decisions on restart while their
        # effects survive. The broker migrations (run by the repository setup
        # above) own the broker_rpc_audit table.
        audit_sink: Any = PostgresRpcAuditSink()
        await audit_sink.setup(pool)
    else:
        repository = InMemoryBrokerRepository(clock=now)
        audit_sink = InMemoryRpcAuditSink(clock=now)
    ledger = BrokerLedger(repository)

    durable = local_durable_adapter_for_docker(runtime_config)
    if external:
        # External mode: the app stages across a uid boundary into the required
        # sibling root; the broker validates the app's ownership and keeps its
        # consumed/discarded markers in a broker-private sibling tree.
        if not settings.broker_ingress_root:
            raise ValueError("broker_ingress_root is required in external broker mode")
        if settings.broker_expected_app_uid is None:
            raise ValueError(
                "broker_expected_app_uid is required in external broker mode"
            )
        workspace_ingress = LocalWorkspaceIngress(
            Path(settings.broker_ingress_root),
            expected_stage_uid=settings.broker_expected_app_uid,
            shared_gid=settings.broker_shared_data_gid,
            marker_root=Path(settings.broker_runtime_root) / ".ingress-markers",
        )
    else:
        # Co-process: one shared instance, unchanged owner-only construction.
        workspace_ingress = LocalWorkspaceIngress(
            runtime_config.runtime_root / ".workspace-ingress"
        )
    # Share the whole-second clock so the driver and ledger agree on time.
    runtime = runtime_driver or DockerOpenHandsRuntimeDriver(
        runtime_config,
        clock=now,
        workspace_materializer=workspace_ingress,
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

    # --- server (not yet bound) ------------------------------------------
    socket_path = Path(settings.broker_control_socket_dir) / "control.sock"
    # A group-readable socket + shared gid lets a separate broker container's
    # relay reach the control UDS; owner-only otherwise (unchanged coprocess).
    server = BrokerRpcServer(
        application=application,
        socket_policy=UnixSocketPolicy(
            socket_path,
            mode=0o660 if settings.broker_shared_data_gid is not None else 0o600,
            gid=settings.broker_shared_data_gid,
        ),
        peer_provider=_peer_credential_provider(),
        limits=BrokerRpcLimits(
            application_timeout_seconds=BROKER_RPC_APPLICATION_TIMEOUT_SECONDS,
            total_timeout_seconds=BROKER_RPC_TOTAL_TIMEOUT_SECONDS,
        ),
    )
    return BrokerServiceHandle(
        application=application,
        server=server,
        ledger=ledger,
        coordinator=coordinator,
        runtime=runtime,
        workspace_ingress=workspace_ingress,
        receipt_verifier=receipt_verifier,
        repository=repository,
        audit_sink=audit_sink,
    )


async def build_broker_client(
    settings: Settings,
    stack: Any,
    *,
    clock: Callable[[], datetime] | None = None,
    identity_private_key: Ed25519PrivateKey | None = None,
    identity_key_id: str | None = None,
    workspace_ingress: LocalWorkspaceIngress | None = None,
) -> BrokerClientHandle:
    """Assemble the app-side broker client (issuer + RPC client + handle).

    The identity issuer signs with the injected ephemeral private key (coprocess)
    or the decoded ``settings.broker_identity_private_key`` (external). Receipt
    issuer+verifier are derived app-side from ``broker_receipt_roots`` (unchanged
    derivation). The stage-side ingress is the injected instance (coprocess — one
    shared instance with the service, so their per-job lock maps stay unified) or
    a plain handle on ``broker_ingress_root`` (external). **External mode requires
    all four** of ``broker_identity_private_key``, ``broker_ingress_root``,
    ``broker_checkpoint_receipt_root``, ``broker_shared_data_gid``; any missing
    raises. **Raises** on any invalid config; the dispatcher's fail-closed wrapper
    turns it into ``None`` + a specific log.
    """
    now = _default_clock(clock)
    external = settings.broker_mode == "external"
    if not settings.broker_control_socket_dir:
        raise ValueError(
            "broker_control_socket_dir is required when the broker is enabled"
        )
    if external:
        missing = [
            name
            for name, value in (
                ("broker_identity_private_key", settings.broker_identity_private_key),
                ("broker_ingress_root", settings.broker_ingress_root),
                (
                    "broker_checkpoint_receipt_root",
                    settings.broker_checkpoint_receipt_root,
                ),
                ("broker_shared_data_gid", settings.broker_shared_data_gid),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"external broker mode requires {', '.join(missing)} to be set"
            )

    # --- identity issuer (client side holds the PRIVATE key) -------------
    if identity_private_key is None:
        identity_private_key = _decode_identity_seed(
            settings.broker_identity_private_key
        )
    if identity_key_id is None:
        identity_key_id = settings.broker_identity_key_id
    identity_issuer = WorkloadIdentityIssuer(
        private_key=identity_private_key,
        key_id=identity_key_id,
        issuer=settings.broker_identity_issuer,
        audience=settings.broker_identity_audience,
        clock=now,
    )

    # --- receipt issuer (current PRIVATE) + verifier (all PUBLIC) --------
    receipt_roots = _decode_roots(
        "receipt",
        settings.broker_receipt_roots,
        settings.broker_receipt_current_version,
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

    # --- stage-side workspace ingress ------------------------------------
    # Injected (co-process) → the service's shared instance, unchanged. Not
    # injected (external) → a stage-side handle on the required sibling root that
    # writes group-shared modes for the broker to read across the boundary.
    if workspace_ingress is None:
        ingress_root = (
            Path(settings.broker_ingress_root)
            if settings.broker_ingress_root
            else Path(settings.broker_runtime_root) / ".workspace-ingress"
        )
        workspace_ingress = LocalWorkspaceIngress(
            ingress_root,
            shared_gid=settings.broker_shared_data_gid,
        )

    # --- RPC client against the control socket ---------------------------
    socket_path = Path(settings.broker_control_socket_dir) / "control.sock"
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

    client = BrokerRpcClient(
        path=socket_path,
        identity_provider=identity_provider,
        io_timeout_seconds=BROKER_RPC_APPLICATION_TIMEOUT_SECONDS + 5.0,
        total_timeout_seconds=BROKER_RPC_TOTAL_TIMEOUT_SECONDS,
    )
    receipt_root = (
        Path(settings.broker_checkpoint_receipt_root)
        if settings.broker_checkpoint_receipt_root
        else None
    )
    return BrokerClientHandle(
        client=client,
        owner=owner,
        receipt_issuer=receipt_issuer,
        receipt_verifier=receipt_verifier,
        loop=asyncio.get_running_loop(),
        workspace_ingress=workspace_ingress,
        shared_data_gid=settings.broker_shared_data_gid,
        receipt_root=receipt_root,
    )


async def build_broker(
    settings: Settings,
    stack: Any,
    *,
    pool: Any | None = None,
    runtime_driver: RuntimeDriver | None = None,
    clock: Callable[[], datetime] | None = None,
) -> BrokerClientHandle | None:
    """Compose the broker for the configured mode behind the flag; fail-closed.

    Returns ``None`` (with a specific log) on any missing/invalid setting so the
    caller disables the coding worker loudly. In coprocess mode all owned
    resources register their teardown on ``stack`` and the socket binds last, so
    a returned ``None`` never leaves a listener bound.
    """
    if not settings.coding_worker_openhands_broker_enabled:
        return None

    if settings.broker_mode == "external":
        # Only the client half lives here; the broker service is a separate,
        # fail-fast process. No ledger/coordinator app-side (broker owns recovery).
        try:
            return await build_broker_client(settings, stack, clock=clock)
        except Exception as exc:  # noqa: BLE001 — boot gate: never crash app startup
            log.error("broker DISABLED: %s", exc)
            return None

    # --- coprocess: the whole graph in this process ----------------------
    # Everything below is fail-closed: any construction or setup failure logs a
    # specific reason and returns None so the caller disables the coding worker
    # loudly, never crashing app startup. The socket binds last (second try), so
    # a returned None never leaves a listener bound.
    try:
        # The full within-process reuse check across all three root rings — the
        # single-process invariant the split otherwise scatters. build_broker_service
        # re-decodes capability/runtime for authority construction; this decode
        # only feeds the cross-domain reuse gate + the injected receipt verifier.
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

        # Ephemeral identity keypair: private -> issuer (client), public ->
        # verifier (service). Nothing at rest; tokens live <=300s in-process.
        identity_key = Ed25519PrivateKey.generate()
        identity_key_id = settings.broker_identity_key_id

        # The service verifies receipts with PUBLIC keys only; derive them here
        # from the shared roots and inject the verifier (decision-2 receipt split).
        receipt_keys = {
            version: _derive_receipt_key(
                root, settings.broker_receipt_domain, version
            )
            for version, root in receipt_roots.items()
        }
        receipt_verifier = CheckpointReceiptVerifier(
            public_keys=VerificationKeySet(
                {version: key.public_key() for version, key in receipt_keys.items()}
            ),
            issuer=_RECEIPT_ISSUER,
        )

        service = await build_broker_service(
            settings,
            stack,
            pool=pool,
            runtime_driver=runtime_driver,
            clock=clock,
            identity_public_keys={identity_key_id: identity_key.public_key()},
            receipt_verifier=receipt_verifier,
        )
        # One shared ingress instance: two instances on the same root would hold
        # independent per-job lock maps, silently breaking today's stage/materialize
        # serialization in-process. Inject the service's instance into the client.
        client = await build_broker_client(
            settings,
            stack,
            clock=clock,
            identity_private_key=identity_key,
            identity_key_id=identity_key_id,
            workspace_ingress=service.workspace_ingress,
        )
    except Exception as exc:  # noqa: BLE001 — boot gate: never crash app startup
        log.error("broker DISABLED: %s", exc)
        return None

    # Bind the socket last; a bind failure is still fail-closed and, because the
    # stop callback is only registered on success, cannot leak a listener. Bind
    # is the last fallible act — a client-construction failure above returns None
    # with no listener ever bound.
    try:
        await service.bind()
    except Exception as exc:  # noqa: BLE001 — boot gate: never crash app startup
        log.error("broker DISABLED: control socket bind failed: %s", exc)
        return None
    stack.push_async_callback(service.server.stop)

    log.info(
        "broker composed (repo=%s, runtime=%s, socket=%s)",
        type(service.repository).__name__,
        type(service.runtime).__name__,
        service.server.path,
    )
    client._ledger = service.ledger
    client._coordinator = service.coordinator
    return client
