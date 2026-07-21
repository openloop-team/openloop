"""build_broker stands up a working co-process broker over a real UDS.

Hermetic: an in-memory broker repository and an injected in-memory runtime
driver, so no Postgres or Docker is needed. Exercises the plan's phase-2 gate —
a real BrokerRpcClient.create_job round-trip over the bound Unix socket, the
receipt private key kept out of the broker graph, and clean teardown.
"""

import base64
import logging
import os
import shutil
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from openloop.broker_control.local_receipts import LocalCheckpointReceiptStore
from openloop.broker_control.receipts import CheckpointReceiptIssuer
from openloop.broker_runtime.memory import InMemoryRuntimeDriver
from openloop.config import Settings
from openloop.tools.openhands_artifacts import WorkspaceArtifactStore
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout
from openloop.wiring.broker import (
    BrokerClientHandle,
    _derive_receipt_key,
    build_broker,
    build_broker_client,
    build_broker_service,
)

_DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop_agents",
)


async def _postgres_reachable() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(_DSN, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
def sock_dir():
    # A short base dir keeps the control.sock path under the ~100-byte sun_path
    # limit (pytest's tmp_path is too deep for a Unix socket).
    directory = Path(tempfile.mkdtemp(prefix="olbrk-", dir="/private/tmp"))
    try:
        directory.chmod(0o700)
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _root(seed: int) -> str:
    # A distinct 32-byte base64 root per domain (reused roots are rejected).
    return base64.b64encode(bytes([seed]) * 32).decode()


def _settings(tmp_path, sock_dir, **overrides):
    state_root = tmp_path / "state"
    runtime_root = tmp_path / "runtime"
    for path in (state_root, runtime_root):
        path.mkdir()
        # The durable state root must be a private 0o700 directory owned by this
        # user (LocalDurableStateAdapter enforces it); apply it to both so the
        # operator constraint is exercised, not accidentally satisfied.
        path.chmod(0o700)
    base = dict(
        coding_worker_openhands_broker_enabled=True,
        broker_control_socket_dir=str(sock_dir),
        broker_state_root=str(state_root),
        broker_runtime_root=str(runtime_root),
        broker_capability_roots={"cap-key-v1": _root(1)},
        broker_runtime_roots={"runtime-key-v1": _root(2)},
        broker_receipt_roots={"receipt-key-v1": _root(3)},
        # InMemoryRuntimeDriver's maximum lifetime bounds the lease; 300 matches
        # the proven broker RPC fixture.
        broker_execution_lease_seconds=300,
    )
    base.update(overrides)
    return Settings(**base)


async def test_build_broker_create_job_round_trip(tmp_path, sock_dir):
    settings = _settings(tmp_path, sock_dir)
    async with AsyncExitStack() as stack:
        handle = await build_broker(
            settings, stack, runtime_driver=InMemoryRuntimeDriver()
        )
        assert isinstance(handle, BrokerClientHandle)

        created = await handle.client.create_job("wiring-create-key-0001")
        assert created.ticket.job_id is not None

        # Idempotent replay returns the same job and capability over the wire.
        replay = await handle.client.create_job("wiring-create-key-0001")
        assert replay.ticket.replayed is True
        assert replay.ticket.job_id == created.ticket.job_id
        assert replay.capability == created.capability

        # The client can inspect the job with the capability it was handed.
        inspected = await handle.client.inspect_job(
            created.ticket.job_id, created.capability
        )
        assert inspected.snapshot.job_id == created.ticket.job_id

        # The receipt PRIVATE key lives on the checkpoint-store side (the handle),
        # never in the broker graph.
        assert isinstance(handle.receipt_issuer, CheckpointReceiptIssuer)
        assert handle.workspace_ingress is not None

    # Teardown unlinked the socket.
    assert not (sock_dir / "control.sock").exists()


async def test_build_broker_returns_none_when_flag_off(tmp_path, sock_dir):
    settings = _settings(
        tmp_path, sock_dir, coding_worker_openhands_broker_enabled=False
    )
    async with AsyncExitStack() as stack:
        assert await build_broker(settings, stack) is None


async def test_build_broker_fails_closed_on_reused_root(tmp_path, sock_dir):
    # Same bytes under two domains = a shared trust line; build must refuse.
    settings = _settings(
        tmp_path, sock_dir, broker_runtime_roots={"runtime-key-v1": _root(1)}
    )
    async with AsyncExitStack() as stack:
        assert (
            await build_broker(
                settings, stack, runtime_driver=InMemoryRuntimeDriver()
            )
            is None
        )


async def test_build_broker_fails_closed_on_missing_root(tmp_path, sock_dir):
    settings = _settings(tmp_path, sock_dir, broker_capability_roots={})
    async with AsyncExitStack() as stack:
        assert (
            await build_broker(
                settings, stack, runtime_driver=InMemoryRuntimeDriver()
            )
            is None
        )


async def test_generation_deadline_caps_the_real_runtime(tmp_path, sock_dir):
    # With the real Docker runtime driver (no injection) the driver's maximum
    # lifetime comes from broker_generation_deadline_seconds. A lease within the
    # deadline builds; a lease longer than the deadline is rejected by the
    # coordinator — proving the deadline is wired, not silently the 86400 default.
    within = _settings(
        tmp_path,
        sock_dir,
        broker_execution_lease_seconds=300,
        broker_generation_deadline_seconds=900,
    )
    async with AsyncExitStack() as stack:
        assert await build_broker(within, stack) is not None


async def test_lease_longer_than_generation_deadline_fails_closed(tmp_path, sock_dir):
    over = _settings(
        tmp_path,
        sock_dir,
        broker_execution_lease_seconds=1000,
        broker_generation_deadline_seconds=500,
    )
    async with AsyncExitStack() as stack:
        assert await build_broker(over, stack) is None


async def test_bad_permission_state_root_fails_closed(tmp_path, sock_dir):
    # The durable adapter rejects a non-0700 state root; that failure happens
    # after the initial config decode, so it exercises the widened fail-closed
    # envelope — it must return None, not escape and crash startup.
    loose = tmp_path / "loose-state"
    loose.mkdir()
    loose.chmod(0o755)
    settings = _settings(tmp_path, sock_dir, broker_state_root=str(loose))
    async with AsyncExitStack() as stack:
        assert (
            await build_broker(
                settings, stack, runtime_driver=InMemoryRuntimeDriver()
            )
            is None
        )


@pytest.mark.postgres
async def test_build_broker_uses_durable_audit_with_postgres(tmp_path, sock_dir):
    # A Postgres-backed broker must pair with the durable RPC audit sink; if the
    # audit table were missing or set up before the migrations, setup would raise
    # and build_broker would fail closed. A returned handle + a create_job that
    # round-trips over the pool proves the durable repo + durable audit path.
    if not await _postgres_reachable():
        pytest.skip(f"no Postgres reachable at {_DSN}")
    import asyncpg

    settings = _settings(tmp_path, sock_dir)
    pool = await asyncpg.create_pool(_DSN, min_size=1, max_size=2)
    try:
        async with AsyncExitStack() as stack:
            handle = await build_broker(
                settings, stack, pool=pool, runtime_driver=InMemoryRuntimeDriver()
            )
            assert isinstance(handle, BrokerClientHandle)
            created = await handle.client.create_job(
                f"wiring-pg-{os.getpid()}-{tmp_path.name}"[:120]
            )
            assert created.ticket.job_id is not None
    finally:
        await pool.close()


# --- external mode -------------------------------------------------------
# The config-distributed key seam: the app signs identity tokens with a private
# seed the broker only holds the public of, and derives receipt keys from roots
# whose public halves the broker verifies with. These build both halves in one
# process from ONE fully-populated Settings (two processes' config combined),
# proving the seam without a subprocess.

_EXTERNAL_SEED = bytes(range(1, 33))
_EXTERNAL_RECEIPT_ROOT = bytes([4]) * 32


def _external_settings(tmp_path, sock_dir, **overrides):
    identity_private = Ed25519PrivateKey.from_private_bytes(_EXTERNAL_SEED)
    identity_pub = identity_private.public_key().public_bytes_raw()
    receipt_pub = _derive_receipt_key(
        _EXTERNAL_RECEIPT_ROOT, "broker-receipt", "receipt-key-v1"
    ).public_key().public_bytes_raw()
    ingress_root = tmp_path / "ingress"
    receipts_root = tmp_path / "receipts"
    for path in (ingress_root, receipts_root):
        path.mkdir(exist_ok=True)
        path.chmod(0o700)
    # The shared ingress root is provisioned group-owned by the shared gid so the
    # broker's materialize-side ownership validation accepts it.
    os.chown(ingress_root, -1, os.getgid())
    external = dict(
        broker_mode="external",
        broker_identity_private_key=SecretStr(
            base64.b64encode(_EXTERNAL_SEED).decode()
        ),
        broker_identity_public_keys={"identity-v1": base64.b64encode(identity_pub).decode()},
        broker_receipt_roots={
            "receipt-key-v1": base64.b64encode(_EXTERNAL_RECEIPT_ROOT).decode()
        },
        broker_receipt_public_keys={
            "receipt-key-v1": base64.b64encode(receipt_pub).decode()
        },
        broker_ingress_root=str(ingress_root),
        broker_checkpoint_receipt_root=str(receipts_root),
        broker_shared_data_gid=os.getgid(),
        broker_expected_app_uid=os.getuid(),
    )
    external.update(overrides)
    return _settings(tmp_path, sock_dir, **external)


async def test_external_client_round_trips_against_built_service(tmp_path, sock_dir):
    settings = _external_settings(tmp_path, sock_dir)
    async with AsyncExitStack() as stack:
        # Build + bind the service half separately (the openloop-broker process
        # in production); the caller registers stop after a successful bind.
        service = await build_broker_service(
            settings, stack, runtime_driver=InMemoryRuntimeDriver()
        )
        await service.bind()
        stack.push_async_callback(service.server.stop)

        # The app-side dispatcher composes ONLY the client in external mode.
        handle = await build_broker(settings, stack)
        assert isinstance(handle, BrokerClientHandle)
        # The broker owns lifecycle recovery in external mode.
        assert handle._ledger is None
        assert handle._coordinator is None
        assert handle.shared_data_gid == os.getgid()
        assert handle.receipt_root == tmp_path / "receipts"

        created = await handle.client.create_job("wiring-external-key-0001")
        assert created.ticket.job_id is not None
        replay = await handle.client.create_job("wiring-external-key-0001")
        assert replay.ticket.replayed is True
        assert replay.ticket.job_id == created.ticket.job_id

    assert not (sock_dir / "control.sock").exists()


@pytest.mark.parametrize(
    "missing_field",
    [
        "broker_identity_private_key",
        "broker_ingress_root",
        "broker_checkpoint_receipt_root",
        "broker_shared_data_gid",
    ],
)
async def test_external_missing_required_client_field_fails_closed(
    tmp_path, sock_dir, caplog, missing_field
):
    settings = _external_settings(tmp_path, sock_dir, **{missing_field: None})
    async with AsyncExitStack() as stack:
        with caplog.at_level(logging.ERROR, logger="openloop"):
            handle = await build_broker(settings, stack)
    assert handle is None
    assert any(missing_field in record.message for record in caplog.records), (
        f"expected a DISABLED log naming {missing_field}: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


async def test_external_reused_receipt_root_fails_closed(tmp_path, sock_dir, caplog):
    reused = base64.b64encode(_EXTERNAL_RECEIPT_ROOT).decode()
    settings = _external_settings(
        tmp_path,
        sock_dir,
        broker_receipt_roots={
            "receipt-key-v1": reused,
            "receipt-key-v2": reused,
        },
        broker_receipt_current_version="receipt-key-v2",
    )

    async with AsyncExitStack() as stack:
        with caplog.at_level(logging.ERROR, logger="openloop"):
            handle = await build_broker(settings, stack)

    assert handle is None
    assert any("fake rotation" in record.message for record in caplog.records)


async def test_external_bind_checkpoint_store_leaves_reconciler_none(
    tmp_path, sock_dir
):
    settings = _external_settings(tmp_path, sock_dir)
    async with AsyncExitStack() as stack:
        handle = await build_broker_client(settings, stack)
    assert handle._ledger is None
    assert handle._coordinator is None

    artifacts = WorkspaceArtifactStore(
        OpenHandsStateLayout(tmp_path / "artifact-state"),
        OpenHandsKeyDeriver(bytes(range(32)), master_key_id="artifact-v1"),
    )
    store = handle.bind_checkpoint_store(artifacts)
    assert isinstance(store, LocalCheckpointReceiptStore)
    # External mode has no app-side ledger/coordinator, so no reconciler is wired.
    assert handle.reconciler is None


async def test_coprocess_client_failure_returns_none_and_no_listener(
    tmp_path, sock_dir, monkeypatch
):
    # Force the client half to fail AFTER the service is constructed. Bind is the
    # last fallible act and happens only after client construction, so the socket
    # must never be bound: build_broker returns None and no inode exists.
    settings = _settings(tmp_path, sock_dir)
    import openloop.wiring.broker as broker_mod

    async def _boom(*args, **kwargs):
        # Simulate a bad broker_receipt_roots decode surfacing in the client half.
        raise ValueError("broker receipt root 'receipt-key-v1' is not valid base64")

    monkeypatch.setattr(broker_mod, "build_broker_client", _boom)
    async with AsyncExitStack() as stack:
        handle = await build_broker(
            settings, stack, runtime_driver=InMemoryRuntimeDriver()
        )
    assert handle is None
    assert not (sock_dir / "control.sock").exists()
