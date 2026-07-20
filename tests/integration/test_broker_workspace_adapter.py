"""BrokerWorkspaceAdapter forward path (phase 3a).

`create` runs against a real in-memory co-process broker (real create_job +
start_segment, real generation semantics) with only the SDK workspace
construction faked. `stream_git_delta`/`attach_conversation`/`probe` are covered
with fakes so no OpenHands SDK, Docker, or relay is needed.
"""

import asyncio
import base64
import shutil
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from openloop.broker.models import VerifiedCheckpointReceipt
from openloop.broker_runtime.memory import InMemoryRuntimeDriver
from openloop.config import Settings
from openloop.tools.openhands_broker_workspace import (
    BrokerWorkspaceAdapter,
    BrokerWorkspaceError,
)
from openloop.tools.openhands_docker import ArchiveStreamResult
from openloop.tools.openhands_relay_client import RelayClientEndpoint, RelayMode
from openloop.wiring.broker import build_broker


@pytest.fixture
def sock_dir():
    directory = Path(tempfile.mkdtemp(prefix="olbrk-", dir="/private/tmp"))
    try:
        directory.chmod(0o700)
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _root(seed: int) -> str:
    return base64.b64encode(bytes([seed]) * 32).decode()


def _settings(tmp_path, sock_dir, **overrides):
    state_root = tmp_path / "state"
    runtime_root = tmp_path / "runtime"
    for path in (state_root, runtime_root):
        path.mkdir()
        path.chmod(0o700)
    base = dict(
        coding_worker_openhands_broker_enabled=True,
        broker_control_socket_dir=str(sock_dir),
        broker_state_root=str(state_root),
        broker_runtime_root=str(runtime_root),
        broker_capability_roots={"cap-key-v1": _root(1)},
        broker_runtime_roots={"runtime-key-v1": _root(2)},
        broker_receipt_roots={"receipt-key-v1": _root(3)},
        broker_execution_lease_seconds=300,
    )
    base.update(overrides)
    return Settings(**base)


class _FakeWorkspace:
    def __init__(self, endpoint):
        self.endpoint = endpoint


async def _broker_adapter(stack, tmp_path, sock_dir, factory):
    handle = await build_broker(
        _settings(tmp_path, sock_dir),
        stack,
        runtime_driver=InMemoryRuntimeDriver(),
    )
    assert handle is not None
    return BrokerWorkspaceAdapter(
        client=handle.client,
        loop=handle.loop,
        receipt_issuer=handle.receipt_issuer,
        workspace_factory=factory,
    )


async def test_create_opens_running_generation_one(tmp_path, sock_dir):
    captured = []

    def factory(endpoint):
        captured.append(endpoint)
        return _FakeWorkspace(endpoint)

    async with AsyncExitStack() as stack:
        adapter = await _broker_adapter(stack, tmp_path, sock_dir, factory)
        # Sync method must be driven from a worker thread — calling it on the
        # loop thread would deadlock the run_coroutine_threadsafe bridge.
        target = await asyncio.to_thread(adapter.create, tmp_path, "job-abc123def4")

    assert isinstance(target, _FakeWorkspace)
    assert len(captured) == 1
    endpoint = captured[0]
    assert isinstance(endpoint, RelayClientEndpoint)
    assert endpoint.mode is RelayMode.RUNNING
    # A fresh job starts at expected generation 0 and runs generation 1.
    state = adapter._jobs["job-abc123def4"]
    assert state.current_generation == 0
    assert state.running_generation == 1
    assert endpoint.conversation_id is not None


def _sign_receipt(handle, adapter, job_id, *, generation, barrier, suffix):
    state = adapter._jobs[job_id]
    return handle.receipt_issuer.issue(
        VerifiedCheckpointReceipt(
            issuer="checkpoint-store",
            receipt_id=f"receipt-{suffix}",
            tenant_id=handle.owner.tenant_id,
            job_id=state.broker_job_id,
            conversation_id=state.conversation_id,
            generation=generation,
            barrier_id=barrier,
            artifact_id=f"artifact-{suffix}",
            base_commit="c" * 40,
            ciphertext_sha256="d" * 64,
            plaintext_sha256="e" * 64,
            byte_count=2048,
            store_version="store-v1",
            envelope_version="envelope-v1",
            key_version="key-v1",
            durable_write_sequence=generation,
        )
    )


async def test_park_resume_finalize_lifecycle(tmp_path, sock_dir):
    job = "job-lifecycle-01"
    async with AsyncExitStack() as stack:
        handle = await build_broker(
            _settings(tmp_path, sock_dir),
            stack,
            runtime_driver=InMemoryRuntimeDriver(),
        )
        assert handle is not None
        adapter = BrokerWorkspaceAdapter(
            client=handle.client,
            loop=handle.loop,
            receipt_issuer=handle.receipt_issuer,
            workspace_factory=lambda endpoint: _FakeWorkspace(endpoint),
        )

        # Segment 1 → running generation 1, quiesce at a barrier, park.
        await asyncio.to_thread(adapter.create, tmp_path, job)
        assert adapter._jobs[job].running_generation == 1
        receipt1 = _sign_receipt(
            handle, adapter, job, generation=1, barrier="barrier-01", suffix="01"
        )
        await asyncio.to_thread(adapter.quiesce, job, "barrier-01")
        await asyncio.to_thread(adapter.park, job, receipt1)
        assert adapter._jobs[job].current_generation == 1
        assert adapter._jobs[job].running_generation is None

        # Resume → running generation 2, quiesce, finalize.
        await asyncio.to_thread(adapter.create, tmp_path, job)
        assert adapter._jobs[job].running_generation == 2
        receipt2 = _sign_receipt(
            handle, adapter, job, generation=2, barrier="barrier-02", suffix="02"
        )
        await asyncio.to_thread(adapter.quiesce, job, "barrier-02")
        await asyncio.to_thread(adapter.finalize, job, receipt2)
        assert adapter._jobs[job].running_generation is None


async def test_checkpoint_store_receipt_is_accepted_by_broker(tmp_path, sock_dir):
    # The production receipt path: a real LocalCheckpointReceiptStore signing with
    # the broker's receipt keypair issues a receipt the broker's release_segment
    # accepts — no worker/SDK/Docker involved.
    import io
    import os

    from openloop.broker_control.local_receipts import (
        LocalCheckpointReceiptStore,
        checkpoint_artifact_identity,
    )
    from openloop.broker_control.receipts import CheckpointReceiptKey
    from openloop.tools.openhands_artifacts import (
        WorkspaceArtifactManifest,
        WorkspaceArtifactStore,
    )
    from openloop.tools.openhands_state import (
        OpenHandsKeyDeriver,
        OpenHandsStateLayout,
    )

    job = "job-checkpoint-01"
    async with AsyncExitStack() as stack:
        handle = await build_broker(
            _settings(tmp_path, sock_dir),
            stack,
            runtime_driver=InMemoryRuntimeDriver(),
        )
        assert handle is not None
        adapter = BrokerWorkspaceAdapter(
            client=handle.client,
            loop=handle.loop,
            receipt_issuer=handle.receipt_issuer,
            workspace_factory=lambda endpoint: _FakeWorkspace(endpoint),
        )
        await asyncio.to_thread(adapter.create, tmp_path, job)
        state = adapter._jobs[job]
        generation = state.running_generation

        artifacts = WorkspaceArtifactStore(
            OpenHandsStateLayout(tmp_path / "ohartstate"),
            OpenHandsKeyDeriver(bytes(range(32)), master_key_id="artifact-v1"),
            scratch_root=tmp_path / "ohartscratch",
        )
        checkpoint_store = LocalCheckpointReceiptStore(
            artifact_store=artifacts,
            issuer=handle.receipt_issuer,
            historical_verifier=handle.receipt_verifier,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )
        key = CheckpointReceiptKey(
            handle.owner.tenant_id,
            state.broker_job_id,
            state.conversation_id,
            generation,
            "barrier-checkpoint-01",
        )
        descriptor = artifacts.put_atomic(
            checkpoint_artifact_identity(key),
            io.BytesIO(b"git-delta payload"),
            WorkspaceArtifactManifest(format="git-delta", base_commit="a" * 40),
        )
        receipt = await checkpoint_store.publish(key, descriptor)

        await asyncio.to_thread(adapter.quiesce, job, "barrier-checkpoint-01")
        await asyncio.to_thread(adapter.park, job, receipt)

    assert state.current_generation == generation
    assert state.running_generation is None


async def test_distinct_jobs_get_distinct_broker_jobs(tmp_path, sock_dir):
    async with AsyncExitStack() as stack:
        adapter = await _broker_adapter(
            stack, tmp_path, sock_dir, lambda endpoint: _FakeWorkspace(endpoint)
        )
        await asyncio.to_thread(adapter.create, tmp_path, "job-one-000001")
        await asyncio.to_thread(adapter.create, tmp_path, "job-two-000002")

    first = adapter._jobs["job-one-000001"].broker_job_id
    second = adapter._jobs["job-two-000002"].broker_job_id
    assert first != second


def test_stream_git_delta_wraps_relay_result():
    # No broker needed: the adapter delegates to the workspace's own streamer
    # and adapts the (base_commit, written) tuple into an ArchiveStreamResult.
    adapter = BrokerWorkspaceAdapter(
        client=object(),  # unused on this path
        loop=asyncio.new_event_loop(),
        workspace_factory=lambda endpoint: None,
    )
    recorded = {}

    def stream(sink, *, base_ref):
        recorded["base_ref"] = base_ref
        return "a" * 40, 512

    workspace = SimpleNamespace(stream_git_delta=stream)
    sink = object()
    result = adapter.stream_git_delta(workspace, sink, base_ref="deadbeef")

    assert isinstance(result, ArchiveStreamResult)
    assert result.base_commit == "a" * 40
    assert result.bytes_written == 512
    assert recorded["base_ref"] == "deadbeef"


def test_attach_conversation_refuses_missing_conversation():
    adapter = BrokerWorkspaceAdapter(
        client=object(),
        loop=asyncio.new_event_loop(),
        workspace_factory=lambda endpoint: None,
    )
    workspace = SimpleNamespace(
        api_key="session-key",
        client=SimpleNamespace(get=lambda path: SimpleNamespace(status_code=404)),
    )
    import uuid

    with pytest.raises(BrokerWorkspaceError):
        adapter.attach_conversation(
            workspace, agent=object(), conversation_id=uuid.uuid4()
        )


def test_probe_is_noop_with_injected_factory():
    # A custom factory means no SDK dependency to verify — probe must not raise.
    adapter = BrokerWorkspaceAdapter(
        client=object(),
        loop=asyncio.new_event_loop(),
        workspace_factory=lambda endpoint: None,
    )
    adapter.probe()


async def test_build_coding_worker_selects_broker_adapter(tmp_path, sock_dir):
    # End-to-end wiring: flag on + openhands + docker + a composed handle makes
    # build_coding_worker construct the OpenHands worker over the broker adapter
    # (not the direct HardenedDockerWorkspace).
    from openloop.tools.openhands_relay import OpenHandsRelayError, probe_relay_compatibility

    try:
        probe_relay_compatibility()
    except OpenHandsRelayError:
        pytest.skip("pinned OpenHands relay SDK is unavailable/incompatible")

    from openloop.wiring.builders import build_coding_worker

    ohstate = tmp_path / "ohstate"
    ohstate.mkdir()
    settings = _settings(
        tmp_path,
        sock_dir,
        coding_worker_backend="openhands",
        coding_worker_sandbox="docker",
        coding_worker_openhands_state_dir=str(ohstate),
        coding_worker_openhands_state_master_key=base64.b64encode(b"y" * 32).decode(),
    )
    async with AsyncExitStack() as stack:
        handle = await build_broker(
            settings, stack, runtime_driver=InMemoryRuntimeDriver()
        )
        worker = build_coding_worker(settings, broker_handle=handle)

    assert worker is not None
    assert isinstance(worker._docker_adapter, BrokerWorkspaceAdapter)
    assert worker._docker_adapter.runs_containers_locally is False
