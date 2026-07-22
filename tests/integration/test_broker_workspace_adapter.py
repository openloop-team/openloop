"""BrokerWorkspaceAdapter forward path (phase 3a).

`create` runs against a real in-memory co-process broker (real create_job +
start_segment, real generation semantics) with only the SDK workspace
construction faked. `stream_git_delta`/`attach_conversation`/`probe` are covered
with fakes so no OpenHands SDK, Docker, or relay is needed.
"""

import asyncio
import base64
import io
import subprocess
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from openloop.broker.models import JobState, VerifiedCheckpointReceipt
from openloop.broker_runtime.memory import InMemoryRuntimeDriver
from openloop.config import Settings
from openloop.openhands.workspace_protocol import ArchiveStreamResult
from openloop.tools.openhands_broker_workspace import (
    BrokerWorkspaceAdapter,
    BrokerWorkspaceError,
)
from openloop.tools.openhands_resume import ResumeDecision, WorkerPaused
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout
from openloop.tools.openhands_artifacts import (
    WorkspaceArtifactManifest,
    WorkspaceArtifactStore,
)
from openloop.tools.openhands_worker import (
    _ColdRuntime,
    OpenHandsCodingWorker,
)
from openloop.tools.coding_worker import WorkerState
from openloop.tools.openhands_relay_client import RelayClientEndpoint, RelayMode
from openloop.wiring.broker import build_broker


@pytest.fixture
def sock_dir(short_socket_root):
    return short_socket_root


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


async def test_create_prunes_staged_tree_after_start(tmp_path, sock_dir):
    # With a real workspace ingress wired, a successful start prunes the staged
    # seed — the producer-deletes contract leaves nothing behind.
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "file.txt").write_text("payload")

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
            workspace_ingress=handle.workspace_ingress,
            workspace_factory=lambda endpoint: _FakeWorkspace(endpoint),
        )
        await asyncio.to_thread(adapter.create, seed, "job-prune-000001")
        state = adapter._jobs["job-prune-000001"]

    ingress_root = handle.workspace_ingress.root
    assert not (ingress_root / str(state.broker_job_id)).exists()


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


class _LifecycleWorkspace:
    def __init__(self, endpoint, patch):
        self.endpoint = endpoint
        self._patch = patch

    def stream_git_delta(self, sink, *, base_ref):
        sink.write(self._patch)
        return base_ref, len(self._patch)


class _LifecycleConversation:
    def __init__(self, workspace, callbacks, status):
        self.workspace = workspace
        self.callbacks = callbacks
        action = type("TerminalAction", (), {"tool_name": "terminal"})()
        self.state = SimpleNamespace(
            execution_status=status,
            events=[action],
        )
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        metrics = SimpleNamespace(accumulated_cost=0.1, accumulated_token_usage=usage)
        self.conversation_stats = SimpleNamespace(
            get_combined_metrics=lambda: metrics
        )
        self.prompt = None

    def send_message(self, prompt):
        self.prompt = prompt

    def set_confirmation_policy(self, _policy):
        return None

    def run(self):
        for callback in self.callbacks:
            callback(self.state.events[-1])

    def close(self):
        return None


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


async def test_worker_checkpoint_park_resume_finalize_over_real_rpc(tmp_path, sock_dir):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "canary@example.invalid")
    _git(repo, "config", "user.name", "Canary")
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    final_patch = (
        "diff --git a/OPENLOOP_PR.md b/OPENLOOP_PR.md\n"
        "new file mode 100644\n--- /dev/null\n+++ b/OPENLOOP_PR.md\n"
        "@@ -0,0 +1,2 @@\n+Broker canary\n+Body\n"
        "diff --git a/result.txt b/result.txt\n"
        "new file mode 100644\n--- /dev/null\n+++ b/result.txt\n"
        "@@ -0,0 +1 @@\n+resumed\n"
    ).encode()
    patches = [b"", final_patch]

    async with AsyncExitStack() as stack:
        handle = await build_broker(
            _settings(tmp_path, sock_dir),
            stack,
            runtime_driver=InMemoryRuntimeDriver(),
        )
        assert handle is not None
        layout = OpenHandsStateLayout(tmp_path / "artifacts")
        keys = OpenHandsKeyDeriver(bytes(range(32)), master_key_id="artifact-v1")
        artifacts = WorkspaceArtifactStore(
            layout, keys, scratch_root=tmp_path / "artifact-scratch"
        )
        checkpoint_store = handle.bind_checkpoint_store(artifacts)
        adapter = BrokerWorkspaceAdapter(
            client=handle.client,
            loop=handle.loop,
            checkpoint_store=checkpoint_store,
            workspace_factory=lambda endpoint: _LifecycleWorkspace(
                endpoint, patches.pop(0)
            ),
            tenant_id=handle.owner.tenant_id,
        )
        worker = OpenHandsCodingWorker(
            "openai/fake",
            docker=True,
            docker_adapter=adapter,
            artifact_store=artifacts,
            cold_resume_enabled=True,
        )
        worker._git_head = lambda workspace: base
        status = ["WAITING_FOR_CONFIRMATION", "FINISHED"]

        def open_runtime(workspace, state, callbacks):
            target = adapter.create(workspace, state.job_id)
            conversation = _LifecycleConversation(target, callbacks, status.pop(0))
            return _ColdRuntime(conversation, target, lambda: None)

        worker._open_cold_runtime = open_runtime
        state = WorkerState(
            job_id="job-canary-0001",
            repo="acme/repo",
            instruction="exercise broker",
            base="main",
            branch="openloop/job-canary-0001",
            requester_id="U123",
        )

        paused = await worker.run(repo, state)
        assert isinstance(paused, WorkerPaused)
        assert state.openhands_resume.status == "parking"
        assert await asyncio.to_thread(
            adapter.is_parked,
            state.job_id,
            state.openhands_resume.broker_job_id,
            state.openhands_resume.broker_generation,
        )
        state.openhands_resume.transition_to("parked")
        decision = ResumeDecision("accept", paused.decision_id, "event-1", "U123")
        state.openhands_resume.transition_to(
            "resuming",
            segment_id=uuid.uuid4().hex,
            resolved_event_id=decision.event_id,
            resolved_decision=decision,
        )

        edit = await worker.run(repo, state)
        assert edit.workspace_artifact.artifact.identity.kind == "checkpoint"
        assert state.openhands_resume.status == "terminal"
        assert state.openhands_resume.broker_generation == 2
        assert (repo / "result.txt").read_text() == "resumed\n"
        assert not (repo / "OPENLOOP_PR.md").exists()
        broker_state = adapter._jobs[state.job_id]
        inspected = await handle.client.inspect_job(
            broker_state.broker_job_id, broker_state.capability
        )
        assert inspected.snapshot.state.value == "terminal"


async def test_recover_checkpoint_replays_pre_effect_app_intents(tmp_path, sock_dir):
    job = "job-recovery-0001"
    async with AsyncExitStack() as stack:
        handle = await build_broker(
            _settings(tmp_path, sock_dir),
            stack,
            runtime_driver=InMemoryRuntimeDriver(),
        )
        assert handle is not None
        artifacts = WorkspaceArtifactStore(
            OpenHandsStateLayout(tmp_path / "recovery-artifacts"),
            OpenHandsKeyDeriver(bytes(range(32)), master_key_id="artifact-v1"),
            scratch_root=tmp_path / "recovery-scratch",
        )
        checkpoint_store = handle.bind_checkpoint_store(artifacts)

        def new_adapter():
            return BrokerWorkspaceAdapter(
                client=handle.client,
                loop=handle.loop,
                checkpoint_store=checkpoint_store,
                workspace_factory=lambda endpoint: _FakeWorkspace(endpoint),
                tenant_id=handle.owner.tenant_id,
            )

        initial = new_adapter()
        await asyncio.to_thread(initial.create, tmp_path, job)
        first = initial.generation_identity(job)
        first_barrier = "barrier-recovery-01"
        first_descriptor = artifacts.put_atomic(
            initial.checkpoint_identity(job, first_barrier),
            io.BytesIO(b"first checkpoint"),
            WorkspaceArtifactManifest(format="git-delta", base_commit="a" * 40),
        )

        # Simulate an app crash after persisting PARKING but before quiesce.
        recovered = new_adapter()
        await asyncio.to_thread(
            recovered.recover_checkpoint,
            job,
            str(first.broker_job_id),
            first.generation,
            first_barrier,
            first_descriptor,
            terminal=False,
        )
        parked = await handle.client.inspect_job(
            first.broker_job_id, recovered._jobs[job].capability
        )
        assert parked.snapshot.state is JobState.PARKED

        await asyncio.to_thread(recovered.create, tmp_path, job)
        second = recovered.generation_identity(job)
        second_barrier = "barrier-recovery-02"
        second_descriptor = artifacts.put_atomic(
            recovered.checkpoint_identity(job, second_barrier),
            io.BytesIO(b"second checkpoint"),
            WorkspaceArtifactManifest(
                format="git-delta",
                base_commit="a" * 40,
                pr_title="Recovered terminal",
            ),
        )

        # Simulate the equivalent FINALIZING crash window in a fresh adapter.
        finalizer = new_adapter()
        await asyncio.to_thread(
            finalizer.recover_checkpoint,
            job,
            str(second.broker_job_id),
            second.generation,
            second_barrier,
            second_descriptor,
            terminal=True,
        )
        terminal = await handle.client.inspect_job(
            second.broker_job_id, finalizer._jobs[job].capability
        )
        assert terminal.snapshot.state is JobState.TERMINAL
