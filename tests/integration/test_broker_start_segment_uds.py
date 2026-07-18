import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
import struct
import tempfile
from uuid import UUID

import pytest

from openloop.broker.models import (
    BrokerOwner,
    GenerationState,
    IsolationMode,
    JobState,
    ReleaseTarget,
    TerminalOutcome,
    VerifiedCheckpointReceipt,
)
from openloop.broker_rpc.audit import PeerCredentials
from openloop.broker_rpc.capability import JobCapability
from openloop.broker_rpc.client import BrokerRpcClient, BrokerRpcRemoteError
from openloop.broker_rpc.codec import decode_response, encode_request
from openloop.broker_rpc.errors import RpcErrorCode
from openloop.broker_rpc.identity import WorkloadIntent
from openloop.broker_rpc.models import (
    RPC_VERSION,
    RpcRequest,
    StartSegmentPayload,
)
from openloop.broker_rpc.peer import StaticPeerCredentialProvider
from openloop.broker_rpc.server import BrokerRpcServer, UnixSocketPolicy
from openloop.broker_runtime.contract import (
    GenerationRuntimeIdentity,
    RuntimeHealthFailure,
)
from openloop.broker_runtime.memory import InMemoryRuntimeDriver
from tests.support.broker_repository_contract import SequenceIds
from tests.support.broker_rpc import broker_rpc_test_fixture


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
OWNER = BrokerOwner("tenant-start", "workload-start")
OTHER_OWNER = BrokerOwner("tenant-other", "workload-other")


class ControlledRuntime(InMemoryRuntimeDriver):
    def __init__(self):
        super().__init__(clock=lambda: NOW, maximum_lifetime_seconds=600)
        self.ensure_calls = 0
        self.fail_next_ensure = False

    async def ensure(self, spec):
        self.ensure_calls += 1
        if self.fail_next_ensure:
            self.fail_next_ensure = False
            raise RuntimeHealthFailure("injected health failure")
        return await super().ensure(spec)

    @property
    def identity_count(self):
        return len(self._specs)


@pytest.fixture
def private_paths():
    directory = Path(tempfile.mkdtemp(prefix="olstart-", dir="/private/tmp"))
    state_root = directory / "state"
    state_root.mkdir(mode=0o700)
    os.chown(state_root, os.getuid(), os.getgid())
    state_root.chmod(0o700)
    try:
        yield directory / "broker.sock", state_root
    finally:
        shutil.rmtree(directory)


def _server(path, fixture):
    return BrokerRpcServer(
        application=fixture.application,
        socket_policy=UnixSocketPolicy(path, mode=0o600),
        peer_provider=StaticPeerCredentialProvider(
            PeerCredentials(os.getpid(), os.getuid(), os.getgid())
        ),
    )


def _client(path, fixture, owner=OWNER, *, start=20_000, **identity_options):
    return BrokerRpcClient(
        path=path,
        identity_provider=fixture.identity_provider(
            owner, **identity_options
        ),
        request_id_factory=SequenceIds(start=start),
    )


async def _raw_exchange(path: Path, request: RpcRequest):
    reader, writer = await asyncio.open_unix_connection(path)
    try:
        writer.write(encode_request(request))
        await writer.drain()
        prefix = await reader.readexactly(4)
        body = await reader.readexactly(struct.unpack(">I", prefix)[0])
        return decode_response(prefix + body)
    finally:
        writer.close()
        await writer.wait_closed()


async def test_real_uds_start_replay_and_inspect(private_paths):
    socket_path, state_root = private_paths
    runtime = ControlledRuntime()
    fixture = broker_rpc_test_fixture(
        state_root=state_root,
        runtime_driver=runtime,
    )
    server = _server(socket_path, fixture)
    await server.start()
    try:
        client = _client(socket_path, fixture)
        created = await client.create_job("uds-start-create-01")
        first = await client.start_segment(
            created.ticket.job_id,
            0,
            "uds-start-key-01",
            created.capability,
        )

        recovery = await fixture.ledger.inspect_job_for_recovery(
            OWNER, created.ticket.job_id
        )
        assert recovery.state is JobState.ACTIVE
        assert recovery.generation_record.state is GenerationState.RUNNING
        assert first.access.conversation_id == created.ticket.conversation_id
        assert first.access.generation == 1
        assert first.access.socket_path.name == "agent.sock"
        assert first.access.relay_capability != first.access.session_api_key
        durable_job = state_root / str(created.ticket.job_id)
        assert sorted(item.name for item in durable_job.iterdir()) == [
            "agent-server"
        ]
        marker = durable_job / "agent-server" / "preserved.txt"
        marker.write_text("preserve", encoding="utf-8")

        replay = await client.start_segment(
            created.ticket.job_id,
            0,
            "uds-start-key-01",
            created.capability,
        )
        inspected = await client.inspect_job(
            created.ticket.job_id, created.capability
        )
        assert replay.replayed is True
        assert replay.operation_id == first.operation_id
        assert replay.access == first.access == inspected.access
        assert marker.read_text(encoding="utf-8") == "preserve"
        assert runtime.ensure_calls == 2
        assert runtime.identity_count == 1

        generation = recovery.generation_record
        await runtime.release(
            GenerationRuntimeIdentity(
                generation.start_operation_id,
                created.ticket.job_id,
                generation.generation,
                generation.execution_lease_deadline,
            )
        )
        unavailable = await client.inspect_job(
            created.ticket.job_id, created.capability
        )
        assert unavailable.snapshot.state is JobState.ACTIVE
        assert unavailable.access is None
        assert runtime.ensure_calls == 2
    finally:
        await server.stop()


async def test_real_uds_checkpoint_release_and_finalize_lifecycle(private_paths):
    socket_path, state_root = private_paths
    runtime = ControlledRuntime()
    fixture = broker_rpc_test_fixture(
        state_root=state_root,
        runtime_driver=runtime,
    )
    server = _server(socket_path, fixture)
    await server.start()
    try:
        client = _client(socket_path, fixture, start=20_500)
        created = await client.create_job("uds-lifecycle-create-01")
        first_start = await client.start_segment(
            created.ticket.job_id,
            0,
            "uds-lifecycle-start-01",
            created.capability,
        )
        first_quiesce = await client.quiesce_segment(
            created.ticket.job_id,
            1,
            "uds-lifecycle-quiesce-01",
            "barrier-uds-01",
            created.capability,
        )
        first_receipt = fixture.receipt_issuer.issue(
            VerifiedCheckpointReceipt(
                issuer="checkpoint-store",
                receipt_id="receipt-uds-01",
                tenant_id=OWNER.tenant_id,
                job_id=created.ticket.job_id,
                conversation_id=created.ticket.conversation_id,
                generation=1,
                barrier_id="barrier-uds-01",
                artifact_id="artifact-uds-01",
                base_commit="c" * 40,
                ciphertext_sha256="d" * 64,
                plaintext_sha256="e" * 64,
                byte_count=2048,
                store_version="store-v1",
                envelope_version="envelope-v1",
                key_version="key-v1",
                durable_write_sequence=1,
            )
        )
        parked = await client.release_segment(
            created.ticket.job_id,
            1,
            "uds-lifecycle-release-01",
            first_receipt,
            ReleaseTarget.PARKED,
            created.capability,
        )

        assert first_quiesce.access.generation == first_start.access.generation
        assert parked.job_state is JobState.PARKED
        assert parked.generation_state is GenerationState.RELEASED
        parked_inspection = await client.inspect_job(
            created.ticket.job_id, created.capability
        )
        assert parked_inspection.snapshot.state is JobState.PARKED
        assert parked_inspection.access is None
        assert runtime.identity_count == 0

        second_start = await client.start_segment(
            created.ticket.job_id,
            1,
            "uds-lifecycle-start-02",
            created.capability,
        )
        await client.quiesce_segment(
            created.ticket.job_id,
            2,
            "uds-lifecycle-quiesce-02",
            "barrier-uds-02",
            created.capability,
        )
        second_receipt = fixture.receipt_issuer.issue(
            VerifiedCheckpointReceipt(
                issuer="checkpoint-store",
                receipt_id="receipt-uds-02",
                tenant_id=OWNER.tenant_id,
                job_id=created.ticket.job_id,
                conversation_id=second_start.access.conversation_id,
                generation=2,
                barrier_id="barrier-uds-02",
                artifact_id="artifact-uds-02",
                base_commit="f" * 40,
                ciphertext_sha256="1" * 64,
                plaintext_sha256="2" * 64,
                byte_count=4096,
                store_version="store-v1",
                envelope_version="envelope-v1",
                key_version="key-v1",
                durable_write_sequence=2,
            )
        )
        finalizing = await client.release_segment(
            created.ticket.job_id,
            2,
            "uds-lifecycle-release-02",
            second_receipt,
            ReleaseTarget.FINALIZING,
            created.capability,
            terminal_outcome=TerminalOutcome.SUCCESS,
        )
        terminal = await client.finalize_job(
            created.ticket.job_id,
            2,
            "uds-lifecycle-finalize-01",
            TerminalOutcome.SUCCESS,
            created.capability,
        )

        assert finalizing.job_state is JobState.FINALIZING
        assert terminal.job_state is JobState.TERMINAL
        terminal_inspection = await client.inspect_job(
            created.ticket.job_id, created.capability
        )
        assert terminal_inspection.snapshot.state is JobState.TERMINAL
        assert terminal_inspection.snapshot.terminal_outcome is TerminalOutcome.SUCCESS
        assert terminal_inspection.access is None
        assert runtime.identity_count == 0
    finally:
        await server.stop()


async def test_lost_success_response_converges_with_same_key(private_paths):
    socket_path, state_root = private_paths
    runtime = ControlledRuntime()
    fixture = broker_rpc_test_fixture(
        state_root=state_root,
        runtime_driver=runtime,
    )
    server = _server(socket_path, fixture)
    await server.start()
    try:
        client = _client(socket_path, fixture, start=21_000)
        created = await client.create_job("uds-lost-create-01")
        request = RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000021500"),
            WorkloadIntent.START_SEGMENT,
            fixture.identity_provider(OWNER)(WorkloadIntent.START_SEGMENT),
            created.capability,
            StartSegmentPayload(
                created.ticket.job_id,
                0,
                "uds-lost-start-01",
            ),
        )
        _, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(encode_request(request))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

        for _ in range(100):
            recovery = await fixture.ledger.inspect_job_for_recovery(
                OWNER, created.ticket.job_id
            )
            if recovery.state is JobState.ACTIVE:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("dropped start response did not commit")

        replay = await client.start_segment(
            created.ticket.job_id,
            0,
            "uds-lost-start-01",
            created.capability,
        )
        assert replay.replayed is True
        assert replay.access.generation == 1
        assert runtime.identity_count == 1
    finally:
        await server.stop()


async def test_start_authorization_and_fencing_precede_runtime(private_paths):
    socket_path, state_root = private_paths
    runtime = ControlledRuntime()
    fixture = broker_rpc_test_fixture(
        state_root=state_root,
        runtime_driver=runtime,
    )
    server = _server(socket_path, fixture)
    await server.start()
    try:
        client = _client(socket_path, fixture, start=22_000)
        created = await client.create_job("uds-auth-create-01")
        other = _client(socket_path, fixture, OTHER_OWNER, start=22_100)

        with pytest.raises(BrokerRpcRemoteError) as wrong_owner:
            await other.start_segment(
                created.ticket.job_id,
                0,
                "uds-auth-owner-01",
                created.capability,
            )
        assert wrong_owner.value.code is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED

        with pytest.raises(BrokerRpcRemoteError) as wrong_capability:
            await client.start_segment(
                created.ticket.job_id,
                0,
                "uds-auth-capability-01",
                JobCapability("A" * 43),
            )
        assert (
            wrong_capability.value.code
            is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        )

        token = fixture.identity_provider(OWNER)(WorkloadIntent.INSPECT_JOB)
        missing_intent = RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000022500"),
            WorkloadIntent.START_SEGMENT,
            token,
            created.capability,
            StartSegmentPayload(
                created.ticket.job_id,
                0,
                "uds-auth-intent-01",
            ),
        )
        response = await _raw_exchange(socket_path, missing_intent)
        assert response.failure.code is RpcErrorCode.METHOD_NOT_ALLOWED

        dedicated_creator = _client(
            socket_path,
            fixture,
            start=22_600,
            required=IsolationMode.DEDICATED,
        )
        dedicated = await dedicated_creator.create_job(
            "uds-auth-dedicated-create-01"
        )
        shared = _client(
            socket_path,
            fixture,
            start=22_700,
            isolation=IsolationMode.SHARED,
        )
        with pytest.raises(BrokerRpcRemoteError) as weak_isolation:
            await shared.start_segment(
                dedicated.ticket.job_id,
                0,
                "uds-auth-isolation-01",
                dedicated.capability,
            )
        assert (
            weak_isolation.value.code
            is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        )

        with pytest.raises(BrokerRpcRemoteError) as stale:
            await client.start_segment(
                created.ticket.job_id,
                1,
                "uds-auth-stale-01",
                created.capability,
            )
        assert stale.value.code is RpcErrorCode.STATE_CONFLICT
        assert runtime.ensure_calls == 0
        assert runtime.identity_count == 0
    finally:
        await server.stop()


async def test_abandoned_start_requires_new_key_and_generation(private_paths):
    socket_path, state_root = private_paths
    runtime = ControlledRuntime()
    fixture = broker_rpc_test_fixture(
        state_root=state_root,
        runtime_driver=runtime,
    )
    server = _server(socket_path, fixture)
    await server.start()
    try:
        client = _client(socket_path, fixture, start=23_000)
        created = await client.create_job("uds-retry-create-01")
        runtime.fail_next_ensure = True

        with pytest.raises(BrokerRpcRemoteError) as failed:
            await client.start_segment(
                created.ticket.job_id,
                0,
                "uds-retry-start-01",
                created.capability,
            )
        assert failed.value.code is RpcErrorCode.RUNTIME_UNAVAILABLE
        recovery = await fixture.ledger.inspect_job_for_recovery(
            OWNER, created.ticket.job_id
        )
        assert recovery.generation_record.state is GenerationState.ABANDONED

        with pytest.raises(BrokerRpcRemoteError) as replay:
            await client.start_segment(
                created.ticket.job_id,
                0,
                "uds-retry-start-01",
                created.capability,
            )
        assert replay.value.code is RpcErrorCode.RUNTIME_UNAVAILABLE
        assert runtime.ensure_calls == 1

        retried = await client.start_segment(
            created.ticket.job_id,
            1,
            "uds-retry-start-02",
            created.capability,
        )
        assert retried.access.generation == 2
        assert runtime.identity_count == 1
    finally:
        await server.stop()
