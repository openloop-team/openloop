import asyncio
import os
from pathlib import Path
import struct
from uuid import UUID

import pytest

from openloop.broker.models import BrokerOwner
from openloop.broker_rpc.audit import PeerCredentials
from openloop.broker_rpc.capability import JobCapability
from openloop.broker_rpc.client import (
    BrokerRpcClient,
    BrokerRpcClientProblem,
    BrokerRpcRemoteError,
)
from openloop.broker_rpc.codec import decode_response, encode_request
from openloop.broker_rpc.errors import RpcErrorCode
from openloop.broker_rpc.identity import WorkloadIntent
from openloop.broker_rpc.limits import BrokerRpcLimits
from openloop.broker_rpc.models import (
    CreateJobPayload,
    CreateJobResult,
    RPC_VERSION,
    RpcRequest,
)
from openloop.broker_rpc.peer import StaticPeerCredentialProvider
from openloop.broker_rpc.server import (
    BrokerRpcServer,
    UnixSocketPolicy,
)
from tests.support.broker_repository_contract import SequenceIds
from tests.support.broker_rpc import broker_rpc_test_fixture


OWNER = BrokerOwner("tenant-a", "workload-a")
OTHER_OWNER = BrokerOwner("tenant-b", "workload-b")


@pytest.fixture
def socket_path(short_socket_root):
    return short_socket_root / "broker.sock"


def _server(path, fixture, *, limits=BrokerRpcLimits()):
    return BrokerRpcServer(
        application=fixture.application,
        socket_policy=UnixSocketPolicy(path, mode=0o600),
        peer_provider=StaticPeerCredentialProvider(
            PeerCredentials(os.getpid(), os.getuid(), os.getgid())
        ),
        limits=limits,
    )


def _client(path, fixture, owner, *, start):
    return BrokerRpcClient(
        path=path,
        identity_provider=fixture.identity_provider(owner),
        request_id_factory=SequenceIds(start=start),
    )


async def test_real_uds_create_replay_inspect_and_cross_tenant_denial(socket_path):
    fixture = broker_rpc_test_fixture()
    path = socket_path
    server = _server(path, fixture)
    await server.start()
    try:
        client = _client(path, fixture, OWNER, start=6000)
        first = await client.create_job("rpc-uds-create-key-01")
        replay = await client.create_job("rpc-uds-create-key-01")
        assert replay.ticket.replayed is True
        assert replay.ticket.job_id == first.ticket.job_id
        assert replay.capability == first.capability

        inspected = await client.inspect_job(
            first.ticket.job_id, first.capability
        )
        assert inspected.snapshot.job_id == first.ticket.job_id

        other = _client(path, fixture, OTHER_OWNER, start=7000)
        with pytest.raises(BrokerRpcRemoteError) as denied:
            await other.inspect_job(first.ticket.job_id, first.capability)
        assert denied.value.code is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED

        with pytest.raises(BrokerRpcRemoteError) as wrong_capability:
            await client.inspect_job(first.ticket.job_id, JobCapability("A" * 43))
        assert (
            wrong_capability.value.code
            is RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        )
    finally:
        await server.stop()
    assert not path.exists()


async def test_real_uds_accepts_fragmented_single_request(socket_path):
    fixture = broker_rpc_test_fixture()
    path = socket_path
    server = _server(path, fixture)
    await server.start()
    writer = None
    try:
        token = fixture.identity_provider(OWNER)(WorkloadIntent.CREATE_JOB)
        request = RpcRequest(
            RPC_VERSION,
            UUID("00000000-0000-4000-8000-000000008001"),
            WorkloadIntent.CREATE_JOB,
            token,
            None,
            CreateJobPayload("rpc-uds-create-key-02"),
        )
        frame = encode_request(request)
        reader, writer = await asyncio.open_unix_connection(path)
        for byte in frame:
            writer.write(bytes((byte,)))
            await writer.drain()
        prefix = await reader.readexactly(4)
        body = await reader.readexactly(struct.unpack(">I", prefix)[0])
        response = decode_response(prefix + body)
        assert isinstance(response.result, CreateJobResult)
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        await server.stop()


async def test_authenticated_principal_overload_is_returned_and_audited(
    socket_path,
):
    fixture = broker_rpc_test_fixture()
    limits = BrokerRpcLimits(
        per_principal_capacity=1,
        per_principal_refill_per_second=0.0001,
    )
    server = _server(socket_path, fixture, limits=limits)
    await server.start()
    try:
        client = _client(socket_path, fixture, OWNER, start=9000)
        await client.create_job("rpc-uds-overload-key-01")
        with pytest.raises(BrokerRpcRemoteError) as overloaded:
            await client.create_job("rpc-uds-overload-key-02")
        assert overloaded.value.code is RpcErrorCode.OVERLOADED
        assert len(await fixture.audit.records_for_test()) == 2
    finally:
        await server.stop()


async def test_pre_auth_slow_prefix_is_closed_without_audit(socket_path):
    fixture = broker_rpc_test_fixture()
    limits = BrokerRpcLimits(prefix_timeout_seconds=0.01)
    server = _server(socket_path, fixture, limits=limits)
    await server.start()
    writer = None
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        assert await asyncio.wait_for(reader.read(), timeout=1) == b""
        assert await fixture.audit.records_for_test() == ()
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        await server.stop()


async def test_concurrent_clients_remain_typed_and_bounded(socket_path):
    fixture = broker_rpc_test_fixture()
    server = _server(socket_path, fixture)
    await server.start()
    try:
        client = _client(socket_path, fixture, OWNER, start=10_000)
        assert not hasattr(client, "call")
        created = await asyncio.gather(
            *(
                client.create_job(f"rpc-uds-concurrent-{number:02d}")
                for number in range(16)
            )
        )
        assert len({item.ticket.job_id for item in created}) == 16
    finally:
        await server.stop()


async def test_application_deadline_returns_safe_error_while_work_converges(
    socket_path,
):
    fixture = broker_rpc_test_fixture()
    original_handle = fixture.application.handle

    async def slow_handle(*args, **kwargs):
        await asyncio.sleep(0.05)
        return await original_handle(*args, **kwargs)

    fixture.application.handle = slow_handle
    limits = BrokerRpcLimits(application_timeout_seconds=0.01)
    server = _server(socket_path, fixture, limits=limits)
    await server.start()
    try:
        client = _client(socket_path, fixture, OWNER, start=11_000)
        with pytest.raises(BrokerRpcRemoteError) as deadline:
            await client.create_job("rpc-uds-deadline-key-01")
        assert deadline.value.code is RpcErrorCode.DEADLINE_EXCEEDED
    finally:
        await server.stop()
    assert len(await fixture.repository.audit_records_for_test()) == 1
    assert len(await fixture.audit.records_for_test()) == 1


async def test_pre_auth_peer_overload_closes_without_extra_audit(socket_path):
    fixture = broker_rpc_test_fixture()
    limits = BrokerRpcLimits(
        per_peer_capacity=1,
        per_peer_refill_per_second=0.0001,
    )
    server = _server(socket_path, fixture, limits=limits)
    await server.start()
    try:
        client = _client(socket_path, fixture, OWNER, start=12_000)
        await client.create_job("rpc-uds-peer-limit-key-01")
        with pytest.raises(BrokerRpcClientProblem):
            await client.create_job("rpc-uds-peer-limit-key-02")
        assert len(await fixture.audit.records_for_test()) == 1
    finally:
        await server.stop()
