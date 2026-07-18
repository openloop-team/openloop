"""Linux broker/client process used only by the opt-in hardened live proof."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import sys
from uuid import UUID, uuid4

from openloop.broker.ledger import BrokerLedger
from openloop.broker.postgres import PostgresBrokerRepository
from openloop.broker_rpc.application import BrokerRpcApplication, BrokerRpcPolicy
from openloop.broker_rpc.audit import PostgresRpcAuditSink
from openloop.broker_rpc.capability import (
    CapabilityRootRing,
    JobCapability,
    JobCapabilityAuthority,
)
from openloop.broker_rpc.client import BrokerRpcClient, BrokerRpcRemoteError
from openloop.broker_rpc.codec import decode_response, encode_request
from openloop.broker_rpc.identity import (
    WorkloadIdentityToken,
    WorkloadIdentityVerifier,
    WorkloadIntent,
)
from openloop.broker_rpc.coordinator import (
    SegmentCoordinatorCode,
    SegmentCoordinatorProblem,
)
from openloop.broker_rpc.keys import VerificationKeySet
from openloop.broker_rpc.limits import BrokerRpcLimits
from openloop.broker_rpc.models import (
    CreateJobPayload,
    CreateJobResult,
    RPC_VERSION,
    RpcRequest,
)
from openloop.broker_rpc.peer import LinuxPeerCredentialProvider
from openloop.broker_rpc.server import (
    BrokerRpcServer,
    UnixSocketPolicy,
)
from openloop.postgres import create_pool


_CONFIG_PATH = Path("/run/openloop/config/broker.json")


class DisabledSegmentCoordinator:
    async def start_segment(self, owner, payload):
        raise SegmentCoordinatorProblem(SegmentCoordinatorCode.INTERNAL)

    async def inspect_running_access(self, owner, job_id):
        return None


def _read_config() -> dict[str, object]:
    with _CONFIG_PATH.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("broker probe config must be an object")
    return value


async def _broker() -> None:
    config = _read_config()
    expected_uid = os.getuid()
    verification_keys = VerificationKeySet.load(
        {str(config["issuer_key_id"]): str(config["issuer_public_key_path"])},
        expected_uid=expected_uid,
    )
    capability_roots = CapabilityRootRing.load(
        {str(config["capability_key_version"]): str(config["capability_root_path"])},
        current_version=str(config["capability_key_version"]),
        expected_uid=expected_uid,
    )
    pool = await create_pool(str(config["postgres_dsn"]), min_size=1, max_size=5)
    repository = PostgresBrokerRepository()
    audit = PostgresRpcAuditSink()
    server = None
    try:
        await repository.setup(pool)
        await audit.setup(pool)
        verifier = WorkloadIdentityVerifier(
            public_keys=verification_keys.snapshot(),
            issuer=str(config["issuer"]),
            audience=str(config["audience"]),
            clock=lambda: datetime.now(UTC),
        )
        application = BrokerRpcApplication(
            ledger=BrokerLedger(repository),
            identity_verifier=verifier,
            capability_authority=JobCapabilityAuthority(capability_roots),
            audit_sink=audit,
            policy=BrokerRpcPolicy("default", "docker", "postgres", 300),
            segment_coordinator=DisabledSegmentCoordinator(),
        )
        limits = BrokerRpcLimits(
            max_in_flight=32,
            per_peer_capacity=64,
            per_peer_refill_per_second=32,
            per_principal_capacity=64,
            per_principal_refill_per_second=32,
        )
        server = BrokerRpcServer(
            application=application,
            socket_policy=UnixSocketPolicy(
                Path(str(config["socket_path"])),
                mode=0o660,
                gid=os.getgid(),
            ),
            peer_provider=LinuxPeerCredentialProvider(),
            limits=limits,
        )
        await server.start()
        print(json.dumps({"ready": True}, separators=(",", ":")), flush=True)
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, stopped.set)
        await stopped.wait()
    finally:
        if server is not None:
            await server.stop()
        await audit.close()
        await repository.close()
        await pool.close()


def _token(value: object) -> WorkloadIdentityToken:
    if not isinstance(value, str):
        raise ValueError("probe token must be a string")
    return WorkloadIdentityToken(value)


def _client(path: Path, token_value: object) -> BrokerRpcClient:
    token = _token(token_value)
    return BrokerRpcClient(
        path=path,
        identity_provider=lambda _intent: token,
    )


async def _fragmented_create(
    path: Path, token_value: object, idempotency_key: str
) -> CreateJobResult:
    request = RpcRequest(
        RPC_VERSION,
        uuid4(),
        WorkloadIntent.CREATE_JOB,
        _token(token_value),
        None,
        CreateJobPayload(idempotency_key),
    )
    reader, writer = await asyncio.open_unix_connection(path)
    try:
        frame = encode_request(request)
        for offset in range(0, len(frame), 7):
            writer.write(frame[offset : offset + 7])
            await writer.drain()
        prefix = await reader.readexactly(4)
        body = await reader.readexactly(struct.unpack(">I", prefix)[0])
        response = decode_response(prefix + body)
        if response.failure is not None:
            raise BrokerRpcRemoteError(response.failure.code)
        if not isinstance(response.result, CreateJobResult):
            raise RuntimeError("fragmented CREATE_JOB returned wrong result")
        return response.result
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def _expect_denial(awaitable) -> str:
    try:
        await awaitable
    except BrokerRpcRemoteError as error:
        return error.code.value
    raise AssertionError("broker RPC request unexpectedly succeeded")


def _fingerprint(capability: JobCapability) -> str:
    return hashlib.sha256(capability.value.encode("ascii")).hexdigest()


async def _client_phase(request: dict[str, object]) -> dict[str, object]:
    path = Path(str(request["socket_path"]))
    key = str(request["idempotency_key"])
    phase = str(request["phase"])
    tokens = request["tokens"]
    if not isinstance(tokens, dict):
        raise ValueError("probe tokens must be an object")

    if phase == "initial":
        created = await _fragmented_create(path, tokens["create"], key)
        if created.ticket.job_id is None:
            raise RuntimeError("CREATE_JOB returned no job ID")
        replay = await _client(path, tokens["replay"]).create_job(key)
        inspected = await _client(path, tokens["inspect"]).inspect_job(
            created.ticket.job_id, created.capability
        )
        cross_tenant = await _expect_denial(
            _client(path, tokens["cross_tenant"]).inspect_job(
                created.ticket.job_id, created.capability
            )
        )
        wrong_capability = await _expect_denial(
            _client(path, tokens["wrong_capability"]).inspect_job(
                created.ticket.job_id, JobCapability("A" * 43)
            )
        )
        dedicated = await _client(path, tokens["create_dedicated"]).create_job(
            str(request["dedicated_idempotency_key"])
        )
        downgrade = await _expect_denial(
            _client(path, tokens["downgrade"]).inspect_job(
                dedicated.ticket.job_id, dedicated.capability
            )
        )
        return {
            "job_id": str(created.ticket.job_id),
            "capability_fingerprint": _fingerprint(created.capability),
            "replay": replay.ticket.replayed,
            "same_job": replay.ticket.job_id == created.ticket.job_id,
            "same_capability": replay.capability == created.capability,
            "inspect": inspected.snapshot.job_id == created.ticket.job_id,
            "cross_tenant": cross_tenant,
            "wrong_capability": wrong_capability,
            "dedicated_job_id": str(dedicated.ticket.job_id),
            "downgrade": downgrade,
        }
    if phase == "restart":
        replay = await _client(path, tokens["replay"]).create_job(key)
        if replay.ticket.job_id is None:
            raise RuntimeError("replayed CREATE_JOB returned no job ID")
        inspected = await _client(path, tokens["inspect"]).inspect_job(
            replay.ticket.job_id, replay.capability
        )
        return {
            "job_id": str(replay.ticket.job_id),
            "capability_fingerprint": _fingerprint(replay.capability),
            "replay": replay.ticket.replayed,
            "inspect": inspected.snapshot.job_id == replay.ticket.job_id,
        }
    raise ValueError("unsupported probe phase")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("broker", "client"))
    arguments = parser.parse_args()
    if arguments.mode == "broker":
        asyncio.run(_broker())
        return
    request = json.load(sys.stdin)
    result = asyncio.run(_client_phase(request))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
