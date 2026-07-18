"""Typed one-shot client for the two broker RPC v1 operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import inspect
import math
import os
from pathlib import Path
import struct
from uuid import UUID, uuid4

from .capability import JobCapability
from .codec import (
    MAX_RPC_FRAME_BYTES,
    decode_response,
    encode_request,
)
from .errors import RpcErrorCode, RpcProtocolProblem
from .identity import WorkloadIdentityToken, WorkloadIntent
from .models import (
    RPC_VERSION,
    CreateJobPayload,
    CreateJobResult,
    InspectJobPayload,
    InspectJobResult,
    RpcRequest,
    RpcResponse,
)


IdentityProvider = Callable[
    [WorkloadIntent], WorkloadIdentityToken | Awaitable[WorkloadIdentityToken]
]


class BrokerRpcClientProblem(Exception):
    def __init__(self) -> None:
        super().__init__("broker RPC transport failed")


class BrokerRpcRemoteError(BrokerRpcClientProblem):
    def __init__(self, code: RpcErrorCode) -> None:
        if not isinstance(code, RpcErrorCode):
            raise TypeError("code must be RpcErrorCode")
        self.code = code
        super().__init__()


class BrokerRpcClient:
    def __init__(
        self,
        *,
        path: Path,
        identity_provider: IdentityProvider,
        request_id_factory: Callable[[], UUID] = uuid4,
        connect_timeout_seconds: float = 2.0,
        io_timeout_seconds: float = 5.0,
        total_timeout_seconds: float = 10.0,
    ) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("path must be an absolute pathlib.Path")
        if not callable(identity_provider) or not callable(request_id_factory):
            raise TypeError("identity and request ID providers must be callable")
        for name, value in (
            ("connect_timeout_seconds", connect_timeout_seconds),
            ("io_timeout_seconds", io_timeout_seconds),
            ("total_timeout_seconds", total_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self._path = path
        self._identity_provider = identity_provider
        self._request_id_factory = request_id_factory
        self._connect_timeout = float(connect_timeout_seconds)
        self._io_timeout = float(io_timeout_seconds)
        self._total_timeout = float(total_timeout_seconds)

    async def _identity(self, intent: WorkloadIntent) -> WorkloadIdentityToken:
        value = self._identity_provider(intent)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, WorkloadIdentityToken):
            raise BrokerRpcClientProblem()
        return value

    async def _exchange(self, request: RpcRequest) -> RpcResponse:
        writer: asyncio.StreamWriter | None = None
        try:
            async with asyncio.timeout(self._total_timeout):
                async with asyncio.timeout(self._connect_timeout):
                    reader, writer = await asyncio.open_unix_connection(
                        os.fspath(self._path)
                    )
                writer.write(encode_request(request))
                async with asyncio.timeout(self._io_timeout):
                    await writer.drain()
                    prefix = await reader.readexactly(4)
                length = struct.unpack(">I", prefix)[0]
                if length == 0 or length > MAX_RPC_FRAME_BYTES:
                    raise BrokerRpcClientProblem()
                async with asyncio.timeout(self._io_timeout):
                    body = await reader.readexactly(length)
                response = decode_response(prefix + body)
                if response.request_id != request.request_id:
                    raise BrokerRpcClientProblem()
                return response
        except BrokerRpcClientProblem:
            raise
        except (
            asyncio.IncompleteReadError,
            ConnectionError,
            OSError,
            RpcProtocolProblem,
            TimeoutError,
            ValueError,
        ) as error:
            raise BrokerRpcClientProblem() from error
        finally:
            if writer is not None:
                writer.close()
                try:
                    async with asyncio.timeout(0.5):
                        await writer.wait_closed()
                except (ConnectionError, OSError, TimeoutError):
                    pass

    @staticmethod
    def _result(response: RpcResponse):
        if response.failure is not None:
            raise BrokerRpcRemoteError(response.failure.code)
        assert response.result is not None
        return response.result

    async def create_job(self, idempotency_key: str) -> CreateJobResult:
        request = RpcRequest(
            RPC_VERSION,
            self._request_id_factory(),
            WorkloadIntent.CREATE_JOB,
            await self._identity(WorkloadIntent.CREATE_JOB),
            None,
            CreateJobPayload(idempotency_key),
        )
        result = self._result(await self._exchange(request))
        if not isinstance(result, CreateJobResult):
            raise BrokerRpcClientProblem()
        return result

    async def inspect_job(
        self, job_id: UUID, capability: JobCapability
    ) -> InspectJobResult:
        request = RpcRequest(
            RPC_VERSION,
            self._request_id_factory(),
            WorkloadIntent.INSPECT_JOB,
            await self._identity(WorkloadIntent.INSPECT_JOB),
            capability,
            InspectJobPayload(job_id),
        )
        result = self._result(await self._exchange(request))
        if not isinstance(result, InspectJobResult):
            raise BrokerRpcClientProblem()
        return result
