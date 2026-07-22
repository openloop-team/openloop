"""Bounded one-request-per-connection AF_UNIX broker RPC server."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
import errno
import math
import os
from pathlib import Path
import socket
import stat
import struct
import time
from typing import TypeVar

from .application import BrokerRpcApplication
from .codec import MAX_RPC_FRAME_BYTES, decode_request, encode_response
from .errors import RpcErrorCode, RpcFailure, RpcProtocolProblem
from .limits import BrokerRpcLimits, InFlightLimiter, TokenBucketLimiter
from .models import RPC_VERSION, RpcRequest, RpcResponse
from .peer import PeerCredentialProblem, PeerCredentialProvider


_T = TypeVar("_T")
MAX_UNIX_SOCKET_PATH_BYTES = 100


class SocketPathProblem(Exception):
    def __init__(self) -> None:
        super().__init__("broker RPC socket path rejected")


def take_over_stale_socket(
    path: Path,
    *,
    expected_uid: int,
    connect_timeout: float = 1.0,
) -> None:
    """Remove an unambiguously stale broker socket owned by ``expected_uid``.

    An existing listener is never displaced.  Only ``ECONNREFUSED`` proves that
    the socket inode has no listener; timeouts and every other connect failure
    remain ambiguous and fail closed.  The inode is re-read immediately before
    unlink to narrow the path-replacement window; the trusted, broker-owned
    parent directory is what excludes an untrusted replacer during the
    irreducible interval before unlink.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not path.is_absolute():
        raise ValueError("socket path must be absolute")
    if (
        isinstance(expected_uid, bool)
        or not isinstance(expected_uid, int)
        or expected_uid < 0
    ):
        raise ValueError("expected_uid must be a nonnegative integer")
    try:
        timeout = float(connect_timeout)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "connect_timeout must be a positive finite number"
        ) from error
    if (
        isinstance(connect_timeout, bool)
        or not isinstance(connect_timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("connect_timeout must be a positive finite number")

    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SocketPathProblem() from error

    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(timeout)
        try:
            probe.connect(os.fspath(path))
        except TimeoutError as error:
            raise SocketPathProblem() from error
        except OSError as error:
            if error.errno != errno.ECONNREFUSED:
                raise SocketPathProblem() from error
        else:
            # A successful connect proves that another broker is listening.
            raise SocketPathProblem()
    finally:
        probe.close()

    if not stat.S_ISSOCK(observed.st_mode) or observed.st_uid != expected_uid:
        raise SocketPathProblem()
    try:
        current = os.lstat(path)
    except OSError as error:
        raise SocketPathProblem() from error
    if (
        not stat.S_ISSOCK(current.st_mode)
        or current.st_uid != expected_uid
        or (current.st_dev, current.st_ino) != (observed.st_dev, observed.st_ino)
    ):
        raise SocketPathProblem()
    try:
        os.unlink(path)
    except OSError as error:
        raise SocketPathProblem() from error


@dataclass(frozen=True, slots=True)
class UnixSocketPolicy:
    path: Path
    mode: int = 0o660
    gid: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")
        if not self.path.is_absolute():
            raise ValueError("socket path must be absolute")
        try:
            encoded = os.fsencode(self.path)
        except (TypeError, UnicodeError) as error:
            raise ValueError("socket path is not encodable") from error
        if (
            not encoded
            or b"\x00" in encoded
            or len(encoded) > MAX_UNIX_SOCKET_PATH_BYTES
        ):
            raise ValueError("socket path is invalid or too long")
        if isinstance(self.mode, bool) or not isinstance(self.mode, int):
            raise TypeError("socket mode must be an integer")
        if (
            self.mode & ~0o770
            or self.mode & 0o007
            or self.mode & 0o600 != 0o600
        ):
            raise ValueError("socket mode must be owner-rw and deny world access")
        if self.gid is not None and (
            isinstance(self.gid, bool)
            or not isinstance(self.gid, int)
            or self.gid < 0
        ):
            raise ValueError("socket gid must be a nonnegative integer")


class BrokerRpcServer:
    def __init__(
        self,
        *,
        application: BrokerRpcApplication,
        socket_policy: UnixSocketPolicy,
        peer_provider: PeerCredentialProvider,
        limits: BrokerRpcLimits = BrokerRpcLimits(),
        monotonic_clock=time.monotonic,
    ) -> None:
        if not isinstance(application, BrokerRpcApplication):
            raise TypeError("application must be BrokerRpcApplication")
        if not isinstance(socket_policy, UnixSocketPolicy):
            raise TypeError("socket_policy must be UnixSocketPolicy")
        if not isinstance(peer_provider, PeerCredentialProvider):
            raise TypeError("peer_provider must implement PeerCredentialProvider")
        if not isinstance(limits, BrokerRpcLimits):
            raise TypeError("limits must be BrokerRpcLimits")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._application = application
        self._socket_policy = socket_policy
        self._peer_provider = peer_provider
        self._limits = limits
        self._clock = monotonic_clock
        self._in_flight = InFlightLimiter(limits.max_in_flight)
        self._peer_limiter = TokenBucketLimiter(
            capacity=limits.per_peer_capacity,
            refill_per_second=limits.per_peer_refill_per_second,
            max_keys=limits.max_rate_limit_keys,
            clock=monotonic_clock,
        )
        self._principal_limiter = TokenBucketLimiter(
            capacity=limits.per_principal_capacity,
            refill_per_second=limits.per_principal_refill_per_second,
            max_keys=limits.max_rate_limit_keys,
            clock=monotonic_clock,
        )
        self._server: asyncio.AbstractServer | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._application_tasks: set[asyncio.Task[RpcResponse]] = set()

    @property
    def path(self) -> Path:
        return self._socket_policy.path

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("monotonic clock returned a non-number")
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError("monotonic clock returned a non-finite value")
        return value

    def _remaining_timeout(self, phase: float, deadline: float) -> float:
        remaining = deadline - self._now()
        if remaining <= 0:
            raise TimeoutError
        return min(phase, remaining)

    async def _phase(
        self, awaitable: Awaitable[_T], phase: float, deadline: float
    ) -> _T:
        timeout = self._remaining_timeout(phase, deadline)
        async with asyncio.timeout(timeout):
            return await awaitable

    def _validate_parent(self) -> None:
        try:
            parent = os.lstat(self.path.parent)
        except OSError as error:
            raise SocketPathProblem() from error
        if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
            raise SocketPathProblem()

    def _path_absent(self) -> None:
        try:
            os.lstat(self.path)
        except FileNotFoundError:
            return
        except OSError as error:
            raise SocketPathProblem() from error
        raise SocketPathProblem()

    def _unlink_owned_socket(self) -> None:
        identity = self._socket_identity
        if identity is None:
            return
        try:
            current = os.lstat(self.path)
        except FileNotFoundError:
            return
        except OSError:
            return
        if (
            stat.S_ISSOCK(current.st_mode)
            and (current.st_dev, current.st_ino) == identity
        ):
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("broker RPC server is already started")
        self._validate_parent()
        self._path_absent()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.setblocking(False)
            listener.bind(os.fspath(self.path))
            bound = os.lstat(self.path)
            if not stat.S_ISSOCK(bound.st_mode):
                raise SocketPathProblem()
            self._socket_identity = (bound.st_dev, bound.st_ino)
            if self._socket_policy.gid is not None:
                os.chown(
                    self.path,
                    -1,
                    self._socket_policy.gid,
                    follow_symlinks=False,
                )
            os.chmod(
                self.path,
                self._socket_policy.mode,
                follow_symlinks=False,
            )
            current = os.lstat(self.path)
            if (
                not stat.S_ISSOCK(current.st_mode)
                or (current.st_dev, current.st_ino) != self._socket_identity
            ):
                raise SocketPathProblem()
            listener.listen(self._limits.backlog)
            self._server = await asyncio.start_unix_server(
                self._client_connected,
                sock=listener,
                backlog=self._limits.backlog,
                start_serving=True,
            )
        except BaseException:
            listener.close()
            self._unlink_owned_socket()
            self._socket_identity = None
            raise

    def _track(
        self, task: asyncio.Task[None], collection: set[asyncio.Task[None]]
    ) -> None:
        collection.add(task)
        task.add_done_callback(collection.discard)

    def _client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(self._serve_connection(reader, writer))
        self._track(task, self._connection_tasks)

    async def _read_request(
        self, reader: asyncio.StreamReader, deadline: float
    ) -> RpcRequest:
        prefix = await self._phase(
            reader.readexactly(4),
            self._limits.prefix_timeout_seconds,
            deadline,
        )
        length = struct.unpack(">I", prefix)[0]
        if length == 0 or length > MAX_RPC_FRAME_BYTES:
            raise RpcProtocolProblem(RpcErrorCode.MALFORMED_FRAME)
        body = await self._phase(
            reader.readexactly(length),
            self._limits.body_timeout_seconds,
            deadline,
        )
        return decode_request(prefix + body)

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        response: RpcResponse,
        deadline: float,
    ) -> None:
        writer.write(encode_response(response))
        await self._phase(
            writer.drain(),
            self._limits.write_timeout_seconds,
            deadline,
        )

    async def _finish_application(
        self, task: asyncio.Task[RpcResponse]
    ) -> None:
        try:
            await task
        except Exception:
            pass
        finally:
            await self._in_flight.release()

    async def _serve_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        acquired = False
        lease_transferred = False
        try:
            acquired = await self._in_flight.try_acquire()
            if not acquired:
                return
            connection = writer.get_extra_info("socket")
            peer = self._peer_provider.get(connection)
            if not await self._peer_limiter.allow((peer.pid, peer.uid, peer.gid)):
                return
            deadline = self._now() + self._limits.total_timeout_seconds
            request = await self._read_request(reader, deadline)
            application_task = asyncio.create_task(
                self._application.handle(
                    request,
                    peer,
                    principal_limiter=self._principal_limiter,
                )
            )
            self._application_tasks.add(application_task)
            application_task.add_done_callback(self._application_tasks.discard)
            try:
                response = await self._phase(
                    asyncio.shield(application_task),
                    self._limits.application_timeout_seconds,
                    deadline,
                )
            except TimeoutError:
                background = asyncio.create_task(
                    self._finish_application(application_task)
                )
                self._track(background, self._background_tasks)
                lease_transferred = True
                response = RpcResponse(
                    RPC_VERSION,
                    request.request_id,
                    failure=RpcFailure(RpcErrorCode.DEADLINE_EXCEEDED),
                )
            await self._write_response(writer, response, deadline)
        except (
            asyncio.IncompleteReadError,
            ConnectionError,
            OSError,
            PeerCredentialProblem,
            RpcProtocolProblem,
            TimeoutError,
        ):
            pass
        finally:
            writer.close()
            try:
                async with asyncio.timeout(0.5):
                    await writer.wait_closed()
            except (ConnectionError, OSError, TimeoutError):
                pass
            if acquired and not lease_transferred:
                await self._in_flight.release()

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        tasks = tuple(
            self._connection_tasks
            | self._background_tasks
            | self._application_tasks
        )
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._limits.shutdown_timeout_seconds,
            )
            del done
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._unlink_owned_socket()
        self._socket_identity = None

    async def __aenter__(self) -> "BrokerRpcServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.stop()
