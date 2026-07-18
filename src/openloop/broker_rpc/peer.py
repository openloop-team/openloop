"""Fail-closed Linux AF_UNIX peer credential extraction."""

from __future__ import annotations

import socket
import struct
import sys
from typing import Protocol, runtime_checkable

from .audit import PeerCredentials


class PeerCredentialProblem(Exception):
    def __init__(self) -> None:
        super().__init__("Unix peer credentials unavailable")


@runtime_checkable
class PeerCredentialProvider(Protocol):
    def get(self, connection: object) -> PeerCredentials: ...


def decode_linux_peer_credentials(value: bytes) -> PeerCredentials:
    if not isinstance(value, bytes) or len(value) != struct.calcsize("3i"):
        raise PeerCredentialProblem()
    try:
        pid, uid, gid = struct.unpack("3i", value)
        return PeerCredentials(pid, uid, gid)
    except (struct.error, TypeError, ValueError) as error:
        raise PeerCredentialProblem() from error


class LinuxPeerCredentialProvider:
    def __init__(self, *, platform: str = sys.platform) -> None:
        if platform != "linux" or not hasattr(socket, "SO_PEERCRED"):
            raise PeerCredentialProblem()

    def get(self, connection: object) -> PeerCredentials:
        getsockopt = getattr(connection, "getsockopt", None)
        if not callable(getsockopt):
            raise PeerCredentialProblem()
        try:
            value = getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                struct.calcsize("3i"),
            )
        except OSError as error:
            raise PeerCredentialProblem() from error
        return decode_linux_peer_credentials(value)


class StaticPeerCredentialProvider:
    """Deterministic provider for local development and transport tests."""

    def __init__(self, credentials: PeerCredentials) -> None:
        if not isinstance(credentials, PeerCredentials):
            raise TypeError("credentials must be PeerCredentials")
        self._credentials = credentials

    def get(self, connection: object) -> PeerCredentials:
        del connection
        return self._credentials
