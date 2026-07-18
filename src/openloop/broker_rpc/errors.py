"""Stable safe errors for broker control RPC."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RpcErrorCode(str, Enum):
    MALFORMED_FRAME = "MALFORMED_FRAME"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    NOT_FOUND_OR_UNAUTHORIZED = "NOT_FOUND_OR_UNAUTHORIZED"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    OVERLOADED = "OVERLOADED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    INTERNAL = "INTERNAL"


@dataclass(frozen=True, slots=True)
class RpcFailure:
    code: RpcErrorCode

    def __post_init__(self) -> None:
        if not isinstance(self.code, RpcErrorCode):
            raise TypeError("code must be RpcErrorCode")


class RpcProtocolProblem(Exception):
    def __init__(self, code: RpcErrorCode = RpcErrorCode.INVALID_REQUEST) -> None:
        if not isinstance(code, RpcErrorCode):
            raise TypeError("code must be RpcErrorCode")
        self.code = code
        super().__init__("broker RPC input rejected")

