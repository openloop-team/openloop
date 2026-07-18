"""Deterministic bounded resource controls for the broker RPC transport."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
import math
import time


def _positive_int(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} is outside its supported range")
    return value


def _positive_seconds(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


@dataclass(frozen=True, slots=True)
class BrokerRpcLimits:
    backlog: int = 64
    max_in_flight: int = 128
    per_peer_capacity: int = 32
    per_peer_refill_per_second: float = 16.0
    per_principal_capacity: int = 64
    per_principal_refill_per_second: float = 32.0
    max_rate_limit_keys: int = 4096
    prefix_timeout_seconds: float = 1.0
    body_timeout_seconds: float = 2.0
    application_timeout_seconds: float = 5.0
    audit_timeout_seconds: float = 2.0
    write_timeout_seconds: float = 2.0
    total_timeout_seconds: float = 10.0
    shutdown_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name, maximum in (
            ("backlog", 4096),
            ("max_in_flight", 65536),
            ("per_peer_capacity", 65536),
            ("per_principal_capacity", 65536),
            ("max_rate_limit_keys", 65536),
        ):
            _positive_int(name, getattr(self, name), maximum=maximum)
        for name in (
            "per_peer_refill_per_second",
            "per_principal_refill_per_second",
            "prefix_timeout_seconds",
            "body_timeout_seconds",
            "application_timeout_seconds",
            "audit_timeout_seconds",
            "write_timeout_seconds",
            "total_timeout_seconds",
            "shutdown_timeout_seconds",
        ):
            _positive_seconds(name, getattr(self, name))
        if self.total_timeout_seconds < max(
            self.prefix_timeout_seconds,
            self.body_timeout_seconds,
            self.application_timeout_seconds,
            self.write_timeout_seconds,
        ):
            raise ValueError("total timeout cannot be shorter than a phase timeout")


class InFlightLimiter:
    def __init__(self, capacity: int) -> None:
        self._capacity = _positive_int("capacity", capacity, maximum=65536)
        self._in_flight = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._in_flight >= self._capacity:
                return False
            self._in_flight += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("in-flight limiter released without a lease")
            self._in_flight -= 1


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    def __init__(
        self,
        *,
        capacity: int,
        refill_per_second: float,
        max_keys: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = float(
            _positive_int("capacity", capacity, maximum=65536)
        )
        self._refill = _positive_seconds(
            "refill_per_second", refill_per_second
        )
        self._max_keys = _positive_int("max_keys", max_keys, maximum=65536)
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._buckets: OrderedDict[Hashable, _Bucket] = OrderedDict()
        self._lock = asyncio.Lock()

    async def allow(self, key: Hashable) -> bool:
        try:
            hash(key)
        except TypeError as error:
            raise TypeError("rate-limit key must be hashable") from error
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise TypeError("rate-limit clock must return a number")
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("rate-limit clock must be finite")
        async with self._lock:
            bucket = self._buckets.pop(key, None)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._buckets.popitem(last=False)
                bucket = _Bucket(self._capacity, now)
            elif now < bucket.updated_at:
                raise ValueError("rate-limit clock moved backwards")
            else:
                bucket.tokens = min(
                    self._capacity,
                    bucket.tokens + (now - bucket.updated_at) * self._refill,
                )
                bucket.updated_at = now
            allowed = bucket.tokens >= 1.0
            if allowed:
                bucket.tokens -= 1.0
            self._buckets[key] = bucket
            return allowed

    async def key_count_for_test(self) -> int:
        async with self._lock:
            return len(self._buckets)
