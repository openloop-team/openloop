import pytest

from openloop.broker_rpc.limits import (
    BrokerRpcLimits,
    InFlightLimiter,
    TokenBucketLimiter,
)


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


async def test_in_flight_limiter_never_waits_when_full():
    limiter = InFlightLimiter(1)
    assert await limiter.try_acquire() is True
    assert await limiter.try_acquire() is False
    await limiter.release()
    assert await limiter.try_acquire() is True


async def test_token_bucket_refills_and_bounds_key_state():
    clock = Clock()
    limiter = TokenBucketLimiter(
        capacity=2,
        refill_per_second=1,
        max_keys=2,
        clock=clock,
    )
    assert await limiter.allow("a") is True
    assert await limiter.allow("a") is True
    assert await limiter.allow("a") is False
    clock.value += 1
    assert await limiter.allow("a") is True
    assert await limiter.allow("b") is True
    assert await limiter.allow("c") is True
    assert await limiter.key_count_for_test() == 2


def test_limits_reject_total_deadline_shorter_than_a_phase():
    with pytest.raises(ValueError):
        BrokerRpcLimits(total_timeout_seconds=1, application_timeout_seconds=2)
