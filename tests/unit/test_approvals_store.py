"""Unit tests for the decide-once approval store.

Pins the atomic-claim, keyset-sweep, and snapshot-isolation invariants on the
in-memory backend; the e2e Postgres module has real-SQL twins.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from openloop.approvals import ApprovalRequest, InMemoryApprovalStore


def _request(rid: str, *, created_at: datetime | None = None) -> ApprovalRequest:
    return ApprovalRequest(
        agent="a",
        action="github.issues:write",
        tool="github",
        permission="issues:write",
        args={},
        approvers=["@u"],
        summary="s",
        id=rid,
        created_at=created_at or datetime.now(timezone.utc),
    )


async def test_claim_decision_wins_exactly_once_under_contention():
    store = InMemoryApprovalStore()
    await store.create(_request("r1"))

    results = await asyncio.gather(
        store.claim_decision("r1", "@a", approve=True),
        store.claim_decision("r1", "@b", approve=True),
        store.claim_decision("r1", "@c", approve=False),
    )

    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    stored = await store.get("r1")
    assert stored.decided_by == winners[0].decided_by
    assert stored.status == winners[0].status


async def test_claim_decision_returns_none_for_decided_and_missing():
    store = InMemoryApprovalStore()
    await store.create(_request("r1"))

    assert (await store.claim_decision("r1", "@a", approve=True)) is not None
    # Already decided → None.
    assert (await store.claim_decision("r1", "@b", approve=True)) is None
    # Missing id → None.
    assert (await store.claim_decision("nope", "@b", approve=True)) is None


async def test_claim_decision_returns_the_decided_row_to_the_winner():
    store = InMemoryApprovalStore()
    await store.create(_request("r1"))

    won = await store.claim_decision("r1", "@a", approve=False)

    assert won.status == "denied"
    assert won.decided_by == "@a"


async def test_decided_unreconciled_orders_excludes_and_paginates():
    store = InMemoryApprovalStore()
    base = datetime(2026, 7, 19, tzinfo=timezone.utc)
    for i in range(5):
        await store.create(_request(f"r{i}", created_at=base + timedelta(minutes=i)))
    # Decide r0..r3; leave r4 pending. Mark r1 reconciled (excluded).
    for i in range(4):
        await store.claim_decision(f"r{i}", "@a", approve=True)
    await store.mark_reconciled("r1")

    first = await store.decided_unreconciled(limit=2)
    assert [r.id for r in first] == ["r0", "r2"]  # r1 excluded, pending r4 excluded

    cursor = (first[-1].created_at, first[-1].id)
    second = await store.decided_unreconciled(limit=2, after=cursor)
    assert [r.id for r in second] == ["r3"]  # cursor paginates to younger rows

    cursor2 = (second[-1].created_at, second[-1].id)
    assert await store.decided_unreconciled(limit=2, after=cursor2) == []


async def test_mark_reconciled_is_idempotent_and_noop_on_missing():
    store = InMemoryApprovalStore()
    await store.create(_request("r1"))
    await store.claim_decision("r1", "@a", approve=True)

    await store.mark_reconciled("r1")
    first = (await store.get("r1")).effect_at
    assert first is not None
    await store.mark_reconciled("r1")  # idempotent — effect_at unchanged
    assert (await store.get("r1")).effect_at == first
    await store.mark_reconciled("does-not-exist")  # no-op, no raise


async def test_snapshot_isolation_get_and_claim():
    store = InMemoryApprovalStore()
    await store.create(_request("r1"))

    got = await store.get("r1")
    got.status = "approved"
    got.decided_by = "@hijack"
    got.args["x"] = 1
    # Mutating the returned row must not change the store.
    assert (await store.get("r1")).status == "pending"
    assert (await store.get("r1")).decided_by is None
    assert "x" not in (await store.get("r1")).args

    won = await store.claim_decision("r1", "@a", approve=True)
    won.status = "pending"  # tamper the winner copy
    assert (await store.get("r1")).status == "approved"


async def test_create_copies_input():
    store = InMemoryApprovalStore()
    req = _request("r1")
    await store.create(req)
    req.status = "approved"  # mutate the caller's object after create
    assert (await store.get("r1")).status == "pending"
