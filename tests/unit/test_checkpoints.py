"""Unit tests for the in-memory worker checkpoint store."""

from openloop.checkpoints import InMemoryCheckpointStore, WorkerCheckpoint


def _cp(job_id="j1", status="running", steps=None):
    return WorkerCheckpoint(
        job_id=job_id,
        repo="a/b",
        instruction="x",
        base="main",
        branch=f"openloop/job-{job_id}",
        status=status,
        completed_steps=steps or [],
        state_json={"job_id": job_id, "completed_steps": steps or []},
    )


async def test_upsert_then_get_roundtrips():
    store = InMemoryCheckpointStore()
    await store.upsert(_cp(steps=["clone"]))
    got = await store.get("j1")
    assert got.status == "running"
    assert got.completed_steps == ["clone"]


async def test_get_missing_returns_none():
    assert await InMemoryCheckpointStore().get("nope") is None


async def test_upsert_overwrites_and_preserves_created_at():
    store = InMemoryCheckpointStore()
    await store.upsert(_cp(steps=["clone"]))
    first = await store.get("j1")

    await store.upsert(_cp(status="opened", steps=["clone", "push"]))
    second = await store.get("j1")

    assert second.status == "opened"
    assert second.completed_steps == ["clone", "push"]
    # created_at is stable across updates; updated_at advances.
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


async def test_recent_orders_by_updated_at_desc():
    store = InMemoryCheckpointStore()
    await store.upsert(_cp(job_id="a"))
    await store.upsert(_cp(job_id="b"))
    recent = await store.recent()
    assert {c.job_id for c in recent} == {"a", "b"}
    assert recent[0].job_id == "b"  # most recently upserted first
