"""Unit tests for the Phase B warm-workspace pool (directory lifecycle only).

The pool manages *directories* — no git runs here — so these tests exercise the
reuse/cold decision, the busy→ephemeral fallback, discard/eviction, capacity and
idle trimming, and the durable ``context_ref`` sink, all against the filesystem.
"""

from pathlib import Path

from openloop.tools.workspace_pool import WarmHandle, WarmWorkspacePool


def _pool(tmp_path: Path, **kw) -> WarmWorkspacePool:
    kw.setdefault("root", tmp_path / "warm")
    return WarmWorkspacePool(replica_id="r1", **kw)


async def test_warm_reuse_returns_same_directory(tmp_path):
    pool = _pool(tmp_path)

    lease1 = await pool.acquire("thread-A", "acme/x")
    assert lease1.warm is False  # first time: cold
    first_path = lease1.path
    assert first_path.exists()
    await lease1.keep()
    await lease1.release()

    lease2 = await pool.acquire("thread-A", "acme/x")
    assert lease2.warm is True  # reused
    assert lease2.path == first_path
    await lease2.keep()
    await lease2.release()


async def test_repo_mismatch_cold_starts_fresh(tmp_path):
    pool = _pool(tmp_path)
    l1 = await pool.acquire("thread-A", "acme/x")
    p1 = l1.path
    await l1.keep()
    await l1.release()

    # Same thread, different repo → the stale checkout is dropped, fresh dir.
    l2 = await pool.acquire("thread-A", "acme/y")
    assert l2.warm is False
    assert l2.path != p1
    assert not p1.exists()  # old checkout removed
    await l2.keep()
    await l2.release()


async def test_busy_key_falls_back_to_ephemeral(tmp_path):
    pool = _pool(tmp_path)
    held = await pool.acquire("thread-A", "acme/x")  # not released → busy

    parallel = await pool.acquire("thread-A", "acme/x")
    assert parallel.ephemeral is True
    assert parallel.warm is False
    assert parallel.path != held.path

    # The ephemeral checkout is removed on release; the pooled one is untouched.
    await parallel.release()
    assert not parallel.path.exists()
    await held.keep()
    await held.release()
    assert held.path.exists()


async def test_discard_forces_cold_next_time(tmp_path):
    cleared = []

    async def sink(key, ref):
        cleared.append((key, ref))

    pool = _pool(tmp_path, on_change=sink)
    l1 = await pool.acquire("thread-A", "acme/x")
    await l1.keep()
    await l1.release()
    p1 = l1.path

    # A failed attempt discards the (possibly corrupt) checkout.
    l2 = await pool.acquire("thread-A", "acme/x")
    assert l2.warm is True
    await l2.discard()
    await l2.release()
    assert not p1.exists()
    assert (("thread-A", None)) in cleared  # handle cleared on eviction

    # Next attempt cold-starts a brand new directory.
    l3 = await pool.acquire("thread-A", "acme/x")
    assert l3.warm is False
    assert l3.path != p1
    await l3.release()


async def test_unmarked_lease_is_evicted_on_release(tmp_path):
    # An exception before keep/discard leaves the lease unsettled: release must
    # evict the possibly-corrupt tree rather than keep it warm.
    pool = _pool(tmp_path)
    l1 = await pool.acquire("thread-A", "acme/x")
    path = l1.path
    await l1.release()  # never kept or discarded
    assert not path.exists()

    l2 = await pool.acquire("thread-A", "acme/x")
    assert l2.warm is False  # nothing was kept
    await l2.release()


async def test_capacity_evicts_least_recently_used(tmp_path):
    pool = _pool(tmp_path, capacity=2)
    paths = {}
    for key in ("A", "B", "C"):
        lease = await pool.acquire(key, "acme/x")
        paths[key] = lease.path
        await lease.keep()
        await lease.release()

    # A third kept checkout past capacity 2 evicts the LRU (A).
    assert not paths["A"].exists()
    assert paths["B"].exists()
    assert paths["C"].exists()


async def test_sweep_evicts_idle_entries(tmp_path):
    cleared = []

    async def sink(key, ref):
        if ref is None:
            cleared.append(key)

    pool = _pool(tmp_path, idle_seconds=0.0, on_change=sink)
    lease = await pool.acquire("thread-A", "acme/x")
    await lease.keep()
    await lease.release()
    path = lease.path

    await pool.sweep()  # idle_seconds=0 → the just-released entry is idle
    assert not path.exists()
    assert "thread-A" in cleared


async def test_shutdown_removes_all_directories(tmp_path):
    pool = _pool(tmp_path)
    paths = []
    for key in ("A", "B"):
        lease = await pool.acquire(key, "acme/x")
        paths.append(lease.path)
        await lease.keep()
        await lease.release()

    await pool.shutdown()
    assert all(not p.exists() for p in paths)


async def test_on_change_persists_handle_once(tmp_path):
    persisted = []

    async def sink(key, ref):
        persisted.append((key, ref))

    pool = _pool(tmp_path, on_change=sink)
    l1 = await pool.acquire("thread-A", "acme/x")
    await l1.keep()
    await l1.release()
    # A warm reuse + keep does not re-persist (already recorded).
    l2 = await pool.acquire("thread-A", "acme/x")
    await l2.keep()
    await l2.release()

    writes = [p for p in persisted if p[0] == "thread-A" and p[1] is not None]
    assert len(writes) == 1
    handle = WarmHandle.from_json(writes[0][1])
    assert handle.repo == "acme/x"
    assert handle.replica == "r1"


def test_warm_handle_roundtrip():
    handle = WarmHandle(workspace_id="w1", repo="acme/x", replica="r7")
    assert WarmHandle.from_json(handle.to_json()) == handle
