"""Warm workspace pool — Phase B of persistent thread sessions.

Keeps a coding worker's git checkout alive *between turns in the same thread* so a
follow-up reuses it (``git fetch`` + reset) instead of cloning cold. This is the
(b) "warm working state" payoff: the coding path re-provisions from cold on every
turn today, throwing away the object store and working tree each time.

Design invariants (from ``docs/persistent-thread-sessions-plan.md`` §2, §8):

- **Warm is only ever a cache.** The pool is the authoritative liveness check on
  *this* replica; a replica that finds no live entry (a restart, another replica,
  an evicted entry) reconstructs cold — the always-correct fallback. Nothing is
  ever *only* answerable warm.
- **Single-replica-correct.** The pool is process-local; cross-replica affinity
  (route a follow-up to the replica holding the warm context) needs the DB
  atomic-claim arbiter and is deferred. A second replica simply cold-starts.
- **The one git-credential boundary is preserved.** The pool manages *directories*
  and their lifecycle (idle TTL, capacity, cleanup) and runs **no git itself** —
  the credential-bearing :class:`~openloop.tools.coding_worker.GitWorkspaceOrchestrator`
  does every git operation into the directory the pool hands out.

A pooled directory is keyed by the **warm-context key** — a thread's durable scope
key (:func:`~openloop.sessions.threads.thread_scope_key`), threaded down to the
worker through the approval args. One key holds at most one warm checkout; a
follow-up on the same thread reuses it when it is free, and cold-starts an
ephemeral (un-pooled) checkout when it is busy so concurrent attempts never share
a working tree.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import uuid
from asyncio import Lock
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Called after a warm entry is committed (ref = serialized handle) or evicted
# (ref = None) so the thread record's durable context_ref tracks it. Best-effort:
# the pool's own liveness is authoritative, the durable handle is a cache pointer.
ContextRefSink = Callable[[str, str | None], Awaitable[None]]


@dataclass(slots=True)
class WarmHandle:
    """The durable pointer stored as a thread's ``context_ref``.

    It records *which* replica holds *which* checkout for *which* repo — enough
    for a future affinity router to prefer that replica, and enough for an
    operator to see a thread has a warm context. It is never load-bearing for
    correctness: a replica that doesn't recognise the ``workspace_id`` (it holds
    no such live entry) just reconstructs cold.
    """

    workspace_id: str
    repo: str
    replica: str

    def to_json(self) -> str:
        return json.dumps(
            {"workspace_id": self.workspace_id, "repo": self.repo, "replica": self.replica}
        )

    @classmethod
    def from_json(cls, raw: str) -> "WarmHandle":
        d = json.loads(raw)
        return cls(workspace_id=d["workspace_id"], repo=d["repo"], replica=d["replica"])


@dataclass(slots=True)
class _Entry:
    warm_key: str
    workspace_id: str
    path: Path
    repo: str
    lock: Lock
    last_used: float
    persisted: bool = False  # whether context_ref has been written for this entry


class WarmLease:
    """A borrowed workspace for one coding attempt.

    ``path`` is the checkout directory; ``warm`` says whether it was reused (an
    existing checkout) or freshly provisioned. The orchestrator runs its git into
    ``path``, then signals the outcome:

    - :meth:`keep` on success — commit the checkout as this thread's warm context
      (and persist its handle the first time);
    - :meth:`discard` on failure — the tree may be dirty/corrupt, so evict it and
      the next turn cold-starts.

    :meth:`release` runs in ``finally`` regardless: it frees the per-key lock (for
    a pooled lease, keeping the directory) or removes the directory (for an
    ephemeral lease, which is never pooled). An unmarked lease (an exception
    before keep/discard) is treated as a discard — evicting a possibly-corrupt
    tree is always safe.
    """

    def __init__(
        self, pool: "WarmWorkspacePool", entry: _Entry, *, warm: bool, ephemeral: bool
    ) -> None:
        self._pool = pool
        self._entry = entry
        self.path = entry.path
        self.warm = warm
        self.ephemeral = ephemeral
        self._settled = False

    async def keep(self) -> None:
        self._settled = True
        if self.ephemeral:
            return
        await self._pool._commit(self._entry)

    async def discard(self) -> None:
        self._settled = True
        if self.ephemeral:
            _rmtree(self._entry.path)
            return
        await self._pool._evict_entry(self._entry)

    async def release(self) -> None:
        if self.ephemeral:
            if not self._settled:
                _rmtree(self._entry.path)
            return
        if not self._settled:
            # Never marked kept/discarded — an exception slipped past the
            # orchestrator's handling. Evict the possibly-corrupt tree.
            await self._pool._evict_entry(self._entry)
        if self._entry.lock.locked():
            self._entry.lock.release()


class WarmWorkspacePool:
    """Process-local pool of warm coding-worker checkouts, keyed by thread scope."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        idle_seconds: float = 900.0,
        capacity: int = 8,
        replica_id: str | None = None,
        on_change: ContextRefSink | None = None,
    ) -> None:
        self._root = root
        self._idle_seconds = idle_seconds
        self._capacity = max(1, capacity)
        self._replica = replica_id or uuid.uuid4().hex[:8]
        self._on_change = on_change
        self._entries: dict[str, _Entry] = {}

    def set_on_change(self, sink: ContextRefSink | None) -> None:
        """Wire (or rewire) the durable ``context_ref`` sink after construction.

        Used by the app to bridge the pool to the current thread-record store,
        selected by the composition root."""
        self._on_change = sink

    async def acquire(self, warm_key: str, repo: str) -> WarmLease:
        """Borrow a checkout for ``warm_key``.

        Returns a **warm** lease reusing the thread's live checkout when it exists,
        matches ``repo``, and is free; otherwise a **cold** lease over a freshly
        provisioned directory. If the thread's checkout is currently in use by
        another attempt, returns a cold **ephemeral** lease (an un-pooled temp dir
        removed on release) so two attempts never share one working tree.
        """
        await self._evict_idle()
        entry = self._entries.get(warm_key)

        if entry is not None and entry.lock.locked():
            # Busy — a concurrent attempt on this thread owns the warm checkout.
            # Fall back to a private cold checkout rather than block or share.
            return self._ephemeral_lease(repo)

        if (
            entry is not None
            and entry.repo == repo
            and entry.path.exists()
        ):
            await entry.lock.acquire()
            # Re-check liveness under the lock (idle sweep can't run concurrently
            # in this single-threaded loop, but the directory could have vanished).
            if entry.path.exists():
                entry.last_used = time.monotonic()
                return WarmLease(self, entry, warm=True, ephemeral=False)
            entry.lock.release()

        # Cold: stale entry (missing dir or different repo) or none at all.
        if entry is not None:
            await self._evict_entry(entry)
        new_entry = self._provision(warm_key, repo)
        await new_entry.lock.acquire()
        await self._evict_over_capacity(keep=warm_key)
        return WarmLease(self, new_entry, warm=False, ephemeral=False)

    def _provision(self, warm_key: str, repo: str) -> _Entry:
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
        workspace_id = uuid.uuid4().hex[:12]
        path = Path(
            tempfile.mkdtemp(prefix=f"openloop-warm-{workspace_id}-", dir=self._root)
        )
        entry = _Entry(
            warm_key=warm_key,
            workspace_id=workspace_id,
            path=path,
            repo=repo,
            lock=Lock(),
            last_used=time.monotonic(),
        )
        self._entries[warm_key] = entry
        return entry

    def _ephemeral_lease(self, repo: str) -> WarmLease:
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="openloop-warm-eph-", dir=self._root))
        entry = _Entry(
            warm_key="",
            workspace_id="ephemeral",
            path=path,
            repo=repo,
            lock=Lock(),
            last_used=time.monotonic(),
        )
        return WarmLease(self, entry, warm=False, ephemeral=True)

    async def _commit(self, entry: _Entry) -> None:
        entry.last_used = time.monotonic()
        if entry.persisted:
            return
        entry.persisted = True
        await self._notify(
            entry.warm_key,
            WarmHandle(entry.workspace_id, entry.repo, self._replica).to_json(),
        )

    async def _evict_entry(self, entry: _Entry) -> None:
        current = self._entries.get(entry.warm_key)
        if current is entry:
            del self._entries[entry.warm_key]
            await self._notify(entry.warm_key, None)
        _rmtree(entry.path)

    async def _evict_over_capacity(self, *, keep: str) -> None:
        # Best-effort LRU trim. Never evict a locked (in-flight) entry or the one
        # just acquired; transient over-capacity while everything is busy is fine.
        while len(self._entries) > self._capacity:
            victim = min(
                (
                    e
                    for k, e in self._entries.items()
                    if k != keep and not e.lock.locked()
                ),
                key=lambda e: e.last_used,
                default=None,
            )
            if victim is None:
                return
            await self._evict_entry(victim)

    async def _evict_idle(self) -> None:
        cutoff = time.monotonic() - self._idle_seconds
        for entry in list(self._entries.values()):
            if entry.lock.locked() or entry.last_used > cutoff:
                continue
            await self._evict_entry(entry)

    async def evict(self, warm_key: str) -> None:
        """Explicitly drop a thread's warm context (e.g. thread abandonment)."""
        entry = self._entries.get(warm_key)
        if entry is not None and not entry.lock.locked():
            await self._evict_entry(entry)

    async def sweep(self) -> None:
        """Evict idle entries. Called periodically from the app lifespan."""
        await self._evict_idle()

    async def shutdown(self) -> None:
        for entry in list(self._entries.values()):
            _rmtree(entry.path)
        self._entries.clear()

    async def _notify(self, warm_key: str, ref: str | None) -> None:
        if self._on_change is None or not warm_key:
            return
        try:
            await self._on_change(warm_key, ref)
        except Exception:  # noqa: BLE001 — durable handle is best-effort cache
            logger.warning("failed to persist warm context_ref", exc_info=True)


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
