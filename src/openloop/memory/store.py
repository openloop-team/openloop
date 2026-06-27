"""Memory records, the store protocol, scope keys, and an in-memory backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from openloop.agents.schema import Agent


@dataclass(slots=True)
class MemoryRecord:
    """One remembered item, scoped to a channel / agent / workspace."""

    scope_key: str
    text: str
    kind: str = "message"
    metadata: dict[str, str] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def scope_key_for(agent: Agent, channel: str | None) -> str:
    """Build the isolation key for an agent's memory.

    Per the README, memory can be scoped per channel so context doesn't leak
    across teams. The scope is declared in the agent's `memory.scope`.
    """
    workspace = agent.metadata.workspace
    name = agent.metadata.name
    scope = agent.spec.memory.scope
    if scope == "workspace":
        return f"ws:{workspace}"
    if scope == "agent":
        return f"ws:{workspace}:agent:{name}"
    # channel scope (default)
    return f"ws:{workspace}:agent:{name}:channel:{channel or '_'}"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence for agent memory.

    Ranking lives in the store: when a `query_embedding` is given and records
    carry embeddings, `recall` returns the most semantically similar items;
    otherwise it falls back to most-recent-first.
    """

    async def remember(self, record: MemoryRecord) -> None: ...

    async def recall(
        self,
        scope_key: str,
        query_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]: ...


class InMemoryStore:
    """Process-local store — good for dev and tests, lost on restart."""

    def __init__(self) -> None:
        self._data: dict[str, list[MemoryRecord]] = {}

    async def remember(self, record: MemoryRecord) -> None:
        self._data.setdefault(record.scope_key, []).append(record)

    async def recall(
        self,
        scope_key: str,
        query_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        records = self._data.get(scope_key, [])
        if not records:
            return []

        embedded = [r for r in records if r.embedding is not None]
        if query_embedding is not None and embedded:
            ranked = sorted(
                embedded,
                key=lambda r: cosine_similarity(query_embedding, r.embedding or []),
                reverse=True,
            )
            return ranked[:limit]

        # Fall back to most recent first.
        return sorted(records, key=lambda r: r.created_at, reverse=True)[:limit]
