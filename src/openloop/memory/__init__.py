"""Channel/thread memory — what an agent recalls, scoped so context doesn't leak.

The runtime talks to a :class:`MemoryStore`; backends differ (in-memory for dev
and tests, Postgres + pgvector for real deployments) but share one protocol.
"""

from openloop.memory.embeddings import Embedder, LiteLLMEmbedder
from openloop.memory.store import (
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
    scope_key_for,
)

__all__ = [
    "Embedder",
    "LiteLLMEmbedder",
    "InMemoryStore",
    "MemoryRecord",
    "MemoryStore",
    "scope_key_for",
]
