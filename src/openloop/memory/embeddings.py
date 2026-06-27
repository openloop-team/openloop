"""Embedders — turn text into vectors for semantic recall.

The default is a LiteLLM-backed embedder so it stays provider-agnostic; the
protocol lets tests swap in a deterministic fake with no network.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LiteLLMEmbedder:
    """Embeds via LiteLLM (`litellm.aembedding`)."""

    def __init__(self, model: str = "openai/text-embedding-3-small") -> None:
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Lazy import so the package loads without LiteLLM / provider keys.
        import litellm

        response = await litellm.aembedding(model=self.model, input=texts)
        # LiteLLM returns OpenAI-shaped data, ordered to match the input.
        return [item["embedding"] for item in response["data"]]
