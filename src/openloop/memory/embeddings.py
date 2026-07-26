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

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        *,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Lazy import so the package loads without LiteLLM / provider keys.
        import litellm

        kwargs = {"model": self.model, "input": texts}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        response = await litellm.aembedding(**kwargs)
        # LiteLLM returns OpenAI-shaped data, ordered to match the input.
        return [item["embedding"] for item in response["data"]]
