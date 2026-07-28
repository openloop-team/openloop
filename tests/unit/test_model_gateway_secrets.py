"""Provider credentials loaded by RuntimeSettings reach each LiteLLM call explicitly."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from openloop.config import RuntimeSettings
from openloop.memory.embeddings import LiteLLMEmbedder
from openloop.models.gateway import ModelGateway
from openloop.wiring.builders import build_embedder, build_model_gateway


class _Completion:
    model = "openai/test"
    usage = None
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None),
        )
    ]


async def test_model_gateway_passes_the_matching_provider_key(monkeypatch):
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return _Completion()

    fake_litellm = SimpleNamespace(
        acompletion=acompletion,
        completion_cost=lambda **kwargs: 0,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    gateway = ModelGateway(
        {"openai": "mounted-openai-secret", "anthropic": "other-secret"}
    )
    await gateway.complete("openai/test", [{"role": "user", "content": "hello"}])

    assert calls[0]["api_key"] == "mounted-openai-secret"


async def test_bare_model_gateway_leaves_litellm_environment_fallback_intact(
    monkeypatch,
):
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return _Completion()

    fake_litellm = SimpleNamespace(
        acompletion=acompletion,
        completion_cost=lambda **kwargs: 0,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    await ModelGateway().complete(
        "openai/test",
        [{"role": "user", "content": "hello"}],
    )

    assert "api_key" not in calls[0]


async def test_embedder_passes_its_provider_key(monkeypatch):
    calls = []

    async def aembedding(**kwargs):
        calls.append(kwargs)
        return {"data": [{"embedding": [1.0, 2.0]}]}

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(aembedding=aembedding),
    )

    embedder = LiteLLMEmbedder("openai/embed", api_key="mounted-openai-secret")

    assert await embedder.embed(["hello"]) == [[1.0, 2.0]]
    assert calls[0]["api_key"] == "mounted-openai-secret"


def test_builders_transfer_a_mounted_key_to_model_clients(tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "openai_api_key").write_text("mounted-openai-secret\n")
    settings = RuntimeSettings(_env_file=None, _secrets_dir=secrets_dir)

    gateway = build_model_gateway(settings)
    embedder = build_embedder(settings)

    assert gateway._provider_api_keys == {"openai": "mounted-openai-secret"}
    assert isinstance(embedder, LiteLLMEmbedder)
    assert embedder._api_key == "mounted-openai-secret"
