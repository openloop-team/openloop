"""Tests for model-policy routing — the core of provider-agnostic dispatch."""

from openloop.agents.schema import ModelPolicy, ModelRoute


def _policy():
    return ModelPolicy(
        default="openai/gpt-4o-mini",
        routes=[
            ModelRoute(match={"task": "code"}, model="anthropic/claude-sonnet-4-6"),
            ModelRoute(match={"task": "investigate"}, model="openrouter/google/gemini-2.5-pro"),
        ],
    )


def test_routes_match_task():
    p = _policy()
    assert p.resolve("code") == "anthropic/claude-sonnet-4-6"
    assert p.resolve("investigate") == "openrouter/google/gemini-2.5-pro"


def test_falls_back_to_default():
    p = _policy()
    assert p.resolve("summarize") == "openai/gpt-4o-mini"
    assert p.resolve(None) == "openai/gpt-4o-mini"
