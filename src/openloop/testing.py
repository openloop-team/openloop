"""Shared test helpers and lightweight fakes.

These helpers are intentionally network-free and database-free. They are used
by OpenLoop's own tests and can also support downstream integration tests.
"""

from pathlib import Path

from openloop.memory.embeddings import Embedder
from openloop.models.gateway import ModelResponse, ToolCall

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_AGENT = REPO_ROOT / "agents" / "dev-platform.yaml"


def tool_call_response(model: str, calls: list[tuple[str, str, dict]]) -> ModelResponse:
    """Build a ModelResponse that asks for tool calls.

    `calls` is a list of (call_id, function_name, arguments).
    """
    import json

    tool_calls = [ToolCall(id=cid, name=name, arguments=args)
                  for cid, name, args in calls]
    raw_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for cid, name, args in calls
        ],
    }
    return ModelResponse(text="", model=model, tool_calls=tool_calls,
                         raw_message=raw_message)


class ScriptedGateway:
    """Returns a pre-scripted sequence of ModelResponses, recording each call."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, model, messages, **kwargs) -> ModelResponse:
        self.calls.append(
            {"model": model, "messages": list(messages), "tools": kwargs.get("tools")}
        )
        return self._responses.pop(0)


class FakeGateway:
    """Records the messages it was called with and returns a canned reply."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.last_messages: list[dict[str, str]] | None = None
        self.last_model: str | None = None

    async def complete(self, model, messages, **kwargs) -> ModelResponse:
        self.last_model = model
        self.last_messages = messages
        return ModelResponse(text=self.reply, model=model)


class FakeGitHub:
    """In-memory GitHub client that records calls without using the network."""

    def __init__(self) -> None:
        self.created: list[dict] = []

    async def create_issue(self, repo, title, body):
        issue = {"number": len(self.created) + 1, "repo": repo, "title": title}
        self.created.append(issue)
        return issue

    async def get_issue(self, repo, number):
        return {"number": number, "repo": repo, "state": "open"}

    async def get_pull(self, repo, number):
        return {"number": number, "repo": repo, "state": "open"}


class FakeEmbedder(Embedder):
    """Deterministic 26-dim bag-of-letters embedding without network calls."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str) -> list[float]:
        vec = [0.0] * 26
        for ch in text.lower():
            idx = ord(ch) - 97
            if 0 <= idx < 26:
                vec[idx] += 1.0
        return vec
