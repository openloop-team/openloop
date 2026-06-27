"""A thin, provider-agnostic gateway over LiteLLM.

The runtime never talks to a provider SDK directly — it asks the gateway for a
completion against a LiteLLM-style model id (e.g. ``openai/gpt-4o-mini``,
``anthropic/claude-sonnet-4-6``). Model selection is the model policy's job
(see :mod:`openloop.agents.schema`); the gateway just executes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(slots=True)
class ToolCall:
    """A function call the model wants to make."""

    id: str
    name: str
    arguments: dict


@dataclass(slots=True)
class ModelResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The assistant message in OpenAI dict form, to append before tool results.
    raw_message: dict | None = None
    # IDs of write actions held for human approval this turn (if any).
    approval_ids: list[str] = field(default_factory=list)


class ModelGateway:
    """Executes completions and reports token/cost usage for the audit trail."""

    async def complete(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        # Imported lazily so importing the package (e.g. in tests) doesn't pull
        # in LiteLLM and its heavy provider dependencies.
        import litellm

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await litellm.acompletion(**kwargs)

        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        raw_tool_calls = getattr(message, "tool_calls", None) or []
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_parse_args(tc.function.arguments),
            )
            for tc in raw_tool_calls
        ]

        raw_message: dict = {"role": "assistant", "content": message.content}
        if raw_tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tool_calls
            ]

        return ModelResponse(
            text=(message.content or "").strip(),
            model=response.model or model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=float(cost or 0.0),
            tool_calls=tool_calls,
            raw_message=raw_message,
        )


def _parse_args(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
