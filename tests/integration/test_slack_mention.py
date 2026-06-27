"""Deterministic coverage of the Slack ``app_mention`` handler — no live Slack.

Drives :func:`handle_mention` with a synthetic event dict and a fake ``say``,
which is exactly what Bolt invokes when a mention arrives. This is the CI gate
for the "Slack mention → runtime → reply" path: the wire (auth, event
subscription, block rendering) is Slack's concern and lives in the gated live
smoke; the glue that turns an event into a :class:`Task` and shapes the reply
is ours, and it's fully testable here.
"""

import pytest

from openloop.agents import load_agent
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.surfaces.approvals import APPROVE_ACTION
from openloop.surfaces.slack import handle_mention
from openloop.testing import EXAMPLE_AGENT, FakeGitHub
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector

pytestmark = pytest.mark.integration


class FakeSay:
    """Records each ``say(...)`` call's kwargs, like Bolt's say utility."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> None:
        self.calls.append(kwargs)

    @property
    def last(self) -> dict:
        return self.calls[-1]


class RecordingRuntime:
    """Minimal runtime stand-in: captures Tasks, returns (or raises) a response.

    The handler only touches ``runtime.handle`` and ``runtime.tools``, so a
    focused fake keeps the test on the glue rather than the pipeline.
    """

    def __init__(self, response, *, tools=None) -> None:
        self._response = response
        self.tools = tools
        self.tasks: list[Task] = []

    async def handle(self, task: Task):
        self.tasks.append(task)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _event(text: str, **overrides) -> dict:
    event = {
        "text": text,
        "channel": "C01DEV",
        "user": "U07ABC123",
        "ts": "1700000000.000100",
    }
    event.update(overrides)
    return event


def _reply(text: str = "done", approval_ids=None) -> ModelResponse:
    return ModelResponse(
        text=text, model="m", approval_ids=list(approval_ids or [])
    )


async def test_mention_becomes_task_with_slack_identity():
    runtime = RecordingRuntime(_reply("here you go"))
    say = FakeSay()

    await handle_mention(runtime, _event("<@U0BOT> ship it"), say)

    # Mention stripped; surface/channel/user carried from the event.
    task = runtime.tasks[0]
    assert task.text == "ship it"
    assert task.surface == "slack"
    assert task.channel == "C01DEV"
    assert task.user == "U07ABC123"
    # Reply posted in-thread off the message ts.
    assert say.last["text"] == "here you go"
    assert say.last["thread_ts"] == "1700000000.000100"


async def test_empty_mention_gets_help_and_skips_runtime():
    runtime = RecordingRuntime(_reply())
    say = FakeSay()

    await handle_mention(runtime, _event("<@U0BOT>   "), say)

    assert runtime.tasks == []  # never reached the runtime
    assert say.last["text"].startswith("Hi — mention me")


async def test_reply_threads_under_existing_thread_ts():
    runtime = RecordingRuntime(_reply())
    say = FakeSay()

    await handle_mention(
        runtime,
        _event("<@U0BOT> follow up", thread_ts="1700000000.000050"),
        say,
    )

    # thread_ts wins over ts so replies stay in the original thread.
    assert say.last["thread_ts"] == "1700000000.000050"


async def test_runtime_failure_posts_error_not_crash():
    runtime = RecordingRuntime(RuntimeError("boom"))
    say = FakeSay()

    await handle_mention(runtime, _event("<@U0BOT> do it"), say)

    assert say.last["text"].startswith("⚠️")
    assert say.last["thread_ts"] == "1700000000.000100"


async def test_approval_reply_renders_blocks_in_thread():
    agent = load_agent(EXAMPLE_AGENT)
    tools = ToolGateway(tools=[GitHubConnector(FakeGitHub())])
    inv = await tools.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    approval_id = inv.approval.id

    runtime = RecordingRuntime(
        _reply("Approval required.", approval_ids=[approval_id]), tools=tools
    )
    say = FakeSay()

    await handle_mention(runtime, _event("<@U0BOT> open an issue"), say)

    # Approval path posts interactive blocks (with the approve button) in-thread.
    assert "blocks" in say.last
    serialized = repr(say.last["blocks"])
    assert APPROVE_ACTION in serialized
    assert say.last["thread_ts"] == "1700000000.000100"
