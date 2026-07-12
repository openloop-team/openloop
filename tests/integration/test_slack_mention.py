"""Deterministic coverage of the Slack ``app_mention`` handler — no live Slack.

Phase D async delivery: a mention creates a :class:`SurfaceSession` and the
:class:`SessionRunner` sets status + posts final (or an approval card) back to
the thread via a :class:`SurfaceDelivery`. This drives :func:`handle_mention`
with a synthetic event and a :class:`FakeSurfaceDelivery` — the full glue from
event → Task → target → delivery, without a live Slack connection.
"""

from pathlib import Path
import types

import pytest

from openloop.analysis import InMemoryUploadStore
from openloop.agents import load_agent
from openloop.models.gateway import ModelResponse
from openloop.runtime import Task
from openloop.sessions import InMemorySurfaceSessionStore, SessionRunner
from openloop.surfaces.approvals import APPROVE_ACTION, approval_blocks
from openloop.sessions.store import SurfaceSession, SurfaceTarget
from openloop.surfaces.slack import _run_mention, handle_message, handle_mention
from openloop.testing import FakeGitHub, FakeSurfaceDelivery
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"

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

    The runner only touches ``runtime.handle``, ``runtime.agent`` and
    ``runtime.tools``, so a focused fake keeps the test on the glue rather than
    the pipeline.
    """

    def __init__(self, response, *, tools=None) -> None:
        self._response = response
        self.agent = load_agent(AGENT_YAML)
        self.tools = tools
        self.tasks: list[Task] = []

    async def handle(self, task: Task, *, instance_id=None):
        self.tasks.append(task)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _runner(response, *, tools=None, uploads=None):
    runtime = RecordingRuntime(response, tools=tools)
    delivery = FakeSurfaceDelivery()
    runner = SessionRunner(
        runtime, InMemorySurfaceSessionStore(), delivery, uploads=uploads
    )
    return runner, runtime, delivery


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
    runner, runtime, delivery = _runner(_reply("here you go"))
    say = FakeSay()

    await handle_mention(runner, _event("<@U0BOT> ship it"), say)

    # Mention stripped; surface/channel/user carried from the event.
    task = runtime.tasks[0]
    assert task.text == "ship it"
    assert task.surface == "slack"
    assert task.channel == "C01DEV"
    assert task.user == "U07ABC123"
    # The indicator is set, then the answer is delivered in-thread.
    assert delivery.statuses[0]["text"] == "is thinking..."
    assert delivery.finals[-1]["text"] == "here you go"
    assert delivery.finals[-1]["target"].thread == "1700000000.000100"


async def test_empty_mention_gets_help_and_skips_runtime():
    runner, runtime, delivery = _runner(_reply())
    say = FakeSay()

    await handle_mention(runner, _event("<@U0BOT>   "), say)

    assert runtime.tasks == []  # never reached the runtime
    assert delivery.statuses == []  # no thinking indicator for help text
    assert say.last["text"].startswith("Hi — mention me")


async def test_reply_threads_under_existing_thread_ts():
    runner, runtime, delivery = _runner(_reply())
    say = FakeSay()

    await handle_mention(
        runner,
        _event("<@U0BOT> follow up", thread_ts="1700000000.000050"),
        say,
    )

    # thread_ts wins over ts so replies stay in the original thread.
    assert delivery.finals[-1]["target"].thread == "1700000000.000050"


async def test_shared_file_metadata_is_recorded_and_exposed_without_its_bytes():
    uploads = InMemoryUploadStore()
    runner, runtime, _ = _runner(_reply(), uploads=uploads)

    await handle_mention(
        runner,
        _event(
            "<@U0BOT> analyze this",
            files=[
                {
                    "id": "F123",
                    "name": "sales.csv",
                    "size": 1234,
                    # Surface payloads can contain private URLs, but upload
                    # inventory must retain metadata only.
                    "url_private_download": "https://secret.invalid/file",
                }
            ],
        ),
        FakeSay(),
    )

    (record,) = await uploads.for_scope(runtime.tasks[0].thread_key)
    assert (record.upload_ref, record.name, record.size) == (
        "F123",
        "sales.csv",
        1234,
    )
    assert not hasattr(record, "url_private_download")
    notes = "\n".join(runtime.tasks[0].context_notes)
    assert "sales.csv" in notes and "upload_ref F123" in notes
    assert "secret.invalid" not in notes


async def test_runtime_failure_posts_error_not_crash():
    runner, runtime, delivery = _runner(RuntimeError("boom"))
    say = FakeSay()

    await handle_mention(runner, _event("<@U0BOT> do it"), say)

    # The runner records the failure and delivers an error notice in-thread.
    assert len(delivery.errors) == 1
    assert delivery.errors[-1]["target"].thread == "1700000000.000100"


async def test_handoff_failure_before_delivery_posts_error_in_thread():
    # A failure *before* a session exists (e.g. the session-store write) escapes
    # the runner; the background wrapper must still surface an error in-thread
    # instead of failing silently.
    class BoomRunner:
        def __init__(self):
            self.runtime = types.SimpleNamespace(agent=load_agent(AGENT_YAML))

        async def run(self, task, target):
            raise RuntimeError("session store down")

    say = FakeSay()
    await _run_mention(BoomRunner(), _event("<@U0BOT> do it"), say)

    assert say.last["text"].startswith("⚠️")
    assert say.last["thread_ts"] == "1700000000.000100"


# --- thread-reply continuation (Slice 5) --------------------------------

async def _seed_thread_session(runner, channel="C01DEV", thread="1700000000.000100"):
    # Match the runtime's agent scope so the (scope-aware) thread lookup finds it.
    md = runner.runtime.agent.metadata
    await runner.sessions.upsert(SurfaceSession(
        id="prev",
        target=SurfaceTarget(
            surface="slack", workspace=md.workspace, agent=md.name,
            channel=channel, thread=thread,
        ),
        status="completed", final_message_id="f0",
    ))


def _reply_event(text, **overrides):
    return _event(text, thread_ts="1700000000.000100", ts="1700000000.000200",
                  **overrides)


async def test_thread_reply_continues_existing_session():
    runner, runtime, delivery = _runner(_reply("follow-up answer"))
    await _seed_thread_session(runner)

    await handle_message(runner, _reply_event("a follow-up question"), FakeSay())

    assert runtime.tasks[-1].text == "a follow-up question"
    assert delivery.finals[-1]["text"] == "follow-up answer"
    assert delivery.finals[-1]["target"].thread == "1700000000.000100"


async def test_thread_reply_ignored_without_existing_session():
    runner, runtime, delivery = _runner(_reply())

    await handle_message(runner, _reply_event("just chatting"), FakeSay())

    assert runtime.tasks == []  # the bot isn't part of this thread
    assert delivery.statuses == [] and delivery.finals == []


async def test_bot_message_is_ignored():
    runner, runtime, delivery = _runner(_reply())
    await _seed_thread_session(runner)

    await handle_message(runner, _reply_event("beep", bot_id="B1"), FakeSay())

    assert runtime.tasks == []


async def test_edited_message_subtype_is_ignored():
    runner, runtime, delivery = _runner(_reply())
    await _seed_thread_session(runner)

    await handle_message(
        runner, _reply_event("edited", subtype="message_changed"), FakeSay()
    )

    assert runtime.tasks == []


async def test_bot_mention_in_thread_is_left_to_app_mention():
    runner, runtime, delivery = _runner(_reply())
    await _seed_thread_session(runner)

    # The bot itself is mentioned → app_mention owns this message.
    await handle_message(
        runner, _reply_event("<@U0BOT> hey again"), FakeSay(), bot_user_id="U0BOT"
    )

    assert runtime.tasks == []


async def test_thread_reply_mentioning_another_user_is_handled():
    # A follow-up mentioning a *different* user is not an app_mention, so it must
    # be handled here rather than silently dropped.
    runner, runtime, delivery = _runner(_reply("done"))
    await _seed_thread_session(runner)

    await handle_message(
        runner, _reply_event("can you ask <@U999> about it?"),
        FakeSay(), bot_user_id="U0BOT",
    )

    assert len(runtime.tasks) == 1
    # The referenced user is preserved in the task text, not stripped away.
    assert runtime.tasks[-1].text == "can you ask <@U999> about it?"
    assert delivery.finals[-1]["text"] == "done"


async def test_top_level_message_is_ignored():
    runner, runtime, delivery = _runner(_reply())
    await _seed_thread_session(runner)

    # No thread_ts → not a reply continuing a thread.
    await handle_message(runner, _event("hello channel"), FakeSay())

    assert runtime.tasks == []


async def test_approval_reply_renders_blocks_in_thread():
    agent = load_agent(AGENT_YAML)
    tools = ToolGateway(tools=[GitHubConnector(FakeGitHub())])
    inv = await tools.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    approval_id = inv.approval.id

    runner, runtime, delivery = _runner(
        _reply("Approval required.", approval_ids=[approval_id]), tools=tools
    )
    say = FakeSay()

    await handle_mention(runner, _event("<@U0BOT> open an issue"), say)

    # A durable approval card is posted (with the buttons), carrying the pending
    # request, in-thread.
    assert len(delivery.approvals) == 1
    card = delivery.approvals[-1]
    assert card["target"].thread == "1700000000.000100"
    assert [r.id for r in card["requests"]] == [approval_id]
    assert APPROVE_ACTION in repr(approval_blocks(card["requests"]))
    assert delivery.finals == []  # no final answer until the approval resolves
