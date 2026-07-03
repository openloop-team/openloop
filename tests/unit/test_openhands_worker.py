"""Unit tests for the OpenHands worker backend (SDK faked at the factory seam)."""

import asyncio

import pytest

import openloop.tools.openhands_worker as ohmod
from openloop.tools.coding_worker import WorkerState
from openloop.tools.openhands_worker import (
    PR_FILE,
    OpenHandsCodingWorker,
    OpenHandsUnavailable,
)


def _state(instruction="add retries to the fetcher"):
    return WorkerState(
        job_id="j1", repo="acme/x", instruction=instruction, base="main",
        branch="openloop/job-j1",
    )


class _Metrics:
    def __init__(self, cost, prompt_tokens, completion_tokens):
        self.accumulated_cost = cost

        class _Usage:
            pass

        usage = _Usage()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        self.accumulated_token_usage = usage


class _Stats:
    def __init__(self, metrics):
        self._metrics = metrics

    def get_combined_metrics(self):
        return self._metrics


class FakeConversation:
    """Stands in for the SDK's Conversation at the factory seam."""

    def __init__(
        self,
        workspace,
        callbacks,
        *,
        pr_text="Add retries\n\nRetries the fetcher on 5xx.",
        cost=0.2,
        prompt_tokens=100,
        completion_tokens=40,
        events=3,
    ):
        self.workspace = workspace
        self.callbacks = callbacks
        self.pr_text = pr_text
        self.events = events
        self.prompt = None
        self.teardown = []  # records close/cleanup ordering
        self.conversation_stats = _Stats(
            _Metrics(cost, prompt_tokens, completion_tokens)
        )

    @property
    def closed(self):
        return "close" in self.teardown

    def send_message(self, prompt):
        self.prompt = prompt

    def run(self):
        for _ in range(self.events):
            for cb in self.callbacks:
                cb(object())  # a chatty agent event stream
        if self.pr_text is not None:
            (self.workspace / PR_FILE).write_text(self.pr_text)

    def close(self):
        self.teardown.append("close")


def _worker(**fake_kwargs):
    created = []

    def factory(workspace, callbacks):
        conversation = FakeConversation(workspace, callbacks, **fake_kwargs)
        created.append(conversation)

        def cleanup():
            conversation.teardown.append("cleanup")

        return conversation, cleanup

    worker = OpenHandsCodingWorker("anthropic/m", conversation_factory=factory)
    return worker, created


async def test_run_parses_pr_file_and_sums_metrics(tmp_path):
    worker, created = _worker()
    state = _state()

    edit = await worker.run(tmp_path, state)

    assert edit.title == "Add retries"
    assert edit.body == "Retries the fetcher on 5xx."
    assert edit.cost_usd == 0.2
    assert edit.prompt_tokens == 100
    assert edit.completion_tokens == 40
    assert state.completed_steps == ["edit"]
    # The handoff file never reaches the commit (git add -A comes next).
    assert not (tmp_path / PR_FILE).exists()
    # Teardown reaps the runtime, in order: the conversation closes first
    # (cleanup tears down the workspace that owns close()'s HTTP client),
    # then cleanup() stops the docker agent-server container — the leak the
    # SDK's close() deliberately does not cover.
    assert created[0].teardown == ["close", "cleanup"]


async def test_prompt_carries_instruction_and_boundaries(tmp_path):
    worker, created = _worker()
    await worker.run(tmp_path, _state("rename the flag"))

    prompt = created[0].prompt
    assert "rename the flag" in prompt
    assert "acme/x" in prompt and "openloop/job-j1" in prompt
    assert "git commit" in prompt and "git push" in prompt  # the do-NOT rules
    assert PR_FILE in prompt


async def test_missing_pr_file_falls_back_to_instruction_title(tmp_path):
    worker, _ = _worker(pr_text=None)
    edit = await worker.run(tmp_path, _state("fix the flaky test in ci"))
    assert edit.title == "fix the flaky test in ci"
    assert edit.body == ""


async def test_markdown_heading_title_is_unwrapped(tmp_path):
    worker, _ = _worker(pr_text="# Fix the bug\nbody line")
    edit = await worker.run(tmp_path, _state())
    assert edit.title == "Fix the bug"
    assert edit.body == "body line"


async def test_events_heartbeat_into_on_step_throttled(tmp_path):
    """Agent events stream progress into the checkpoint callback, throttled:
    many quick events → one heartbeat, plus the final edit step."""
    worker, _ = _worker(events=25)
    state = _state()
    calls = []

    async def on_step(s):
        calls.append(list(s.completed_steps))

    await worker.run(tmp_path, state, on_step)
    await asyncio.sleep(0)  # flush heartbeats scheduled from the worker thread

    assert len(calls) == 2  # 1 throttled heartbeat + 1 "edit" step
    assert calls[-1] == ["edit"]


async def test_worker_holds_no_git_credential(tmp_path):
    """Phase 2 contract carried into Phase 4: the heavy worker receives only
    the prepared workspace — nothing credential-shaped in its state or its
    conversation inputs."""
    worker, created = _worker()
    await worker.run(tmp_path, _state())

    assert "token" not in repr(vars(worker)).lower()
    assert not hasattr(worker, "_credentials")
    # The conversation got exactly the workspace + callbacks — no resolver,
    # no token argument, and no token-shaped content in the prompt.
    assert "x-access-token" not in created[0].prompt
    assert created[0].workspace == tmp_path


async def test_metrics_api_drift_fails_the_attempt(tmp_path):
    """No defensive zeros: unreadable metrics must fail the run, not report
    $0 spend and waltz past the fail-closed cap."""

    class NoStatsConversation(FakeConversation):
        @property
        def conversation_stats(self):
            raise AttributeError("SDK moved")

        @conversation_stats.setter
        def conversation_stats(self, value):
            pass

    worker = OpenHandsCodingWorker(
        "m",
        conversation_factory=lambda ws, cbs: (
            NoStatsConversation(ws, cbs), lambda: None
        ),
    )
    with pytest.raises(AttributeError):
        await worker.run(tmp_path, _state())


async def test_failed_run_still_reaps_the_runtime(tmp_path):
    """A crashed agent run must not leak the docker agent-server container:
    close + cleanup run even when run() raises."""

    class BoomConversation(FakeConversation):
        def run(self):
            raise RuntimeError("agent crashed")

    torn_down = []

    def factory(ws, cbs):
        conversation = BoomConversation(ws, cbs)
        conversation.teardown = torn_down
        return conversation, lambda: torn_down.append("cleanup")

    worker = OpenHandsCodingWorker("m", conversation_factory=factory)
    with pytest.raises(RuntimeError, match="agent crashed"):
        await worker.run(tmp_path, _state())
    assert torn_down == ["close", "cleanup"]


def test_probe_fails_closed_without_the_sdk():
    try:
        import openhands.sdk  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("openhands extra installed — probe would pass")
    with pytest.raises(OpenHandsUnavailable, match="openhands"):
        OpenHandsCodingWorker("m").probe()


def test_defaults_are_the_safe_ones():
    worker = OpenHandsCodingWorker("m")
    assert worker.docker is False
    assert worker.max_iterations == 100
    assert worker.server_image == ohmod._DEFAULT_SERVER_IMAGE
