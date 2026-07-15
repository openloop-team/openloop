"""Unit tests for the OpenHands worker backend (SDK faked at the factory seam)."""

import asyncio
import base64
import time

import pytest

import openloop.tools.openhands_worker as ohmod
from openloop.tools.coding_worker import WorkerRunAborted, WorkerState
from openloop.tools.openhands_artifacts import WorkspaceArtifactStore
from openloop.tools.openhands_resume import ResumeDecision, WorkerPaused
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout
from openloop.tools.openhands_worker import (
    PR_FILE,
    _ColdRuntime,
    OpenHandsCodingWorker,
    OpenHandsUnavailable,
)


BASE = "a" * 40


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

    def factory(workspace, callbacks, job_id):
        assert job_id == "j1"
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


class GrowingCostConversation(FakeConversation):
    """Cost climbs by ``per_event`` on every agent event — for the spend guard."""

    def __init__(self, workspace, callbacks, *, per_event=0.5, **kw):
        super().__init__(workspace, callbacks, cost=0.0, **kw)
        self._per_event = per_event

    def run(self):
        for _ in range(self.events):
            self.conversation_stats._metrics.accumulated_cost += self._per_event
            for cb in self.callbacks:
                cb(object())  # the guard reads current cost and may abort here
        if self.pr_text is not None:
            (self.workspace / PR_FILE).write_text(self.pr_text)


class SleepyConversation(FakeConversation):
    """Wall-time advances between events — for the deadline guard."""

    def run(self):
        for _ in range(self.events):
            time.sleep(0.02)
            for cb in self.callbacks:
                cb(object())
        if self.pr_text is not None:
            (self.workspace / PR_FILE).write_text(self.pr_text)


def _worker_with(conv_cls, *, deadline_seconds=None, **fake_kwargs):
    created = []

    def factory(workspace, callbacks, job_id):
        assert job_id == "j1"
        conversation = conv_cls(workspace, callbacks, **fake_kwargs)
        created.append(conversation)
        return conversation, lambda: conversation.teardown.append("cleanup")

    worker = OpenHandsCodingWorker(
        "anthropic/m",
        deadline_seconds=deadline_seconds,
        conversation_factory=factory,
    )
    return worker, created


async def test_in_run_cost_abort_stops_and_reports_partial_spend(tmp_path):
    # $0.50/event, cap $1.00 → aborts on the 3rd event ($1.50), long before the
    # 10-event ($5.00) completion.
    worker, created = _worker_with(GrowingCostConversation, per_event=0.5, events=10)
    state = _state()
    state.budget_usd = 1.0

    with pytest.raises(WorkerRunAborted) as exc:
        await worker.run(tmp_path, state)

    assert exc.value.cost_usd == pytest.approx(1.5)  # stopped near the cap
    assert exc.value.prompt_tokens == 100 and exc.value.completion_tokens == 40
    assert "cap" in exc.value.reason
    assert not (tmp_path / PR_FILE).exists()  # aborted before the handoff
    assert created[0].teardown == ["close", "cleanup"]  # runtime still reaped


async def test_no_cost_abort_when_state_carries_no_budget(tmp_path):
    # budget_usd unset → the guard never trips on cost; the run completes.
    worker, _ = _worker_with(GrowingCostConversation, per_event=5.0, events=3)
    edit = await worker.run(tmp_path, _state())  # state.budget_usd is None
    assert edit.title == "Add retries"


async def test_deadline_abort_stops_a_slow_run(tmp_path):
    worker, created = _worker_with(SleepyConversation, deadline_seconds=0.01, events=10)
    with pytest.raises(WorkerRunAborted) as exc:
        await worker.run(tmp_path, _state())
    assert "deadline" in exc.value.reason
    assert created[0].teardown == ["close", "cleanup"]


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
        conversation_factory=lambda ws, cbs, job_id: (
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

    def factory(ws, cbs, job_id):
        assert job_id == "j1"
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


class _ColdAdapter:
    def stream_git_delta(self, workspace, sink, *, base_ref):
        sink.write(b"")

        class Archived:
            base_commit = base_ref

        return Archived()


class _ColdConversation(FakeConversation):
    def __init__(self, workspace, callbacks, status):
        super().__init__(workspace, callbacks, pr_text=None)
        self.state = type("State", (), {})()
        self.state.execution_status = status
        self.state.events = [type("TerminalAction", (), {"tool_name": "terminal"})()]
        self.policy = None
        self.rejection = None

    def set_confirmation_policy(self, policy):
        self.policy = policy

    def reject_pending_actions(self, reason):
        self.rejection = reason

    def run(self):
        for cb in self.callbacks:
            cb(self.state.events[-1])


def _cold_worker(tmp_path, conversation):
    layout = OpenHandsStateLayout(tmp_path / "state")
    keys = OpenHandsKeyDeriver.from_base64(
        base64.b64encode(bytes(range(32))).decode(), master_key_id="key-v1"
    )
    store = WorkspaceArtifactStore(layout, keys, scratch_root=tmp_path / "scratch")
    adapter = _ColdAdapter()
    worker = OpenHandsCodingWorker(
        "anthropic/m",
        docker=True,
        docker_adapter=adapter,
        artifact_store=store,
        cold_resume_enabled=True,
    )
    worker._git_head = lambda workspace: BASE
    worker._open_cold_runtime = lambda workspace, state, callbacks: _ColdRuntime(
        conversation(workspace, callbacks), object(), lambda: None
    )
    return worker, store


async def test_cold_run_durably_allocates_identity_and_returns_pause(tmp_path):
    created = []

    def conversation(workspace, callbacks):
        conv = _ColdConversation(workspace, callbacks, "WAITING_FOR_CONFIRMATION")
        created.append(conv)
        return conv

    worker, store = _cold_worker(tmp_path, conversation)
    state = _state()
    state.requester_id = "U123"
    checkpoints = []

    async def checkpoint(current):
        checkpoints.append(current.to_dict())

    result = await worker.run(tmp_path, state, checkpoint)

    assert isinstance(result, WorkerPaused)
    assert checkpoints[0]["openhands_resume"]["status"] == "running"
    assert state.openhands_resume.resolved_base_commit == BASE
    assert result.workspace_artifact.artifact.identity.kind == "paused"
    assert result.pending_action_summary.endswith("terminal")
    assert created[0].prompt.count("add retries") == 1
    assert created[0].policy is not None
    with store.open_verified(
        result.workspace_artifact.artifact,
        result.workspace_artifact.artifact.identity,
    ) as verified:
        assert verified.manifest.base_commit == BASE
    assert created[0].closed


async def test_cold_reject_resume_sends_no_second_prompt_and_captures_final(tmp_path):
    created = []

    def conversation(workspace, callbacks):
        conv = _ColdConversation(workspace, callbacks, "FINISHED")

        def finish():
            for cb in callbacks:
                cb(conv.state.events[-1])
            (workspace / PR_FILE).write_text("Resume complete\nBody")

        conv.run = finish
        created.append(conv)
        return conv

    worker, _ = _cold_worker(tmp_path, conversation)
    state = _state()
    state.requester_id = "U123"
    # First allocate a compatible durable identity, then model the already
    # parked state being accepted by the orchestrator as a new segment.
    state.openhands_resume = ohmod.OpenHandsResumeState(
        status="running",
        conversation_id="00000000-0000-0000-0000-000000000001",
        segment_id="segment-1",
        base_ref="main",
        resolved_base_commit=BASE,
        image_digest=worker.server_image,
        master_key_id="key-v1",
        slack_requester_id="U123",
    )
    paused_worker, _ = _cold_worker(
        tmp_path / "paused",
        lambda workspace, callbacks: _ColdConversation(
            workspace, callbacks, "WAITING_FOR_CONFIRMATION"
        ),
    )
    paused_worker.server_image = worker.server_image
    paused = await paused_worker.run(tmp_path, state)
    state.openhands_resume.transition_to(
        "parking",
        decision_id=paused.decision_id,
        pending_action_summary=paused.pending_action_summary,
        pending_action_fingerprint=paused.pending_action_fingerprint,
        workspace_artifact=paused.workspace_artifact,
        cumulative_cost=paused.cumulative_cost,
        cumulative_prompt_tokens=paused.cumulative_prompt_tokens,
        cumulative_completion_tokens=paused.cumulative_completion_tokens,
    )
    state.openhands_resume.transition_to("parked")
    decision = ResumeDecision("reject", paused.decision_id, "Ev123", "U123")
    state.openhands_resume.transition_to(
        "resuming",
        segment_id="segment-2",
        resolved_event_id=decision.event_id,
        resolved_decision=decision,
    )

    edit = await worker.run(tmp_path, state)

    assert edit.title == "Resume complete"
    assert edit.workspace_artifact.artifact.identity.kind == "final"
    assert created[0].prompt is None
    assert created[0].rejection == ohmod.OPENHANDS_REJECTION_REASON
    assert not (tmp_path / PR_FILE).exists()
