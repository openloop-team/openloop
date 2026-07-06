"""Unit tests for the Claude Code worker backend (the `claude` CLI faked at the
runner seam, so no real binary is ever spawned)."""

import json

import pytest

from openloop.tools.claude_worker import (
    PR_FILE,
    ClaudeCodeCodingWorker,
    ClaudeCodeUnavailable,
)
from openloop.tools.coding_worker import WorkerRunAborted, WorkerState


def _state(instruction="add retries to the fetcher"):
    return WorkerState(
        job_id="j1", repo="acme/x", instruction=instruction, base="main",
        branch="openloop/job-j1",
    )


def _result_json(
    *, cost=0.12, input_tokens=100, output_tokens=40, is_error=False, subtype="success"
):
    return json.dumps(
        {
            "type": "result",
            "subtype": subtype,
            "is_error": is_error,
            "result": "done",
            "session_id": "s1",
            "total_cost_usd": cost,
            "num_turns": 3,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


def _worker(
    *,
    pr_text="Add retries\n\nRetries the fetcher on 5xx.",
    stdout=None,
    rc=0,
    stderr="",
    timed_out=False,
    **worker_kwargs,
):
    """Build a worker whose runner writes the PR file and returns canned output."""
    calls = []

    async def runner(cmd, cwd, timeout):
        calls.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
        if not timed_out and pr_text is not None:
            (cwd / PR_FILE).write_text(pr_text)
        out = _result_json() if stdout is None else stdout
        return (rc, "" if timed_out else out, stderr, timed_out)

    worker = ClaudeCodeCodingWorker(
        "anthropic/claude-sonnet-4-6", runner=runner, **worker_kwargs
    )
    return worker, calls


async def test_run_parses_pr_file_and_reports_metrics(tmp_path):
    worker, _ = _worker()
    state = _state()

    edit = await worker.run(tmp_path, state)

    assert edit.title == "Add retries"
    assert edit.body == "Retries the fetcher on 5xx."
    assert edit.cost_usd == pytest.approx(0.12)
    assert edit.prompt_tokens == 100
    assert edit.completion_tokens == 40
    assert state.completed_steps == ["edit"]
    # The handoff file never reaches the commit (git add -A comes next).
    assert not (tmp_path / PR_FILE).exists()


async def test_command_carries_headless_flags_and_stripped_model(tmp_path):
    worker, calls = _worker(max_turns=42)
    await worker.run(tmp_path, _state())

    cmd = calls[0]["cmd"]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--max-turns") + 1] == "42"
    # provider prefix stripped — the CLI names Claude models directly.
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-6"
    assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
    # The deadline is handed to the runner as the kill timeout.
    assert calls[0]["timeout"] == 600.0


async def test_prompt_carries_instruction_and_boundaries(tmp_path):
    worker, calls = _worker()
    await worker.run(tmp_path, _state("rename the flag"))

    prompt = calls[0]["cmd"][calls[0]["cmd"].index("-p") + 1]
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


async def test_zero_cost_estimate_still_ships_the_edit(tmp_path):
    """The C stance: under a subscription total_cost_usd is often 0. The run was
    still bounded by turns + deadline, so a within-cap edit must ship — the cost
    signal being 0 is not a reason to fail."""
    worker, _ = _worker(stdout=_result_json(cost=0.0, input_tokens=0, output_tokens=0))
    edit = await worker.run(tmp_path, _state())
    assert edit.title == "Add retries"
    assert edit.cost_usd == 0.0


async def test_deadline_abort_fails_closed(tmp_path):
    worker, _ = _worker(timed_out=True, deadline_seconds=0.01)
    with pytest.raises(WorkerRunAborted) as exc:
        await worker.run(tmp_path, _state())
    assert "deadline" in exc.value.reason
    assert not (tmp_path / PR_FILE).exists()  # nothing shipped


async def test_nonzero_exit_fails_the_attempt(tmp_path):
    worker, _ = _worker(rc=1, stderr="boom", pr_text=None)
    with pytest.raises(RuntimeError, match="exited 1"):
        await worker.run(tmp_path, _state())


async def test_error_result_fails_the_attempt(tmp_path):
    worker, _ = _worker(stdout=_result_json(is_error=True, subtype="error_max_turns"))
    with pytest.raises(RuntimeError, match="error_max_turns"):
        await worker.run(tmp_path, _state())


async def test_unparseable_output_fails_rather_than_reporting_zero(tmp_path):
    # No defensive zeros: garbage output must fail, not silently report $0 spend
    # and no change.
    worker, _ = _worker(stdout="not json at all", pr_text=None)
    with pytest.raises(RuntimeError, match="could not parse"):
        await worker.run(tmp_path, _state())


async def test_worker_holds_no_git_credential(tmp_path):
    worker, calls = _worker()
    await worker.run(tmp_path, _state())

    assert "token" not in repr(vars(worker)).lower()
    assert not hasattr(worker, "_credentials")
    prompt = calls[0]["cmd"][calls[0]["cmd"].index("-p") + 1]
    assert "x-access-token" not in prompt


def test_probe_fails_closed_without_the_cli(monkeypatch):
    monkeypatch.setattr("openloop.tools.claude_worker.shutil.which", lambda _: None)
    with pytest.raises(ClaudeCodeUnavailable, match="not found on PATH"):
        ClaudeCodeCodingWorker("m").probe()


def test_probe_requires_a_bounded_run(monkeypatch):
    # The CLI exists, but an unbounded run has no fail-closed ceiling.
    monkeypatch.setattr(
        "openloop.tools.claude_worker.shutil.which", lambda _: "/usr/bin/claude"
    )
    with pytest.raises(ClaudeCodeUnavailable, match="deadline"):
        ClaudeCodeCodingWorker("m", deadline_seconds=0).probe()
    with pytest.raises(ClaudeCodeUnavailable, match="turn cap"):
        ClaudeCodeCodingWorker("m", max_turns=0).probe()


def test_defaults_are_the_safe_ones():
    worker = ClaudeCodeCodingWorker("m")
    assert worker.claude_bin == "claude"
    assert worker.max_turns == 100
    assert worker.deadline_seconds == 600.0
    assert worker.permission_mode == "acceptEdits"
