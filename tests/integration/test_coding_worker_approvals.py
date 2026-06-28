"""Integration: the coding worker through the approval gate and tool loop.

Mirrors test_tools_approvals.py / test_tool_loop.py but for the multi-step
``coding_worker.pr:write`` action: approve-before-work, then a draft PR appears.
"""

from openloop.agents import load_agent
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.coding_worker import CodingWorkerConnector
from openloop.tools.github import GitHubConnector
from openloop.testing import (
    EXAMPLE_AGENT,
    FakeCodingWorker,
    FakeGitHub,
    ScriptedGateway,
    tool_call_response,
)


def _agent():
    return load_agent(EXAMPLE_AGENT)


def _gateway(worker=None, github=None):
    github = github or FakeGitHub()
    return ToolGateway(
        tools=[
            GitHubConnector(github),
            CodingWorkerConnector(worker or FakeCodingWorker(), github),
        ]
    )


def _task(text="ship it"):
    return Task(text=text, surface="slack", channel="#dev-platform", user="U1")


async def test_coding_worker_is_held_for_approval():
    agent = _agent()
    github = FakeGitHub()
    gw = _gateway(github=github)

    inv = await gw.invoke(
        agent,
        "coding_worker.pr:write",
        {"repo": "acme/x", "instruction": "add retries"},
        requested_by="U1",
    )

    assert inv.status == "pending_approval"
    assert inv.result is None  # worker has not run yet
    # The summary states "run worker + open draft PR" — never implies diff review.
    assert inv.approval.summary.startswith("run coding worker + open draft PR")
    assert "diff" not in inv.approval.summary
    # job_id is minted before approval and persisted in the request args.
    assert inv.approval.args.get("job_id")
    assert github.pulls == []


async def test_approve_runs_worker_and_opens_draft_pr():
    agent = _agent()
    worker = FakeCodingWorker(title="Add retries")
    github = FakeGitHub()
    gw = _gateway(worker, github)

    pending = await gw.invoke(
        agent, "coding_worker.pr:write", {"repo": "acme/x", "instruction": "add retries"}
    )
    job_id = pending.approval.args["job_id"]

    resolved = await gw.resolve(pending.approval.id, "@maciag.artur", approve=True)

    assert resolved.status == "executed"
    assert resolved.result.ok
    # The same job_id flowed from approval → execution → PR branch.
    assert resolved.result.data["job_id"] == job_id
    assert github.pulls[0]["draft"] is True
    assert github.pulls[0]["head"] == f"openloop/job-{job_id}"
    assert worker.runs[0].job_id == job_id


async def test_denied_approval_never_runs_worker():
    agent = _agent()
    worker = FakeCodingWorker()
    github = FakeGitHub()
    gw = _gateway(worker, github)

    pending = await gw.invoke(
        agent, "coding_worker.pr:write", {"repo": "acme/x", "instruction": "x"}
    )
    inv = await gw.resolve(pending.approval.id, "@maciag.artur", approve=False)

    assert inv.status == "denied"
    assert worker.runs == []
    assert github.pulls == []


async def test_tool_loop_holds_coding_worker_for_approval():
    agent = _agent()
    github = FakeGitHub()
    gw = _gateway(github=github)
    model = ScriptedGateway([
        tool_call_response(
            "m",
            [("c1", "coding_worker_pr_write",
              {"repo": "acme/x", "instruction": "add retries"})],
        ),
    ])
    runtime = Runtime(agent, gateway=model, tools=gw)

    result = await runtime.handle(_task("open a PR adding retries"))

    assert result.model == "approval-gate"
    assert "approval required" in result.text.lower()
    assert github.pulls == []  # not executed
    assert await gw.approvals.pending(agent="dev-platform")
