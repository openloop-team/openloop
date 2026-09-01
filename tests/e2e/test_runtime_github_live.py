"""Automated live end-to-end test against REAL services — gated on credentials.

Drives the real chain: a real model (LiteLLM) returns a tool call, the gateway
holds it for approval, approval triggers a real GitHub issue, and persistence
is verified. The issue is CLOSED afterward, so the test is safe to re-run / use
in CI. Uses Postgres when DATABASE_URL is reachable, else in-memory.

Runs only when enabled; skips cleanly otherwise so the normal suite stays green:
  E2E_LIVE=1
  GITHUB_TOKEN, E2E_GITHUB_REPO=owner/repo
  OPENAI_API_KEY or ANTHROPIC_API_KEY
  E2E_MODEL (optional), DATABASE_URL (optional → exercises Postgres too)
"""

import os
import uuid

import pytest

from openloop.agents.schema import Agent
from openloop.approvals import InMemoryApprovalStore
from openloop.credentials import EnvCredentialResolver
from openloop.memory import InMemoryStore
from openloop.models.gateway import ModelGateway
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector, HttpGitHubClient
from openloop.usage import InMemoryUsageStore, budget_scope_key
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.workflows.postgres import PostgresWorkflowStore

APPROVER = "@e2e-runner"
# The nightly runner is one recurring principal, so its durable id is pinned
# rather than minted per run: budgets and the usage ledger are keyed on the id
# (`budget_scope_key`), and a fresh id each night would file every run under a
# new agent. Pinned literal, like the fixture in tests/e2e/data/agent.yaml.
AGENT_ID = "6e2ea40fe1594761b452c98ebd6ae7e3"


def _model() -> str | None:
    # This test exercises the live tool loop, so it needs a model that picks
    # tools reliably — mini/haiku-class models flake here.
    if os.environ.get("E2E_MODEL"):
        return os.environ["E2E_MODEL"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-6"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o"
    return None


def _missing() -> str | None:
    if os.environ.get("E2E_LIVE") != "1":
        return "set E2E_LIVE=1 to run the live end-to-end test"
    for var in ("GITHUB_TOKEN", "E2E_GITHUB_REPO"):
        if not os.environ.get(var):
            return f"{var} not set"
    if _model() is None:
        return "no model key (OPENAI_API_KEY/ANTHROPIC_API_KEY)"
    return None


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.live,
    pytest.mark.skipif(_missing() is not None, reason=_missing() or ""),
]


def _build_agent(model: str) -> Agent:
    return Agent.model_validate(
        {
            "apiVersion": "openloop.team/v1alpha1",
            "kind": "Agent",
            "metadata": {"name": "e2e", "workspace": "e2e", "id": AGENT_ID},
            "spec": {
                "model_policy": {"default": model},
                "tools": [
                    {
                        "name": "github",
                        "type": "native",
                        "permissions": ["issues:read", "issues:write"],
                    }
                ],
                "approvals": {
                    "require_for": ["github.issues:write"],
                    "approvers": [APPROVER],
                },
                "budget": {"monthly_usd": 5, "per_task_usd": 1, "on_exceeded": "warn"},
            },
        }
    )


async def _maybe_postgres_stores():
    """Return ((memory, usage, approvals, workflows), engine) when reachable."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    engine = None
    try:
        from openloop.approvals.postgres import PostgresApprovalStore
        from openloop.db import create_engine
        from openloop.memory.postgres import PostgresMemoryStore
        from openloop.usage.postgres import PostgresUsageStore

        # min_size=1 makes construction the reachability check this used to do
        # with a throwaway connection.
        engine = await create_engine(dsn, min_size=1, max_size=10)
        memory = PostgresMemoryStore(embedding_dim=1536)
        usage = PostgresUsageStore()
        approvals = PostgresApprovalStore()
        workflows = PostgresWorkflowStore()
        await memory.setup(engine)
        await usage.setup(engine)
        await approvals.setup(engine)
        await workflows.setup(engine)
        return (memory, usage, approvals, workflows), engine
    except Exception:
        if engine is not None:
            await engine.dispose()
        return None


async def test_live_end_to_end():
    repo = os.environ["E2E_GITHUB_REPO"]
    token = os.environ["GITHUB_TOKEN"]
    model = _model()

    postgres = await _maybe_postgres_stores()
    stores, engine = postgres if postgres else (None, None)
    memory, usage, approvals, workflows = stores or (
        InMemoryStore(),
        InMemoryUsageStore(),
        InMemoryApprovalStore(),
        InMemoryWorkflowStore(),
    )

    tools = ToolGateway(
        tools=[
            GitHubConnector(HttpGitHubClient(EnvCredentialResolver({"github": token})))
        ],
        approvals=approvals,
    )
    agent = _build_agent(model)
    runtime = Runtime(
        agent,
        gateway=ModelGateway(),
        memory=memory,
        usage=usage,
        tools=tools,
        engine=WorkflowEngine(workflows),
    )

    title = f"[openloop e2e] live check {uuid.uuid4().hex[:8]}"
    issue: dict = {}
    try:
        # 1) Real model call — it must choose the GitHub issue-creation tool.
        result = await runtime.handle(
            Task(
                text=(
                    f"Open a GitHub issue in the repo {repo} with the exact title "
                    f"'{title}' and a short body noting this is an automated "
                    f"end-to-end check. Use the available GitHub tool to create it."
                ),
                surface="cli",
                channel=f"e2e-{title[-8:]}",
                user="U_e2e",
            )
        )
        assert result.approval_ids, (
            f"model did not call the tool: {result.text[:200]!r}"
        )
        approval_id = result.approval_ids[0]

        # 2) Approve — performs the REAL GitHub API write.
        inv = await tools.resolve(approval_id, APPROVER, approve=True)
        assert inv.status == "executed" and inv.result and inv.result.ok, (
            f"execution failed: status={inv.status} msg={inv.message}"
        )
        issue = inv.result.data or {}
        assert issue.get("html_url") and issue.get("number")

        # 3) Persistence: approval recorded as approved, usage logged.
        stored = await approvals.get(approval_id)
        assert stored and stored.status == "approved"
        assert await usage.monthly_total(budget_scope_key(agent)) >= 0.0
        assert len(await usage.recent(limit=10)) >= 1
    finally:
        number = issue.get("number")
        if number:  # close the issue we created so runs don't accumulate junk
            await HttpGitHubClient(EnvCredentialResolver({"github": token}))._request(
                "PATCH", f"/repos/{repo}/issues/{number}", json={"state": "closed"}
            )
        if stores:
            for store in stores:
                await store.close()
        if engine is not None:
            await engine.dispose()
