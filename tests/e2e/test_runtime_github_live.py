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
from openloop.memory import InMemoryStore
from openloop.models.gateway import ModelGateway
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.credentials import EnvCredentialResolver
from openloop.tools.github import GitHubConnector, HttpGitHubClient
from openloop.usage import InMemoryUsageStore, budget_scope_key

APPROVER = "@e2e-runner"


def _model() -> str | None:
    if os.environ.get("E2E_MODEL"):
        return os.environ["E2E_MODEL"]
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-3-5-haiku-latest"
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
            "apiVersion": "openloop.ai/v1alpha1",
            "kind": "Agent",
            "metadata": {"name": "e2e", "workspace": "e2e"},
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
    """Return (memory, usage, approvals) on a reachable DATABASE_URL, else None."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    try:
        import asyncpg

        from openloop.approvals.postgres import PostgresApprovalStore
        from openloop.memory.postgres import PostgresMemoryStore
        from openloop.usage.postgres import PostgresUsageStore

        conn = await asyncpg.connect(dsn, timeout=3)
        await conn.close()
        memory = PostgresMemoryStore(dsn, embedding_dim=1536)
        usage = PostgresUsageStore(dsn)
        approvals = PostgresApprovalStore(dsn)
        await memory.setup()
        await usage.setup()
        await approvals.setup()
        return memory, usage, approvals
    except Exception:
        return None


async def test_live_end_to_end():
    repo = os.environ["E2E_GITHUB_REPO"]
    token = os.environ["GITHUB_TOKEN"]
    model = _model()

    stores = await _maybe_postgres_stores()
    memory, usage, approvals = stores or (
        InMemoryStore(), InMemoryUsageStore(), InMemoryApprovalStore()
    )

    tools = ToolGateway(
        tools=[
            GitHubConnector(
                HttpGitHubClient(EnvCredentialResolver({"github": token}))
            )
        ],
        approvals=approvals,
    )
    agent = _build_agent(model)
    runtime = Runtime(
        agent, gateway=ModelGateway(), memory=memory, usage=usage, tools=tools
    )

    title = f"[openloop e2e] live check {uuid.uuid4().hex[:8]}"
    issue: dict = {}
    try:
        # 1) Real model call — it must choose the GitHub issue-creation tool.
        result = await runtime.handle(Task(
            text=(
                f"Open a GitHub issue in the repo {repo} with the exact title "
                f"'{title}' and a short body noting this is an automated "
                f"end-to-end check. Use the available GitHub tool to create it."
            ),
            surface="cli", channel=f"e2e-{title[-8:]}", user="U_e2e",
        ))
        assert result.approval_ids, f"model did not call the tool: {result.text[:200]!r}"
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
            await HttpGitHubClient(
                EnvCredentialResolver({"github": token})
            )._request(
                "PATCH", f"/repos/{repo}/issues/{number}", json={"state": "closed"}
            )
        if stores:
            for store in stores:
                await store.close()
