"""Lightweight FastAPI shell over the async application composition root."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from openloop.agents import load_agents
from openloop.agents.schema import Agent
from openloop.config import get_settings
from openloop.tools import Invocation
from openloop.usage import budget_scope_key
from openloop.wiring import AppContext, compose

log = logging.getLogger("openloop")


class InvokeBody(BaseModel):
    action: str
    args: dict = {}
    requested_by: str | None = None


class ResolveBody(BaseModel):
    approver: str
    approve: bool = True


def _invocation_json(inv: Invocation) -> dict:
    return {
        "status": inv.status,
        "message": inv.message,
        "result": dataclasses.asdict(inv.result) if inv.result else None,
        "approval_id": inv.approval.id if inv.approval else None,
        # The canonical decider of a resolved approval — without it a racing
        # second HTTP resolution would silently drop the winner's identity.
        "decided_by": inv.decided_by,
    }


def _context(request: Request) -> AppContext:
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(503, "application startup is incomplete")
    return ctx


def _require_primary(request: Request) -> Agent:
    agent = _context(request).agents.primary
    if agent is None:
        raise HTTPException(404, "no agents configured")
    return agent


def create_app(*, compose_overrides: Mapping[str, Any] | None = None) -> FastAPI:
    """Create the sync ASGI shell; all store-capturing wiring happens at startup."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    agents = load_agents(settings.agents_dir)
    log.info("loaded %d agent(s): %s", len(agents), ", ".join(agents) or "none")
    slack_agent = next(
        (agent for agent in agents.values() if agent.has_slack_surface()), None
    )
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            ctx = await stack.enter_async_context(
                compose(settings, agents, overrides=compose_overrides)
            )
            app.state.ctx = ctx
            yield

    app = FastAPI(title="OpenLoop", version="0.0.1", lifespan=lifespan)

    if slack_agent and settings.slack_bot_token and settings.slack_signing_secret:

        @app.post("/slack/events")
        async def slack_events(req: Request):  # type: ignore[no-untyped-def]
            handler = _context(req).slack_handler
            if handler is None:
                raise HTTPException(503, "Slack surface startup is incomplete")
            return await handler.handle(req)

        log.info(
            "Slack HTTP events configured for agent %r", slack_agent.metadata.name
        )

    @app.post("/tools/invoke")
    async def invoke_tool(
        body: InvokeBody, request: Request
    ):  # type: ignore[no-untyped-def]
        """Run a tool action through the gateway (allowlist + approval gate)."""
        ctx = _context(request)
        inv = await ctx.tools.invoke(
            _require_primary(request),
            body.action,
            body.args,
            requested_by=body.requested_by,
        )
        return _invocation_json(inv)

    @app.get("/approvals")
    async def list_approvals(request: Request):  # type: ignore[no-untyped-def]
        agent = _require_primary(request)
        pending = await _context(request).approvals.pending(agent=agent.metadata.name)
        return [
            {
                "id": record.id,
                "action": record.action,
                "summary": record.summary,
                "approvers": record.approvers,
                "requested_by": record.requested_by,
            }
            for record in pending
        ]

    @app.post("/approvals/{request_id}/resolve")
    async def resolve_approval(
        request_id: str, body: ResolveBody, request: Request
    ):  # type: ignore[no-untyped-def]
        inv = await _context(request).tools.resolve(
            request_id, body.approver, approve=body.approve
        )
        if inv.status == "forbidden":
            raise HTTPException(403, inv.message)
        return _invocation_json(inv)

    @app.get("/usage")
    async def usage_summary(request: Request):  # type: ignore[no-untyped-def]
        """Month-to-date spend vs. budget for the primary agent."""
        agent = _require_primary(request)
        spent = await _context(request).usage.monthly_total(budget_scope_key(agent))
        budget = agent.spec.budget
        limits = agent.spec.limits
        return {
            "agent": agent.metadata.name,
            "workspace": agent.metadata.workspace,
            "month_to_date_usd": round(spent, 6),
            "monthly_budget_usd": budget.monthly_usd,
            "per_task_budget_usd": budget.per_task_usd,
            "on_exceeded": budget.on_exceeded,
            "limits": {
                "max_concurrent_tasks": limits.max_concurrent_tasks,
                "tasks_per_minute": limits.tasks_per_minute,
            },
        }

    @app.get("/audit")
    async def audit(request: Request, limit: int = 50):  # type: ignore[no-untyped-def]
        """Recent usage records — the audit trail."""
        records = await _context(request).usage.recent(limit=min(limit, 500))
        return [
            {
                "agent": record.agent,
                "channel": record.channel,
                "surface": record.surface,
                "user": record.user,
                "task_kind": record.task_kind,
                "model": record.model,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "cost_usd": record.cost_usd,
                "outcome": record.outcome,
                "created_at": record.created_at.isoformat(),
            }
            for record in records
        ]

    @app.get("/healthz")
    async def healthz(request: Request):  # type: ignore[no-untyped-def]
        ctx = _context(request)
        primary = ctx.agents.primary
        actions = ctx.tools.available_actions(primary) if primary else []
        return {
            "status": "ok",
            "agents": list(ctx.agents.loaded),
            "providers": settings.configured_providers,
            "memory": type(ctx.memory).__name__,
            "usage": type(ctx.usage).__name__,
            "tools": actions,
        }

    return app


app = create_app()
