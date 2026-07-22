"""Async settle-before-wire application composition root."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

from openloop.agents import load_agents
from openloop.agents.schema import Agent
from openloop.analysis import (
    InMemoryAnalysisAttemptStore,
    InMemoryArtifactStore,
    InMemoryInputStore,
    InMemoryUploadStore,
)
from openloop.approvals import InMemoryApprovalStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.config import Settings
from openloop.memory import InMemoryStore
from openloop.postgres import BorrowedPostgresStore, create_pool
from openloop.sessions import InMemorySurfaceSessionStore, InMemoryThreadRecordStore
from openloop.surfaces.slack import build_slack_app
from openloop.usage import InMemoryTaskLimiter, InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.wiring import builders
from openloop.wiring.broker import build_broker
from openloop.wiring.context import AgentRuntimes, AppContext, SettledStores

log = logging.getLogger("openloop")

_STORE_KEYS = {
    "memory",
    "usage",
    "approvals",
    "checkpoints",
    "workflows",
    "sessions",
    "threads",
    "analysis_inputs",
    "analysis_artifacts",
    "analysis_attempts",
    "analysis_uploads",
}
_LEAF_KEYS = _STORE_KEYS | {
    "embedder",
    "limiter",
    "model_gateway",
    "coordinator",
    "pool_factory",
    "tools_factory",
    "broker_handle",
}


async def _close_safely(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        close = getattr(resource, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # noqa: BLE001 - shutdown must release remaining resources
        log.warning("failed to close application resource", exc_info=True)


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _validate_overrides(overrides: Mapping[str, Any]) -> None:
    unknown = set(overrides) - _LEAF_KEYS
    if "tools" in overrides:
        raise TypeError(
            "compose overrides accept tools_factory(settled), not a prebuilt "
            "ToolGateway"
        )
    if unknown:
        raise TypeError(f"unknown compose override(s): {', '.join(sorted(unknown))}")


def _override_or(
    overrides: Mapping[str, Any], key: str, factory: Callable[[], Any]
) -> Any:
    return overrides[key] if key in overrides else factory()


async def _settle_store(
    stack: AsyncExitStack,
    candidate: Any,
    fallback: Callable[[], Any],
    *,
    pool: Any | None,
    mode: str,
    label: str,
    post_setup: Callable[[Any], Any] | None = None,
) -> Any:
    """Return one final store, registering cleanup before any dependent exists."""
    if not isinstance(candidate, BorrowedPostgresStore):
        if hasattr(candidate, "close") or hasattr(candidate, "aclose"):
            stack.push_async_callback(_close_safely, candidate)
        log.info("%s backend: in-memory (process-local)", label)
        return candidate

    if pool is None:
        return fallback()

    try:
        await candidate.setup(pool)
        if post_setup is not None:
            await post_setup(candidate)
    except Exception:
        await _close_safely(candidate)
        if mode == "postgres":
            log.exception("postgres %s setup failed", label)
            raise
        log.exception(
            "postgres %s setup failed — using process-local fallback", label
        )
        return fallback()

    stack.push_async_callback(_close_safely, candidate)
    backend = "postgres (pgvector)" if label == "memory" else "postgres"
    log.info("%s backend: %s", label, backend)
    return candidate


async def _reset_thread_claims(store: Any) -> None:
    cleared = await store.reset_active_claims()
    if cleared:
        log.info("cleared %d stale thread claim(s) at startup", cleared)


@asynccontextmanager
async def compose(
    settings: Settings,
    agents: dict[str, Agent] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> AsyncIterator[AppContext]:
    """Resolve backends, wire dependents once, and own their async lifecycle."""
    selected = dict(overrides or {})
    _validate_overrides(selected)
    loaded_agents = agents if agents is not None else load_agents(settings.agents_dir)
    mode = settings.effective_storage_mode

    async with AsyncExitStack() as stack:
        pool = None
        if mode in ("auto", "postgres"):
            pool_factory = selected.get("pool_factory", create_pool)
            try:
                pool = await pool_factory(
                    settings.database_url,
                    min_size=settings.postgres_pool_min_size,
                    max_size=settings.postgres_pool_max_size,
                )
            except Exception:
                if mode == "postgres":
                    log.exception("postgres shared pool setup failed")
                    raise
                log.exception(
                    "postgres shared pool setup failed — durable stores will "
                    "use process-local fallbacks"
                )
            else:
                stack.push_async_callback(_close_safely, pool)
                log.info(
                    "postgres shared pool ready (min=%d, max=%d)",
                    settings.postgres_pool_min_size,
                    settings.postgres_pool_max_size,
                )

        memory = await _settle_store(
            stack,
            _override_or(
                selected, "memory", lambda: builders.build_memory_store(settings)
            ),
            InMemoryStore,
            pool=pool,
            mode=mode,
            label="memory",
        )
        usage = await _settle_store(
            stack,
            _override_or(
                selected, "usage", lambda: builders.build_usage_store(settings)
            ),
            InMemoryUsageStore,
            pool=pool,
            mode=mode,
            label="usage",
        )
        approvals = await _settle_store(
            stack,
            _override_or(
                selected, "approvals", lambda: builders.build_approval_store(settings)
            ),
            InMemoryApprovalStore,
            pool=pool,
            mode=mode,
            label="approval",
        )
        checkpoints = await _settle_store(
            stack,
            _override_or(
                selected,
                "checkpoints",
                lambda: builders.build_checkpoint_store(settings),
            ),
            InMemoryCheckpointStore,
            pool=pool,
            mode=mode,
            label="checkpoint",
        )
        workflows = await _settle_store(
            stack,
            _override_or(
                selected, "workflows", lambda: builders.build_workflow_store(settings)
            ),
            InMemoryWorkflowStore,
            pool=pool,
            mode=mode,
            label="workflow",
        )
        sessions = await _settle_store(
            stack,
            _override_or(
                selected,
                "sessions",
                lambda: builders.build_surface_session_store(settings),
            ),
            InMemorySurfaceSessionStore,
            pool=pool,
            mode=mode,
            label="surface-session",
        )
        threads = await _settle_store(
            stack,
            _override_or(
                selected, "threads", lambda: builders.build_thread_record_store(settings)
            ),
            InMemoryThreadRecordStore,
            pool=pool,
            mode=mode,
            label="thread-record",
            post_setup=_reset_thread_claims,
        )
        analysis_inputs = await _settle_store(
            stack,
            _override_or(
                selected,
                "analysis_inputs",
                lambda: builders.build_analysis_input_store(settings),
            ),
            InMemoryInputStore,
            pool=pool,
            mode=mode,
            label="analysis input",
        )
        analysis_artifacts = await _settle_store(
            stack,
            _override_or(
                selected,
                "analysis_artifacts",
                lambda: builders.build_analysis_artifact_store(settings),
            ),
            InMemoryArtifactStore,
            pool=pool,
            mode=mode,
            label="analysis artifact",
        )
        analysis_attempts = await _settle_store(
            stack,
            _override_or(
                selected,
                "analysis_attempts",
                lambda: builders.build_analysis_attempt_store(settings),
            ),
            InMemoryAnalysisAttemptStore,
            pool=pool,
            mode=mode,
            label="analysis attempt",
        )
        analysis_uploads = await _settle_store(
            stack,
            _override_or(
                selected,
                "analysis_uploads",
                lambda: builders.build_analysis_upload_store(settings),
            ),
            InMemoryUploadStore,
            pool=pool,
            mode=mode,
            label="analysis upload",
        )
        stores = SettledStores(
            memory=memory,
            usage=usage,
            approvals=approvals,
            checkpoints=checkpoints,
            workflows=workflows,
            sessions=sessions,
            threads=threads,
            analysis_inputs=analysis_inputs,
            analysis_artifacts=analysis_artifacts,
            analysis_attempts=analysis_attempts,
            analysis_uploads=analysis_uploads,
        )

        coordinator_candidate = _override_or(
            selected, "coordinator", lambda: builders.build_lock(settings)
        )
        coordinator = await builders._setup_coordination(
            coordinator_candidate, settings
        )
        if hasattr(coordinator, "close") or hasattr(coordinator, "aclose"):
            stack.push_async_callback(_close_safely, coordinator)

        embedder = _override_or(
            selected, "embedder", lambda: builders.build_embedder(settings)
        )
        limiter = _override_or(selected, "limiter", InMemoryTaskLimiter)
        engine = WorkflowEngine(stores.workflows)
        # Compose the configured broker client behind its flag; build_broker
        # dispatches between the co-process graph and the external client-only
        # graph, and owns any app-side teardown on `stack`. Fail-closed: if the
        # flag is on but the broker cannot be built, the handle stays None and
        # build_coding_worker disables the worker rather than falling back to the
        # direct launch path.
        broker_handle = selected.get("broker_handle")
        if broker_handle is None and settings.coding_worker_openhands_broker_enabled:
            broker_handle = await build_broker(settings, stack, pool=pool)

        tools_factory = selected.get("tools_factory")
        if tools_factory is None:
            tools = builders.build_tool_gateway(
                settings,
                loaded_agents,
                stores.approvals,
                stores.checkpoints,
                engine,
                usage=stores.usage,
                analysis_inputs=stores.analysis_inputs,
                analysis_artifacts=stores.analysis_artifacts,
                analysis_attempts=stores.analysis_attempts,
                analysis_uploads=stores.analysis_uploads,
                broker_handle=broker_handle,
            )
        else:
            tools = tools_factory(stores)

        warm_pool = getattr(tools, "warm_pool", None)
        if warm_pool is not None:
            stack.push_async_callback(warm_pool.shutdown)

            async def persist_context_ref(warm_key: str, ref: str | None) -> None:
                await stores.threads.set_context_ref(warm_key, ref)

            warm_pool.set_on_change(persist_context_ref)

        runtimes = AgentRuntimes(
            loaded=loaded_agents,
            stores=stores,
            embedder=embedder,
            tools=tools,
            engine=engine,
            limiter=limiter,
            model_gateway=selected.get("model_gateway"),
        )
        slack_app: AsyncApp | None = None
        session_runner = None
        slack_handler = None
        if runtimes.slack_agent is not None and settings.slack_bot_token:
            runtime = runtimes.slack_runtime()
            assert runtime is not None
            slack_app = build_slack_app(
                runtime,
                stores.sessions,
                bot_token=settings.slack_bot_token,
                signing_secret=settings.slack_signing_secret or None,
                threads=stores.threads,
                artifacts=stores.analysis_artifacts,
                uploads=stores.analysis_uploads,
            )
            session_runner = getattr(slack_app, "_session_runner", None)
            if settings.slack_signing_secret:
                slack_handler = AsyncSlackRequestHandler(slack_app)
        else:
            log.warning(
                "Slack surface not bound: need a Slack-enabled agent and "
                "SLACK_BOT_TOKEN"
            )

        broker_reconciler = (
            getattr(broker_handle, "reconciler", None)
            if broker_handle is not None
            else None
        )
        await builders.run_recovery_pass(
            coordinator,
            tools,
            session_runner,
            broker_reconciler=broker_reconciler,
        )
        recovery_task = None
        if settings.recovery_interval_seconds > 0:
            recovery_task = asyncio.create_task(
                builders._recovery_loop(
                    coordinator,
                    tools,
                    session_runner,
                    interval=settings.recovery_interval_seconds,
                    broker_reconciler=broker_reconciler,
                )
            )
            stack.push_async_callback(_cancel_task, recovery_task)

        warm_sweep_task = None
        if warm_pool is not None:
            warm_sweep_task = asyncio.create_task(
                builders._warm_sweep_loop(
                    warm_pool,
                    interval=settings.coding_worker_warm_idle_seconds,
                )
            )
            stack.push_async_callback(_cancel_task, warm_sweep_task)

        for connector in getattr(tools, "mcp_connectors", []):
            try:
                await connector.setup()
            except Exception:
                log.exception(
                    "MCP connector %r setup failed — its tools stay unavailable",
                    connector.name,
                )

        yield AppContext(
            settings=settings,
            agents=runtimes,
            stores=stores,
            embedder=embedder,
            limiter=limiter,
            engine=engine,
            tools=tools,
            coordinator=coordinator,
            slack_app=slack_app,
            session_runner=session_runner,
            slack_handler=slack_handler,
            postgres_pool=pool,
            recovery_task=recovery_task,
            warm_sweep_task=warm_sweep_task,
        )
