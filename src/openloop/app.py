"""FastAPI application — the runtime's HTTP entrypoint.

Loads agents from config-as-code, wires the first agent that exposes a Slack
surface to the Slack events endpoint, sets up channel memory, and exposes a
health check. Run with:

    uvicorn openloop.app:app --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from openloop.analysis import (
    AnalysisAttemptStore,
    ArtifactStore,
    GithubProvisioner,
    InMemoryAnalysisAttemptStore,
    InMemoryArtifactStore,
    InMemoryInputStore,
    InMemoryUploadStore,
    InputStore,
    StagedProvisioner,
    UploadProvisioner,
    UploadStore,
)
from openloop.analysis.postgres import (
    PostgresAnalysisAttemptStore,
    PostgresArtifactStore,
    PostgresInputStore,
    PostgresUploadStore,
)
from openloop.agents import load_agents
from openloop.agents.schema import Agent
from openloop.approvals import ApprovalStore, InMemoryApprovalStore
from openloop.approvals.postgres import PostgresApprovalStore
from openloop.checkpoints import CheckpointStore, InMemoryCheckpointStore
from openloop.checkpoints.postgres import PostgresCheckpointStore
from openloop.config import Settings, get_settings
from openloop.credentials import (
    CredentialResolver,
    CredentialScope,
    EnvCredentialResolver,
    GitHubAppResolver,
)
from openloop.coordination import (
    DistributedLock,
    InProcessLock,
    PostgresLock,
    RedisLock,
    guard,
)
from openloop.memory import Embedder, InMemoryStore, LiteLLMEmbedder, MemoryStore
from openloop.memory.postgres import PostgresMemoryStore
from openloop.runtime import Runtime
from openloop.sessions import (
    InMemorySurfaceSessionStore,
    InMemoryThreadRecordStore,
    SurfaceSessionStore,
    ThreadRecordStore,
)
from openloop.sessions.postgres import PostgresSurfaceSessionStore
from openloop.sessions.threads import PostgresThreadRecordStore
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from openloop.surfaces.slack import build_slack_app
from openloop.tools import Invocation, ToolGateway
from openloop.sandbox import (
    DockerSandbox,
    HostSandbox,
    Sandbox,
    SandboxLimits,
    SandboxUnavailable,
    sweep_expired_sandboxes,
)
from openloop.tools.analysis_worker import (
    ANALYSIS_REPORT_WRITE,
    AnalysisWorker,
    AnalysisWorkerConnector,
    BuiltinAnalysisWorker,
    SealedAnalysisOrchestrator,
)
from openloop.tools.coding_worker import (
    CodingWorker,
    CodingWorkerConnector,
    CODING_WORKER_PR_WRITE,
    BuiltinCodingWorker,
    GitWorkspaceOrchestrator,
)
from openloop.tools.github import GitHubConnector, HttpGitHubClient
from openloop.tools.mcp import HttpMCPClient, MCPConnector
from openloop.tools.workspace_pool import WarmWorkspacePool
from openloop.tools.openhands_worker import (
    OpenHandsCodingWorker,
    OpenHandsUnavailable,
)
from openloop.usage import (
    InMemoryTaskLimiter,
    InMemoryUsageStore,
    UsageStore,
    WorkerSpendLedger,
    budget_scope_key,
)
from openloop.usage.postgres import PostgresUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine, WorkflowStore
from openloop.workflows.analysis_worker import build_analysis_worker_workflow
from openloop.workflows.coding_worker import build_coding_worker_workflow
from openloop.workflows.postgres import PostgresWorkflowStore

log = logging.getLogger("openloop")


def build_embedder(settings: Settings) -> Embedder | None:
    """Build an embedder only if enabled and its provider key is configured."""
    if not settings.embeddings_enabled:
        return None
    if settings.embedding_provider not in settings.configured_providers:
        log.warning(
            "embeddings disabled: no API key for provider %r — "
            "memory will use recency-only recall",
            settings.embedding_provider,
        )
        return None
    return LiteLLMEmbedder(settings.embedding_model)


def build_memory_store(settings: Settings) -> MemoryStore:
    """Pick a memory backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresMemoryStore(
            settings.database_url, embedding_dim=settings.embedding_dim
        )
    return InMemoryStore()


def build_usage_store(settings: Settings) -> UsageStore:
    """Pick a usage/audit backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresUsageStore(settings.database_url)
    return InMemoryUsageStore()


def build_approval_store(settings: Settings) -> ApprovalStore:
    """Pick an approval backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresApprovalStore(settings.database_url)
    return InMemoryApprovalStore()


def build_checkpoint_store(settings: Settings) -> CheckpointStore:
    """Pick a worker-checkpoint backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresCheckpointStore(settings.database_url)
    return InMemoryCheckpointStore()


def build_workflow_store(settings: Settings) -> WorkflowStore:
    """Pick a workflow-instance backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresWorkflowStore(settings.database_url)
    return InMemoryWorkflowStore()


def build_surface_session_store(settings: Settings) -> SurfaceSessionStore:
    """Pick a surface-session backend (Phase D). Postgres setup at startup."""
    if settings.memory_backend == "postgres":
        return PostgresSurfaceSessionStore(settings.database_url)
    return InMemorySurfaceSessionStore()


def build_thread_record_store(settings: Settings) -> ThreadRecordStore:
    """Pick a thread-record backend (Phase A). Postgres setup at startup."""
    if settings.memory_backend == "postgres":
        return PostgresThreadRecordStore(settings.database_url)
    return InMemoryThreadRecordStore()


def build_analysis_input_store(settings: Settings) -> InputStore:
    """Pick the capability-ref-keyed sealed-analysis staging store."""
    if settings.memory_backend == "postgres":
        return PostgresInputStore(settings.database_url)
    return InMemoryInputStore()


def build_analysis_artifact_store(settings: Settings) -> ArtifactStore:
    """Pick the job-keyed sealed-analysis artifact store."""
    if settings.memory_backend == "postgres":
        return PostgresArtifactStore(settings.database_url)
    return InMemoryArtifactStore()


def build_analysis_attempt_store(settings: Settings) -> AnalysisAttemptStore:
    """Pick the durable spend-attempt state store for sealed analysis."""
    if settings.memory_backend == "postgres":
        return PostgresAnalysisAttemptStore(settings.database_url)
    return InMemoryAnalysisAttemptStore()


def build_analysis_upload_store(settings: Settings) -> UploadStore:
    """Pick the thread-scoped surface-upload metadata store (Phase 4)."""
    if settings.memory_backend == "postgres":
        return PostgresUploadStore(settings.database_url)
    return InMemoryUploadStore()


def _resolve_lock_backend(settings: Settings) -> str:
    """Resolve ``lock_backend``, expanding ``auto`` to follow ``memory_backend``."""
    backend = settings.lock_backend
    if backend == "auto":
        # A Postgres deploy already has the shared dependency advisory locks need,
        # so a multi-replica Postgres deploy gets coordination without extra infra;
        # otherwise there's nothing to coordinate against → process-local.
        return "postgres" if settings.memory_backend == "postgres" else "memory"
    return backend


def build_lock(settings: Settings) -> DistributedLock:
    """Pick a coordination backend; its store/connection is set up at startup.

    ``auto`` (default) follows ``memory_backend``. ``postgres`` reuses the existing
    database; ``redis`` needs the optional ``redis`` extra (a missing package
    degrades to in-process here, a connectivity failure degrades at startup).
    """
    backend = _resolve_lock_backend(settings)
    if backend == "postgres":
        return PostgresLock(settings.database_url)
    if backend == "redis":
        try:
            return RedisLock.from_url(settings.redis_url)
        except Exception:
            log.exception(
                "redis lock unavailable (is the `redis` extra installed?) — "
                "falling back to in-process coordination"
            )
    return InProcessLock()


_COORD_LABEL = {RedisLock: "redis", PostgresLock: "postgres"}


async def _setup_coordination(
    coordinator: DistributedLock, settings: Settings
) -> DistributedLock:
    """Set up the coordination backend, degrading to in-process on failure.

    A backend the operator *explicitly* asked for (``lock_backend`` postgres/redis)
    is a deliberate request for cross-process coordination, so a setup failure is
    logged loudly — silently running process-local locks across replicas is the
    footgun this feature exists to remove. An ``auto``-selected backend degrades
    quietly (consistent with the other stores' "degrade, don't fail boot" posture).
    """
    setup = getattr(coordinator, "setup", None)
    if setup is None:  # InProcessLock — nothing to start
        log.info("coordination backend: in-process (single-replica)")
        return coordinator
    try:
        await setup()
        log.info(
            "coordination backend: %s",
            _COORD_LABEL.get(type(coordinator), type(coordinator).__name__),
        )
        return coordinator
    except Exception:
        if settings.lock_backend in ("postgres", "redis"):
            log.error(
                "CROSS-PROCESS COORDINATION DISABLED: LOCK_BACKEND=%s could not "
                "start; multiple replicas may run recovery concurrently. Falling "
                "back to a process-local lock.",
                settings.lock_backend, exc_info=True,
            )
        else:
            log.exception(
                "coordination backend setup failed — falling back to in-process"
            )
        await _safe_close(coordinator)
        return InProcessLock()


def build_github_credentials(settings: Settings) -> CredentialResolver | None:
    """Pick the GitHub auth backend behind the Phase 1 resolver seam.

    Prefers GitHub App installation tokens (short-lived, minted on demand) when
    all three ``GITHUB_APP_*`` values are set; a static ``GITHUB_TOKEN`` is the
    fallback. An App that is configured but can't start (unreadable key file,
    missing ``githubapp`` extra) was an *explicit* request for short-lived auth,
    so it fails loudly, then degrades to the token if one is set — mirroring
    the lock backend's explicit-vs-auto policy.
    """
    if settings.github_app_configured:
        try:
            import jwt

            private_key = Path(settings.github_app_private_key_path).read_text()
            # Prove the WHOLE signing path at boot, not just importability:
            # PyJWT without its crypto backend, or a malformed key, would
            # otherwise pass this gate and fail on the first tool call — after
            # the App resolver was already selected over the token fallback.
            jwt.encode({"iss": "boot-check"}, private_key, algorithm="RS256")
        except Exception:
            log.error(
                "GITHUB APP AUTH DISABLED: could not sign with the App private "
                "key (is the `githubapp` extra installed, and the key a valid "
                "RSA PEM?)%s",
                " — falling back to GITHUB_TOKEN" if settings.github_token else "",
                exc_info=True,
            )
        else:
            log.info(
                "github auth: App installation tokens (app %s)",
                settings.github_app_id,
            )
            return GitHubAppResolver(
                settings.github_app_id,
                private_key,
                settings.github_app_installation_id,
                repositories=settings.github_app_repository_list or None,
            )
    if settings.github_token:
        log.info("github auth: static token from env")
        return EnvCredentialResolver({"github": settings.github_token})
    return None


def _worker_workspace_root(settings: Settings) -> Path | None:
    """The one place the attempt-workspace root is derived from settings.

    Shared by the sandbox probe and the orchestrator so what boot verifies is
    exactly what attempts later use.
    """
    if settings.coding_worker_workspace_dir:
        return Path(settings.coding_worker_workspace_dir)
    return None


def _analysis_workspace_root(settings: Settings) -> Path | None:
    """The host-visible root used for sealed analysis workspaces and its probe."""
    if settings.analysis_worker_workspace_dir:
        return Path(settings.analysis_worker_workspace_dir)
    return None


def build_worker_sandbox(settings: Settings) -> "Sandbox | None":
    """Pick where the coding worker's model-influenced execution runs.

    ``host`` returns the no-isolation default. ``docker`` runs a full-fidelity
    probe at boot — a real container run with the configured image, network,
    and a bind mount under the configured workspace root — and returns
    ``None`` when any of it can't work; the caller then disables the coding
    worker entirely (fail-closed) rather than silently running model-generated
    edits on the host, or registering a worker that would only fail after a
    human approved a job. An unknown value also returns ``None``: a typo in a
    security setting must not select a weaker boundary.
    """
    if settings.coding_worker_sandbox == "host":
        return HostSandbox()
    if settings.coding_worker_sandbox == "docker":
        sandbox = DockerSandbox(
            settings.coding_worker_sandbox_image,
            network=settings.coding_worker_sandbox_network,
        )
        try:
            sandbox.probe(workspace_root=_worker_workspace_root(settings))
        except SandboxUnavailable:
            log.error("docker sandbox probe failed", exc_info=True)
            return None
        return sandbox
    log.error(
        "unknown CODING_WORKER_SANDBOX=%r (expected host|docker)",
        settings.coding_worker_sandbox,
    )
    return None


def build_analysis_sandbox(settings: Settings) -> "DockerSandbox | None":
    """Build the analysis sandbox, rejecting every weaker or mutable variant.

    Model-authored analysis is arbitrary Python, so a host fallback would be
    controller RCE. The image must be digest-pinned and the only network mode is
    ``none``; all other configuration failures disable the worker before anyone
    can approve a job.
    """
    if settings.analysis_worker_sandbox != "docker":
        log.error(
            "ANALYSIS WORKER DISABLED: ANALYSIS_WORKER_SANDBOX=%r; only docker "
            "is allowed for model-authored analysis (host is forbidden)",
            settings.analysis_worker_sandbox,
        )
        return None
    if settings.analysis_worker_sandbox_network != "none":
        log.error(
            "ANALYSIS WORKER DISABLED: ANALYSIS_WORKER_SANDBOX_NETWORK=%r; "
            "sealed analysis requires network=none",
            settings.analysis_worker_sandbox_network,
        )
        return None
    image = settings.analysis_worker_sandbox_image
    if not image or "@sha256:" not in image:
        log.error(
            "ANALYSIS WORKER DISABLED: ANALYSIS_WORKER_SANDBOX_IMAGE must be "
            "a digest-pinned image reference (…@sha256:…)",
        )
        return None
    sandbox = DockerSandbox(image, network="none", kind="analysis")
    try:
        sandbox.probe_sealed(workspace_root=_analysis_workspace_root(settings))
    except SandboxUnavailable:
        log.error("sealed analysis sandbox probe failed", exc_info=True)
        return None
    return sandbox


def build_warm_workspace_pool(settings: Settings) -> "WarmWorkspacePool | None":
    """The process-local warm-workspace pool (Phase B), or None when disabled.

    Off unless ``CODING_WORKER_WARM_CONTEXT`` is set. Shares the attempt-workspace
    root so warm checkouts live where the sandbox can reach them (same constraint
    as the ephemeral path). The durable ``context_ref`` sink is wired later, once
    the thread-record store exists (and stays repoint-safe on a PG fallback)."""
    if not settings.coding_worker_warm_context:
        return None
    return WarmWorkspacePool(
        root=_worker_workspace_root(settings),
        idle_seconds=settings.coding_worker_warm_idle_seconds,
        capacity=settings.coding_worker_warm_capacity,
    )


def _provider_key(settings: Settings, model: str) -> str | None:
    """The configured API key for a LiteLLM-style model's provider prefix."""
    provider = model.split("/", 1)[0]
    return {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.gemini_api_key,
        "openrouter": settings.openrouter_api_key,
    }.get(provider)


def build_coding_worker(settings: Settings) -> "CodingWorker | None":
    """Pick the worker backend behind ``CODING_WORKER_BACKEND`` — fail-closed.

    Returns ``None`` when the requested backend can't run safely (missing
    extra, unusable docker, or a typo'd value); the caller then disables the
    coding worker loudly rather than degrading to a different worker or a
    weaker isolation boundary.
    """
    backend = settings.coding_worker_backend
    if backend == "builtin":
        sandbox = build_worker_sandbox(settings)
        if sandbox is None:
            return None
        return BuiltinCodingWorker(model=settings.coding_worker_model, sandbox=sandbox)
    if backend == "openhands":
        if settings.coding_worker_sandbox not in ("host", "docker"):
            log.error(
                "unknown CODING_WORKER_SANDBOX=%r (expected host|docker)",
                settings.coding_worker_sandbox,
            )
            return None
        worker = OpenHandsCodingWorker(
            settings.coding_worker_model,
            api_key=_provider_key(settings, settings.coding_worker_model),
            max_iterations=settings.coding_worker_max_iterations,
            deadline_seconds=settings.coding_worker_deadline_seconds or None,
            docker=settings.coding_worker_sandbox == "docker",
            server_image=settings.coding_worker_openhands_image,
            network=settings.coding_worker_openhands_network,
        )
        try:
            worker.probe()
        except OpenHandsUnavailable:
            log.error("openhands backend probe failed", exc_info=True)
            return None
        return worker
    log.error(
        "unknown CODING_WORKER_BACKEND=%r (expected builtin|openhands)", backend
    )
    return None


def build_analysis_worker(settings: Settings) -> "AnalysisWorker | None":
    """Build the one builtin analysis backend, fail-closed on every mismatch."""
    if settings.analysis_worker_backend != "builtin":
        log.error(
            "unknown ANALYSIS_WORKER_BACKEND=%r (expected builtin)",
            settings.analysis_worker_backend,
        )
        return None
    if settings.analysis_worker_strategy not in ("single", "iterative"):
        log.error(
            "unknown ANALYSIS_WORKER_STRATEGY=%r (expected single|iterative)",
            settings.analysis_worker_strategy,
        )
        return None
    if (
        settings.analysis_worker_timeout_seconds <= 0
        or settings.analysis_worker_kill_after_seconds <= 0
        or settings.analysis_worker_report_max_bytes <= 0
        or settings.analysis_worker_stream_cap_bytes <= 0
        or settings.analysis_worker_output_watch_interval_seconds <= 0
        or settings.analysis_worker_summary_lines <= 0
        or settings.analysis_worker_pids_limit <= 0
        or settings.analysis_worker_cpus <= 0
        or settings.analysis_worker_max_iterations <= 0
        or settings.analysis_worker_exec_feedback_max_chars <= 0
        or settings.analysis_worker_max_input_bytes <= 0
        or settings.analysis_worker_github_max_bytes <= 0
        or settings.analysis_worker_upload_max_bytes <= 0
        or not settings.analysis_worker_memory
        or not settings.analysis_worker_tmp_size
    ):
        log.error("ANALYSIS WORKER DISABLED: sealed resource limits must be positive")
        return None
    sandbox = build_analysis_sandbox(settings)
    if sandbox is None:
        return None
    return BuiltinAnalysisWorker(
        settings.analysis_worker_model,
        sandbox,
        limits=SandboxLimits(
            timeout_seconds=settings.analysis_worker_timeout_seconds,
            kill_after_seconds=settings.analysis_worker_kill_after_seconds,
            memory=settings.analysis_worker_memory,
            memory_swap=settings.analysis_worker_memory_swap,
            cpus=settings.analysis_worker_cpus,
            pids_limit=settings.analysis_worker_pids_limit,
            tmp_size=settings.analysis_worker_tmp_size,
            stream_cap_bytes=settings.analysis_worker_stream_cap_bytes,
        ),
        output_cap_bytes=settings.analysis_worker_report_max_bytes,
        output_watch_interval_seconds=settings.analysis_worker_output_watch_interval_seconds,
        strategy=settings.analysis_worker_strategy,
        max_iterations=settings.analysis_worker_max_iterations,
        exec_feedback_max_chars=settings.analysis_worker_exec_feedback_max_chars,
    )


def _exposes_coding_worker(agent: Agent) -> bool:
    return any(
        t.name == CodingWorkerConnector.name
        and CODING_WORKER_PR_WRITE in t.permissions
        for t in agent.spec.tools
    )


def _exposes_analysis_worker(agent: Agent) -> bool:
    return any(
        t.name == AnalysisWorkerConnector.name
        and ANALYSIS_REPORT_WRITE in t.permissions
        for t in agent.spec.tools
    )


def _build_worker_ledger(
    settings: Settings, agents: dict[str, Agent], usage: UsageStore
) -> WorkerSpendLedger | None:
    """The worker spend ledger (Phases 4+5), attributing per invoking agent.

    Each attempt's spend is recorded against the *invoking* agent's budget
    scope key — threaded through the approval args — so it shows up in
    ``/usage`` month-to-date next to that agent's chat spend, its
    ``per_task_usd`` is the fail-closed per-attempt cap, and its monthly
    budget gates the attempt before any work. The first agent exposing the
    tool stays the attribution fallback for attempts that carry no agent
    identity (pre-Phase 5 approvals/checkpoints).
    """
    owner = next(
        (a for a in agents.values() if _exposes_coding_worker(a)),
        next(iter(agents.values()), None),
    )
    if owner is None:
        return None
    return WorkerSpendLedger(
        usage=usage,
        model=settings.coding_worker_model,
        agents=agents,
        default_agent=owner.metadata.name,
        require_per_task_cap=settings.coding_worker_backend == "openhands",
    )


def _build_analysis_ledger(
    settings: Settings, agents: dict[str, Agent], usage: UsageStore
) -> WorkerSpendLedger | None:
    """A separate audit kind, with the same budget gates as other workers."""
    owner = next(
        (a for a in agents.values() if _exposes_analysis_worker(a)),
        next(iter(agents.values()), None),
    )
    if owner is None:
        return None
    return WorkerSpendLedger(
        usage=usage,
        model=settings.analysis_worker_model,
        agents=agents,
        default_agent=owner.metadata.name,
        task_kind="analysis_worker",
        # Opt-in hard posture (mirrors openhands when enabled): refuse any
        # attempt for a capless agent, keeping stale approved jobs fail-closed
        # even if config drifts between approval and recovery. Off by default:
        # iterative spend is structurally bounded (max_iterations completions,
        # capped feedback growth) and every run is human-approved; capped
        # agents keep the in-run abort and settle enforcement regardless.
        require_per_task_cap=settings.analysis_worker_require_per_task_cap,
    )


def _uncapped_worker_agents(
    agents: dict[str, Agent],
    ledger: WorkerSpendLedger,
    exposes: "Callable[[Agent], bool]" = _exposes_coding_worker,
) -> list[str]:
    """Agents whose worker attempts would run without a per-task cap.

    With per-invoker attribution the cap enforced is the invoking agent's, so
    the fail-closed gate for an agentic backend/strategy must hold for every
    agent that can start the worker — plus the ledger's fallback attribution
    target, which covers attempts with no agent identity.
    """
    exposed = {a.metadata.name: a for a in agents.values() if exposes(a)}
    exposed.setdefault(ledger.default_agent, agents[ledger.default_agent])
    return sorted(
        name for name, a in exposed.items()
        if a.spec.budget.per_task_usd is None
    )


def build_tool_gateway(
    settings: Settings,
    agents: dict[str, Agent],
    approvals: ApprovalStore,
    checkpoints: CheckpointStore,
    engine: WorkflowEngine,
    usage: UsageStore | None = None,
    analysis_inputs: InputStore | None = None,
    analysis_artifacts: ArtifactStore | None = None,
    analysis_attempts: AnalysisAttemptStore | None = None,
    analysis_uploads: UploadStore | None = None,
) -> ToolGateway:
    """Register native connectors plus an MCP connector per configured server.

    MCP connectors need an async setup() (tool discovery); the returned list is
    set up in the app lifespan.
    """
    gateway = ToolGateway(approvals=approvals, engine=engine)
    github_credentials = build_github_credentials(settings)
    github_client = (
        HttpGitHubClient(github_credentials)
        if github_credentials is not None
        else None
    )
    if github_credentials is not None:
        gateway.register(GitHubConnector(github_client))
        log.info("registered native tool: github")
        # The coding worker runs model-generated edits, so it stays off unless
        # explicitly enabled (it needs a contents:write credential + a sandbox).
        if settings.coding_worker_enabled:
            worker = build_coding_worker(settings)
            ledger = (
                _build_worker_ledger(settings, agents, usage)
                if usage is not None
                else None
            )
            if worker is None:
                # Fail-closed (Phase 3): a requested backend/isolation
                # boundary that can't start must NOT degrade to a different
                # worker or to running model-generated edits on the host.
                log.error(
                    "CODING WORKER DISABLED: CODING_WORKER_BACKEND=%s with "
                    "CODING_WORKER_SANDBOX=%s is not usable on this host and "
                    "the worker will not run with a weaker boundary. Fix the "
                    "configuration named in the error above.",
                    settings.coding_worker_backend,
                    settings.coding_worker_sandbox,
                )
            elif settings.coding_worker_backend == "openhands" and (
                ledger is None or _uncapped_worker_agents(agents, ledger)
            ):
                # Fail-closed (Phase 4 gate): an agentic worker with no
                # fail-closed spend cap must not run — on either durable
                # path, with or without a workflow engine. Phase 5's
                # per-invoker attribution means EVERY agent that can start
                # the worker needs its own cap.
                log.error(
                    "CODING WORKER DISABLED: CODING_WORKER_BACKEND=openhands "
                    "requires a fail-closed per-task spend cap. Set "
                    "spec.budget.per_task_usd on every agent exposing the "
                    "coding_worker tool%s.",
                    (
                        f" (missing on: "
                        f"{', '.join(_uncapped_worker_agents(agents, ledger))})"
                        if ledger is not None
                        else ""
                    ),
                )
            else:
                # Phase 2 split: the worker is credential-free (edits a
                # prepared workspace); the orchestrator alone holds the git
                # credential and is the ONE helper both durable paths run
                # attempts through — which is also what lets the Phase 4
                # ledger cover both paths at once.
                # Phase B: an optional warm-workspace pool lets the orchestrator
                # reuse a thread's checkout across turns. Attached to the gateway
                # so create_app can wire its durable context_ref sink + lifecycle.
                warm_pool = build_warm_workspace_pool(settings)
                gateway.warm_pool = warm_pool
                orchestrator = GitWorkspaceOrchestrator(
                    worker,
                    github_credentials,
                    workspace_root=_worker_workspace_root(settings),
                    ledger=ledger,
                    warm_pool=warm_pool,
                )
                gateway.register(
                    CodingWorkerConnector(
                        orchestrator, github_client, checkpoints=checkpoints
                    )
                )
                # Register the worker as a durable workflow (approval = wait
                # node).
                engine.register(
                    build_coding_worker_workflow(orchestrator, github_client)
                )
                log.info(
                    "registered native tool: coding_worker "
                    "(backend=%s, model=%s, sandbox=%s, default_per_task_cap=%s)",
                    settings.coding_worker_backend,
                    settings.coding_worker_model,
                    settings.coding_worker_sandbox,
                    ledger.per_task_usd_for(None) if ledger is not None else None,
                )
        else:
            log.info(
                "coding_worker tool not registered: set CODING_WORKER_ENABLED=1"
            )
    else:
        log.warning(
            "github tool not registered: set GITHUB_TOKEN or GITHUB_APP_*"
        )

    # Phase 1 sealed analysis is independent of GitHub credentials. It stays
    # disabled until the operator explicitly enables it and the digest-pinned
    # image + real sealed probe succeed; it never falls back to host execution.
    if settings.analysis_worker_enabled:
        worker = build_analysis_worker(settings)
        ledger = (
            _build_analysis_ledger(settings, agents, usage)
            if usage is not None
            else None
        )
        if worker is None or ledger is None:
            log.error(
                "ANALYSIS WORKER DISABLED: a usable sealed Docker backend and "
                "usage ledger are required; it will not run unsandboxed or "
                "without spend gates."
            )
        elif settings.analysis_worker_require_per_task_cap and (
            uncapped := _uncapped_worker_agents(
                agents, ledger, exposes=_exposes_analysis_worker
            )
        ):
            # Opt-in fail-closed gate (the openhands posture): the operator
            # demanded per-task caps, so the worker must not run for any agent
            # whose attempts would be uncapped.
            log.error(
                "ANALYSIS WORKER DISABLED: "
                "ANALYSIS_WORKER_REQUIRE_PER_TASK_CAP is set, so every agent "
                "exposing the analysis tool needs spec.budget.per_task_usd "
                "(missing on: %s).",
                ", ".join(uncapped),
            )
        else:
            inputs = analysis_inputs or build_analysis_input_store(settings)
            artifacts = analysis_artifacts or build_analysis_artifact_store(settings)
            attempts = analysis_attempts or build_analysis_attempt_store(settings)
            uploads = analysis_uploads or build_analysis_upload_store(settings)
            # Phase 4: one provisioner per available source. Sources whose
            # dependency isn't configured are simply not registered — the
            # connector's invoke-time resolution refuses them before a human
            # is ever asked to approve.
            provisioners = [StagedProvisioner(inputs)]
            available_sources = {"staged"}
            if settings.slack_bot_token:
                from openloop.surfaces.slack_files import SlackUploadFetcher

                provisioners.append(
                    UploadProvisioner(
                        uploads,
                        SlackUploadFetcher(settings.slack_bot_token),
                        max_bytes=settings.analysis_worker_upload_max_bytes,
                    )
                )
                available_sources.add("upload")
            if github_client is not None:
                provisioners.append(
                    GithubProvisioner(
                        github_client,
                        max_bytes=settings.analysis_worker_github_max_bytes,
                    )
                )
                available_sources.add("github")
            orchestrator = SealedAnalysisOrchestrator(
                worker,
                provisioners,
                artifacts,
                attempts=attempts,
                ledger=ledger,
                workspace_root=_analysis_workspace_root(settings),
                report_max_bytes=settings.analysis_worker_report_max_bytes,
                summary_lines=settings.analysis_worker_summary_lines,
                max_input_bytes=settings.analysis_worker_max_input_bytes,
            )
            gateway.analysis_input_store = inputs  # type: ignore[attr-defined]
            gateway.analysis_artifact_store = artifacts  # type: ignore[attr-defined]
            gateway.analysis_attempt_store = attempts  # type: ignore[attr-defined]
            gateway.analysis_upload_store = uploads  # type: ignore[attr-defined]
            gateway.analysis_sandbox_enabled = True  # type: ignore[attr-defined]
            gateway.register(
                AnalysisWorkerConnector(
                    orchestrator,
                    uploads=uploads,
                    available_sources=available_sources,
                )
            )
            # Register the sealed run as a durable workflow (approval = wait
            # node; store_result persists the {artifact_ref, prose_summary}
            # the session runner delivers from).
            engine.register(build_analysis_worker_workflow(orchestrator))
            log.info(
                "registered native tool: analysis "
                "(backend=builtin, strategy=%s, model=%s, sandbox=docker, "
                "sources=%s, default_per_task_cap=%s)",
                settings.analysis_worker_strategy,
                settings.analysis_worker_model,
                ",".join(sorted(available_sources)),
                ledger.per_task_usd_for(None),
            )
    else:
        log.info("analysis tool not registered: set ANALYSIS_WORKER_ENABLED=1")

    mcp_connectors: list[MCPConnector] = []
    seen: set[str] = set()
    for agent in agents.values():
        for tool in agent.spec.tools:
            if tool.type == "mcp" and tool.server and tool.name not in seen:
                credentials = None
                scope = None
                if tool.credentials == "github" and github_credentials is not None:
                    credentials = github_credentials
                    scope = CredentialScope(integration="github")
                elif tool.credentials:
                    log.warning(
                        "MCP tool %r wants %r credentials but none are "
                        "configured — registering unauthenticated",
                        tool.name,
                        tool.credentials,
                    )
                client = HttpMCPClient(
                    tool.server,
                    credentials=credentials,
                    scope=scope,
                    headers=tool.headers,
                )
                connector = MCPConnector(tool.name, client)
                gateway.register(connector)
                mcp_connectors.append(connector)
                seen.add(tool.name)
                log.info(
                    "registered MCP tool %r -> %s%s",
                    tool.name,
                    tool.server,
                    f" (credentials: {tool.credentials})" if credentials else "",
                )
    gateway.mcp_connectors = mcp_connectors  # type: ignore[attr-defined]
    return gateway


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
    }


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    agents = load_agents(settings.agents_dir)
    log.info("loaded %d agent(s): %s", len(agents), ", ".join(agents) or "none")

    embedder = build_embedder(settings)
    store = build_memory_store(settings)
    usage = build_usage_store(settings)
    approvals = build_approval_store(settings)
    checkpoints = build_checkpoint_store(settings)
    workflows = build_workflow_store(settings)
    engine = WorkflowEngine(workflows)
    sessions = build_surface_session_store(settings)
    analysis_inputs = build_analysis_input_store(settings)
    analysis_artifacts = build_analysis_artifact_store(settings)
    analysis_attempts = build_analysis_attempt_store(settings)
    analysis_uploads = build_analysis_upload_store(settings)
    # Phase A: the thread-scoped delivered-transcript store. Captured by the Slack
    # SessionRunner (like `sessions`); repointed in the lifespan on a PG fallback.
    threads = build_thread_record_store(settings)
    # Cross-process lock: lets one replica lead startup recovery. Rebound to an
    # in-process lock in the lifespan if a configured Redis can't be reached.
    coordinator = build_lock(settings)
    # Phase 5 throughput limits — one limiter shared by every runtime so
    # per-(tenant, agent) counters aggregate across surfaces in this process.
    limiter = InMemoryTaskLimiter()
    # The Slack SessionRunner captures the session store by reference; the lifespan
    # needs a handle to it to repoint after a Postgres fallback. Set in the Slack
    # block below (stays None when no Slack surface is bound).
    session_runner = None
    tools = build_tool_gateway(
        settings,
        agents,
        approvals,
        checkpoints,
        engine,
        usage=usage,
        analysis_inputs=analysis_inputs,
        analysis_artifacts=analysis_artifacts,
        analysis_attempts=analysis_attempts,
        analysis_uploads=analysis_uploads,
    )
    # The agent that tool/approval endpoints act on (first configured).
    primary_agent: Agent | None = next(iter(agents.values()), None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal coordinator, analysis_inputs, analysis_artifacts, analysis_attempts
        nonlocal analysis_uploads
        recovery_task: asyncio.Task | None = None
        warm_sweep_task: asyncio.Task | None = None
        if isinstance(store, PostgresMemoryStore):
            try:
                await store.setup()
                log.info("memory backend: postgres (pgvector)")
            except Exception:
                log.exception(
                    "postgres memory setup failed — falling back to in-memory"
                )
                app.state.memory = InMemoryStore()
                _rebind(app, "memory", app.state.memory)
        else:
            log.info("memory backend: in-memory (process-local)")

        if isinstance(usage, PostgresUsageStore):
            try:
                await usage.setup()
                log.info("usage backend: postgres")
            except Exception:
                log.exception(
                    "postgres usage setup failed — falling back to in-memory"
                )
                app.state.usage = InMemoryUsageStore()
                _rebind(app, "usage", app.state.usage)
                # The worker-spend ledger captured the Postgres store by
                # reference; repoint it so attempts don't hit an un-setup
                # pool. The per-task cap still enforces (it lives on the
                # ledger); only durable spend recording degrades.
                _repoint_worker_ledger(tools, app.state.usage)
        else:
            log.info("usage backend: in-memory (process-local)")

        if isinstance(analysis_inputs, PostgresInputStore):
            try:
                await analysis_inputs.setup()
                log.info("analysis input backend: postgres")
            except Exception:
                log.exception(
                    "postgres analysis input setup failed — falling back to in-memory"
                )
                analysis_inputs = InMemoryInputStore()
                app.state.analysis_inputs = analysis_inputs
                _repoint_analysis_stores(tools, inputs=analysis_inputs)
        else:
            log.info("analysis input backend: in-memory (process-local)")

        if isinstance(analysis_artifacts, PostgresArtifactStore):
            try:
                await analysis_artifacts.setup()
                log.info("analysis artifact backend: postgres")
            except Exception:
                log.exception(
                    "postgres analysis artifact setup failed — falling back to in-memory"
                )
                analysis_artifacts = InMemoryArtifactStore()
                app.state.analysis_artifacts = analysis_artifacts
                _repoint_analysis_stores(tools, artifacts=analysis_artifacts)
                # The session runner dereferences report refs at delivery time
                # from this same store — keep it pointed at the live instance.
                if session_runner is not None:
                    session_runner.artifacts = analysis_artifacts
        else:
            log.info("analysis artifact backend: in-memory (process-local)")

        if isinstance(analysis_attempts, PostgresAnalysisAttemptStore):
            try:
                await analysis_attempts.setup()
                log.info("analysis attempt backend: postgres")
            except Exception:
                log.exception(
                    "postgres analysis attempt setup failed — falling back to in-memory"
                )
                analysis_attempts = InMemoryAnalysisAttemptStore()
                app.state.analysis_attempts = analysis_attempts
                _repoint_analysis_stores(tools, attempts=analysis_attempts)
        else:
            log.info("analysis attempt backend: in-memory (process-local)")

        if isinstance(analysis_uploads, PostgresUploadStore):
            try:
                await analysis_uploads.setup()
                log.info("analysis upload backend: postgres")
            except Exception:
                log.exception(
                    "postgres analysis upload setup failed — falling back to in-memory"
                )
                analysis_uploads = InMemoryUploadStore()
                app.state.analysis_uploads = analysis_uploads
                _repoint_analysis_stores(tools, uploads=analysis_uploads)
                # The Slack surface records shares — and the runner reads the
                # inventory — through this same store; keep them live.
                if session_runner is not None:
                    session_runner.uploads = analysis_uploads
        else:
            log.info("analysis upload backend: in-memory (process-local)")

        if isinstance(approvals, PostgresApprovalStore):
            try:
                await approvals.setup()
                log.info("approval backend: postgres")
            except Exception:
                log.exception(
                    "postgres approval setup failed — falling back to in-memory"
                )
                tools.approvals = InMemoryApprovalStore()
        else:
            log.info("approval backend: in-memory (process-local)")

        if isinstance(checkpoints, PostgresCheckpointStore):
            try:
                await checkpoints.setup()
                log.info("checkpoint backend: postgres")
            except Exception:
                log.exception(
                    "postgres checkpoint setup failed — worker resume disabled"
                )
                _disable_checkpoints(tools)
        else:
            log.info("checkpoint backend: in-memory (process-local)")

        if isinstance(workflows, PostgresWorkflowStore):
            try:
                await workflows.setup()
                log.info("workflow backend: postgres")
            except Exception:
                log.exception(
                    "postgres workflow setup failed — falling back to in-memory"
                )
                # Swap the shared engine's store so BOTH the gateway and the
                # already-constructed runtime keep working (process-local, not
                # crash-durable) rather than hitting an un-setup pool.
                engine.store = InMemoryWorkflowStore()
        else:
            log.info("workflow backend: in-memory (process-local)")

        if isinstance(sessions, PostgresSurfaceSessionStore):
            try:
                await sessions.setup()
                log.info("surface-session backend: postgres")
            except Exception:
                log.exception(
                    "postgres surface-session setup failed — falling back "
                    "to in-memory"
                )
                # Repoint BOTH the app state and the already-built Slack runner
                # (which captured the Postgres store) at one shared in-memory
                # fallback, so mentions don't hit an un-setup pool in the
                # background — mirrors the workflow engine's store swap above.
                fallback_sessions = InMemorySurfaceSessionStore()
                app.state.sessions = fallback_sessions
                if session_runner is not None:
                    session_runner.sessions = fallback_sessions
        else:
            log.info("surface-session backend: in-memory (process-local)")

        if isinstance(threads, PostgresThreadRecordStore):
            try:
                await threads.setup()
                # A crashed drain leader leaves its active-turn claim set; clear
                # stale claims at startup so threads aren't wedged (single-replica).
                cleared = await threads.reset_active_claims()
                if cleared:
                    log.info("cleared %d stale thread claim(s) at startup", cleared)
                log.info("thread-record backend: postgres")
            except Exception:
                log.exception(
                    "postgres thread-record setup failed — falling back to in-memory"
                )
                fallback_threads = InMemoryThreadRecordStore()
                app.state.threads = fallback_threads
                if session_runner is not None:
                    session_runner.threads = fallback_threads
        else:
            log.info("thread-record backend: in-memory (process-local)")

        coordinator = await _setup_coordination(coordinator, settings)
        app.state.coordinator = coordinator  # the resolved lock (post-fallback)

        # Recover work left incomplete by a crash, as the recovery *leader* so that
        # when several replicas boot together only one runs the (idempotent but
        # wasteful-to-duplicate) sweeps. Run once now (before serving), then keep
        # re-running on an interval: a leader that crashes mid-sweep — whose Redis
        # lock then lingers for its whole TTL, even past its own restart — is
        # covered when a surviving replica's next pass acquires the lapsed lock.
        # Without the periodic retry a one-shot skip would strand that work until
        # an unrelated restart.
        await run_recovery_pass(coordinator, tools, session_runner)
        interval = settings.recovery_interval_seconds
        if interval > 0:
            recovery_task = asyncio.create_task(
                _recovery_loop(coordinator, tools, session_runner, interval=interval)
            )

        # Phase B: periodically evict idle warm checkouts so a thread that goes
        # quiet doesn't leak its directory (acquire also evicts opportunistically).
        warm_pool = getattr(tools, "warm_pool", None)
        if warm_pool is not None:
            warm_sweep_task = asyncio.create_task(
                _warm_sweep_loop(
                    warm_pool, interval=settings.coding_worker_warm_idle_seconds
                )
            )

        for connector in getattr(tools, "mcp_connectors", []):
            try:
                await connector.setup()
            except Exception:
                log.exception(
                    "MCP connector %r setup failed — its tools stay unavailable",
                    connector.name,
                )

        yield

        if recovery_task is not None:
            recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recovery_task
        if warm_sweep_task is not None:
            warm_sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warm_sweep_task
        warm_pool = getattr(tools, "warm_pool", None)
        if warm_pool is not None:
            await warm_pool.shutdown()
        if isinstance(store, PostgresMemoryStore):
            await store.close()
        if isinstance(usage, PostgresUsageStore):
            await usage.close()
        if isinstance(approvals, PostgresApprovalStore):
            await approvals.close()
        if isinstance(checkpoints, PostgresCheckpointStore):
            await checkpoints.close()
        if isinstance(workflows, PostgresWorkflowStore):
            await workflows.close()
        if isinstance(sessions, PostgresSurfaceSessionStore):
            await sessions.close()
        if isinstance(threads, PostgresThreadRecordStore):
            await threads.close()
        if isinstance(analysis_inputs, PostgresInputStore):
            await analysis_inputs.close()
        if isinstance(analysis_artifacts, PostgresArtifactStore):
            await analysis_artifacts.close()
        if isinstance(analysis_attempts, PostgresAnalysisAttemptStore):
            await analysis_attempts.close()
        if isinstance(analysis_uploads, PostgresUploadStore):
            await analysis_uploads.close()
        if hasattr(coordinator, "close"):  # RedisLock / PostgresLock (not in-process)
            await _safe_close(coordinator)

    app = FastAPI(title="OpenLoop", version="0.0.1", lifespan=lifespan)
    app.state.settings = settings
    app.state.agents = agents
    app.state.memory = store
    app.state.usage = usage
    app.state.tools = tools
    app.state.sessions = sessions
    app.state.threads = threads
    app.state.analysis_inputs = analysis_inputs
    app.state.analysis_artifacts = analysis_artifacts
    app.state.analysis_attempts = analysis_attempts
    app.state.analysis_uploads = analysis_uploads
    app.state.limiter = limiter

    # Phase B: bridge the warm-workspace pool's durable handle to the thread
    # record. Reads app.state.threads at call time so it survives a PG fallback
    # repoint; best-effort by design (the pool's liveness is authoritative).
    warm_pool = getattr(tools, "warm_pool", None)
    if warm_pool is not None:
        async def _persist_context_ref(warm_key: str, ref: str | None) -> None:
            await app.state.threads.set_context_ref(warm_key, ref)

        warm_pool.set_on_change(_persist_context_ref)
    # app.state.coordinator is set in the lifespan, after Redis connectivity is
    # checked, so it reflects the resolved lock (post any fallback).
    app.state.primary_agent = primary_agent

    # Bind the first Slack-enabled agent. The Bolt app is built when a bot
    # token exists (Socket Mode reuses it); the HTTP events route is added only
    # when a signing secret is present to verify requests.
    app.state.slack_app = None
    slack_agent = next(
        (a for a in agents.values() if a.has_slack_surface()), None
    )
    if slack_agent and settings.slack_bot_token:
        runtime = Runtime(
            slack_agent, memory=store, embedder=embedder, usage=usage,
            tools=tools, engine=engine, limiter=limiter,
        )
        app.state.runtime = runtime
        slack_app = build_slack_app(
            runtime,
            sessions,
            bot_token=settings.slack_bot_token,
            signing_secret=settings.slack_signing_secret or None,
            threads=threads,
            # Lets the runner dereference an analysis report ref into an
            # Artifact deliverable at delivery time (Phase 2).
            artifacts=analysis_artifacts,
            # Phase 4: the surface records thread-scoped file shares here and
            # the runner injects the shared-file inventory from it.
            uploads=analysis_uploads,
        )
        app.state.slack_app = slack_app
        # Captured so the session-store fallback can repoint the runner (above).
        session_runner = getattr(slack_app, "_session_runner", None)
        app.state.session_runner = session_runner

        if settings.slack_signing_secret:
            slack_handler = AsyncSlackRequestHandler(slack_app)

            @app.post("/slack/events")
            async def slack_events(req: Request):  # type: ignore[no-untyped-def]
                return await slack_handler.handle(req)

            log.info("Slack HTTP events bound to agent %r", slack_agent.metadata.name)
        else:
            log.info(
                "Slack app built for %r (Socket Mode); HTTP events disabled "
                "without SLACK_SIGNING_SECRET",
                slack_agent.metadata.name,
            )
    else:
        log.warning(
            "Slack surface not bound: need a Slack-enabled agent and "
            "SLACK_BOT_TOKEN"
        )

    @app.post("/tools/invoke")
    async def invoke_tool(body: InvokeBody, request: Request):  # type: ignore[no-untyped-def]
        """Run a tool action through the gateway (allowlist + approval gate)."""
        inv = await request.app.state.tools.invoke(
            _require_primary(request),
            body.action,
            body.args,
            requested_by=body.requested_by,
        )
        return _invocation_json(inv)

    @app.get("/approvals")
    async def list_approvals(request: Request):  # type: ignore[no-untyped-def]
        agent = _require_primary(request)
        pending = await request.app.state.tools.approvals.pending(
            agent=agent.metadata.name
        )
        return [
            {"id": r.id, "action": r.action, "summary": r.summary,
             "approvers": r.approvers, "requested_by": r.requested_by}
            for r in pending
        ]

    @app.post("/approvals/{request_id}/resolve")
    async def resolve_approval(request_id: str, body: ResolveBody, request: Request):  # type: ignore[no-untyped-def]
        inv = await request.app.state.tools.resolve(
            request_id, body.approver, approve=body.approve
        )
        if inv.status == "forbidden":
            raise HTTPException(403, inv.message)
        return _invocation_json(inv)

    @app.get("/usage")
    async def usage_summary(request: Request):  # type: ignore[no-untyped-def]
        """Month-to-date spend vs. budget for the primary agent."""
        agent = _require_primary(request)
        store: UsageStore = request.app.state.usage
        spent = await store.monthly_total(budget_scope_key(agent))
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
        store: UsageStore = request.app.state.usage
        records = await store.recent(limit=min(limit, 500))
        return [
            {
                "agent": r.agent,
                "channel": r.channel,
                "surface": r.surface,
                "user": r.user,
                "task_kind": r.task_kind,
                "model": r.model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "cost_usd": r.cost_usd,
                "outcome": r.outcome,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]

    @app.get("/healthz")
    async def healthz():  # type: ignore[no-untyped-def]
        actions = tools.available_actions(primary_agent) if primary_agent else []
        return {
            "status": "ok",
            "agents": list(agents),
            "providers": settings.configured_providers,
            "memory": type(app.state.memory).__name__,
            "usage": type(app.state.usage).__name__,
            "tools": actions,
        }

    return app


def _require_primary(request: Request) -> Agent:
    agent = getattr(request.app.state, "primary_agent", None)
    if agent is None:
        raise HTTPException(404, "no agents configured")
    return agent


def _rebind(app: FastAPI, attr: str, value) -> None:
    """Point the live runtime at a fallback store after a setup failure."""
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        setattr(runtime, attr, value)


def _repoint_worker_ledger(tools: ToolGateway, usage: UsageStore) -> None:
    """Point every registered worker-spend ledger at a fallback usage store."""
    for name in ("coding_worker", "analysis"):
        worker = tools._tools.get(name)
        orchestrator = getattr(worker, "orchestrator", None)
        ledger = getattr(orchestrator, "_ledger", None)
        if ledger is not None:
            ledger.usage = usage


def _repoint_analysis_stores(
    tools: ToolGateway,
    *,
    inputs: InputStore | None = None,
    artifacts: ArtifactStore | None = None,
    attempts: AnalysisAttemptStore | None = None,
    uploads: UploadStore | None = None,
) -> None:
    """Move a wired analysis orchestrator onto an in-memory store fallback."""
    connector = tools._tools.get("analysis")
    orchestrator = getattr(connector, "orchestrator", None)
    provisioners = getattr(orchestrator, "_provisioners", {})
    if inputs is not None and orchestrator is not None:
        staged = provisioners.get("staged")
        if staged is not None:
            staged.inputs = inputs
        tools.analysis_input_store = inputs  # type: ignore[attr-defined]
    if artifacts is not None and orchestrator is not None:
        orchestrator._artifacts = artifacts
        tools.analysis_artifact_store = artifacts  # type: ignore[attr-defined]
    if attempts is not None and orchestrator is not None:
        orchestrator._attempts = attempts
        tools.analysis_attempt_store = attempts  # type: ignore[attr-defined]
    if uploads is not None and connector is not None:
        # Both halves of the upload path: the connector's pre-approval
        # resolution and the provisioner's post-approval scope re-check + fetch.
        connector.uploads = uploads
        upload_provisioner = provisioners.get("upload")
        if upload_provisioner is not None:
            upload_provisioner.uploads = uploads
        tools.analysis_upload_store = uploads  # type: ignore[attr-defined]


def _disable_checkpoints(tools: ToolGateway) -> None:
    """Fall back to process-local checkpoints if the durable store can't start.

    The worker still runs and stays idempotent within a process, but jobs no
    longer resume across a restart.
    """
    worker = tools._tools.get("coding_worker")
    if worker is not None:
        worker.checkpoints = InMemoryCheckpointStore()


# Lease for the recovery lock. Short on purpose — a crashed leader's lock frees
# within this window — but renewed every _RECOVERY_LOCK_RENEW_SECONDS while a pass
# runs, so a live leader keeps the lock no matter how long the sweep takes (the
# worker resume can re-drive model generation + git push + PR open, well past the
# raw TTL). This decouples "self-heal latency" from "max sweep duration", so two
# replicas can't both acquire and re-drive the same jobs concurrently.
_RECOVERY_LOCK_TTL_SECONDS = 60.0
_RECOVERY_LOCK_RENEW_SECONDS = 20.0


async def run_recovery_pass(coordinator, tools: ToolGateway, session_runner) -> bool:
    """Guarded sweep of the crash-recovery reconcilers; returns whether we led it.

    Acquires the shared ``startup-recovery`` lock so that across replicas only one
    sweeps at a time; a contended replica returns ``False`` without doing the work
    (the leader's shared-store reconcile covers it). Idempotent and safe to repeat,
    which is what lets the periodic loop heal a leader that crashed mid-sweep once
    its lock TTL lapses. The lock is coordination, not correctness — a TTL expiry
    or in-process fallback at worst re-does idempotent work.
    """
    async with guard(
        coordinator,
        "startup-recovery",
        ttl_seconds=_RECOVERY_LOCK_TTL_SECONDS,
        renew_interval=_RECOVERY_LOCK_RENEW_SECONDS,
    ) as is_leader:
        if not is_leader:
            log.debug("another replica is leading recovery — skipping this pass")
            return False
        # The workflow engine re-drives any instance left "running"; the connector
        # reconciler covers the Phase B (no-engine) checkpoint path. resolve() won't
        # re-invoke an approved request, so these reconcilers trigger the resume.
        engine = getattr(tools, "engine", None)
        if engine is not None:
            try:
                resumed = await engine.resume_incomplete()
                if resumed:
                    log.info("resumed %d incomplete workflow(s)", len(resumed))
            except Exception:
                log.exception("workflow resume failed")
        await _resume_worker_jobs(tools)

        # Phase 0's layer-3 cleanup is relevant only once sealed analysis is
        # actually configured. It is safe under any replica count because the
        # helper reaps only OpenLoop analysis containers past their own stamped
        # deadline; Docker failures are contained inside the best-effort helper.
        if getattr(tools, "analysis_sandbox_enabled", False):
            reaped = await sweep_expired_sandboxes()
            if reaped:
                log.info("reaped %d expired sealed analysis sandbox(es)", len(reaped))

        # Repair surface-session delivery left mid-flight by a crash — after the
        # engine resume above so each session's workflow is already terminal.
        if session_runner is not None:
            try:
                repaired = await session_runner.reconcile()
                if repaired:
                    log.info("reconciled %d surface session(s)", len(repaired))
            except Exception:
                log.exception("surface-session reconcile failed")
        return True


async def _recovery_loop(
    coordinator, tools: ToolGateway, session_runner, *, interval: float
) -> None:
    """Re-run :func:`run_recovery_pass` every ``interval`` seconds until cancelled.

    The self-healing backstop for a leader that died mid-sweep: a surviving
    replica's next pass acquires the lapsed lock and finishes the recovery.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await run_recovery_pass(coordinator, tools, session_runner)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("periodic recovery pass failed")


async def _warm_sweep_loop(pool: WarmWorkspacePool, *, interval: float) -> None:
    """Evict idle warm checkouts every ``interval`` seconds until cancelled."""
    while True:
        await asyncio.sleep(interval)
        try:
            await pool.sweep()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("warm-workspace sweep failed", exc_info=True)


async def _safe_close(closeable) -> None:
    """Best-effort close of a coordination client; never raise from teardown."""
    try:
        await closeable.close()
    except Exception:
        log.warning("failed to close coordination client", exc_info=True)


async def _resume_worker_jobs(tools: ToolGateway) -> None:
    """Drive the coding worker's startup reconciler, if it is registered."""
    worker = tools._tools.get("coding_worker")
    if worker is None or not hasattr(worker, "resume_incomplete"):
        return
    try:
        resumed = await worker.resume_incomplete()
        if resumed:
            log.info("resumed %d incomplete coding-worker job(s)", len(resumed))
    except Exception:
        log.exception("coding-worker job resume failed")


app = create_app()
