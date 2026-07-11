"""Runtime configuration loaded from environment / `.env`.

Mirrors the keys documented in `.env.example`. Only what the first vertical
slice needs is wired up here; more lands as the runtime grows.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model providers — LiteLLM reads these from the environment directly, but we
    # surface them here so the runtime can report which providers are configured.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Slack surface
    slack_bot_token: str | None = None
    slack_signing_secret: str | None = None
    slack_app_token: str | None = None

    # GitHub connector — either a static token (GITHUB_TOKEN) or, preferred, a
    # GitHub App whose short-lived installation tokens are minted on demand
    # (all three GITHUB_APP_* values required; needs the `githubapp` extra).
    # When both are set the App wins; the token remains a fallback.
    github_token: str | None = None
    github_app_id: str | None = None
    github_app_private_key_path: str | None = None
    github_app_installation_id: str | None = None
    # Optional least-privilege restriction: comma-separated bare repo names
    # (no owner). Unset = the minted token spans every repo the installation
    # can access.
    github_app_repositories: str | None = None

    # Coding worker — model the worker uses to generate edits. Matches the
    # `task: code` route in the example agent. Codegen is multi-step and
    # token-heavy; revisit `per_task_usd` for `task: code` accordingly.
    coding_worker_model: str = "anthropic/claude-sonnet-4-6"
    # Enable the real git-backed worker (needs a contents:write token + a
    # sandboxed checkout). Off by default — the connector stays unregistered.
    coding_worker_enabled: bool = False
    # Which worker engine edits the prepared workspace:
    #   "builtin"   (default) — OpenLoop's own light worker (BuiltinCodingWorker):
    #               one model call for a unified diff, applied through the
    #               sandbox.
    #   "openhands" — the heavy agentic worker (needs the `openhands` extra
    #               AND a per-task budget on the owning agent — the run is
    #               refused without a fail-closed spend cap).
    # FAIL-CLOSED: an unknown value disables the coding worker loudly; a typo
    # in a spend/safety setting must not select a different worker.
    coding_worker_backend: str = "builtin"
    # In-run iteration cap handed to the OpenHands conversation. The budget
    # cap is enforced by the worker-spend ledger (per_task_usd), not in-run.
    coding_worker_max_iterations: int = 100
    # Wall-clock ceiling for a single OpenHands attempt, checked between agent
    # events (a soft deadline: it cannot interrupt a truly-frozen single call —
    # that needs the docker sandbox to hard-kill the container). 0 disables it.
    coding_worker_deadline_seconds: float = 600.0
    # Agent-server image for the OpenHands docker runtime
    # (CODING_WORKER_SANDBOX=docker + backend=openhands).
    coding_worker_openhands_image: str = (
        "ghcr.io/openhands/agent-server:latest-python"
    )
    # Docker network for the OpenHands agent-server container. Unset = the
    # default bridge (the agent loop runs in-container and needs egress to
    # the model provider — "none" would break it). Point at an egress-proxy
    # network to move to an allowlist model.
    coding_worker_openhands_network: str | None = None
    # Where the worker's model-influenced execution (applying generated edits)
    # runs:
    #   "host"   (default) — a plain subprocess in this process's environment.
    #   "docker" — a throwaway container per command: default-deny egress
    #              (network none), no env forwarded, capabilities dropped.
    # FAIL-CLOSED: if "docker" is requested but docker can't run, the coding
    # worker is DISABLED (loudly) — it never silently falls back to the host.
    coding_worker_sandbox: str = "host"
    # Image for the docker sandbox; needs a `git` binary on PATH.
    coding_worker_sandbox_image: str = "alpine/git"
    # Container network. "none" = default-deny (the worker needs no egress —
    # the model call happens in the controller). Point at a user-defined
    # docker network fronted by an egress proxy for an allowlist model.
    coding_worker_sandbox_network: str = "none"
    # Where attempt workspaces are created (default: system tempdir). Required
    # when the runtime itself runs in a container with the docker sandbox:
    # sibling sandbox containers resolve bind-mount paths on the HOST, so this
    # must be a host path mounted into the runtime at the same location.
    coding_worker_workspace_dir: str | None = None
    # Phase B — warm execution context. When on, a coding worker keeps its git
    # checkout alive between turns in the same thread so a follow-up reuses it
    # (fetch + reset) instead of cloning cold. Process-local and single-replica-
    # correct: warm is only a cache, a cold clone is always the fallback, so this
    # can default on — a warm miss (restart, eviction, busy, or a discarded dirty
    # tree) degrades to the unchanged ephemeral clone-and-discard path. Set to
    # false to force that path everywhere.
    coding_worker_warm_context: bool = True
    # Evict a thread's warm checkout after this many idle seconds (leak guard).
    coding_worker_warm_idle_seconds: float = 900.0
    # Cap on concurrently-kept warm checkouts (LRU-evicted past it).
    coding_worker_warm_capacity: int = 8

    # Sealed analysis worker (Phase 1). It executes model-authored Python over
    # controller-provisioned data, so unlike the coding worker it NEVER permits
    # host execution. The worker remains off until an operator supplies the
    # digest-pinned image produced from docker/analysis.Dockerfile.
    analysis_worker_enabled: bool = False
    analysis_worker_backend: str = "builtin"
    analysis_worker_model: str = "anthropic/claude-sonnet-4-6"
    # Only ``docker`` is accepted. ``host`` is an explicit unsafe value and
    # disables the worker rather than weakening the execution boundary.
    analysis_worker_sandbox: str = "docker"
    # Must be a digest reference (contains ``@sha256:``); no mutable image tag
    # is allowed for arbitrary model-authored execution.
    analysis_worker_sandbox_image: str | None = None
    # This worker has no adaptive access: its sandbox always stays fully sealed.
    analysis_worker_sandbox_network: str = "none"
    # Host path visible to the Docker daemon; required in a containerized deploy
    # for the same sibling-container bind-mount reason as coding workspaces.
    analysis_worker_workspace_dir: str | None = None
    analysis_worker_timeout_seconds: float = 120.0
    analysis_worker_kill_after_seconds: float = 10.0
    analysis_worker_memory: str = "512m"
    analysis_worker_memory_swap: str | None = None
    analysis_worker_cpus: float = 1.0
    analysis_worker_pids_limit: int = 128
    analysis_worker_tmp_size: str = "64m"
    # Each stdout/stderr stream is retained to this cap (but drained to EOF).
    analysis_worker_stream_cap_bytes: int = 262_144
    # Both the best-effort outputs-dir watchdog and the hard report read-out
    # boundary use this cap. The watchdog only limits overshoot; read-out is the
    # exfiltration guarantee.
    analysis_worker_report_max_bytes: int = 1_000_000
    analysis_worker_output_watch_interval_seconds: float = 2.0
    analysis_worker_summary_lines: int = 12
    # Strategy inside the one builtin backend (strategies never become sibling
    # backends): "iterative" (default) = generate → run → feed capped
    # stdout/stderr back to the model → refine (Phase 3); "single" = one
    # completion + one sealed run (Phase 1). Iterative spend is structurally
    # bounded even without a dollar cap: at most max_iterations completions
    # per attempt, prompt growth hard-capped by the exec-feedback limit, and
    # every run is human-approved.
    analysis_worker_strategy: str = "iterative"
    # Optional hard boot gate (the openhands-style posture): when set, every
    # agent exposing the analysis tool must carry spec.budget.per_task_usd or
    # the worker is disabled, and stale approved jobs are refused if caps
    # drift after approval. Off by default — agents that do carry a cap still
    # get the in-run spend abort and the fail-closed settle either way.
    analysis_worker_require_per_task_cap: bool = False
    # Iterative only: model completions (each followed by one sealed run)
    # allowed per attempt.
    analysis_worker_max_iterations: int = 4
    # Iterative only: per-stream cap on the exec_feedback (stdout/stderr) the
    # in-controller model sees each round. Feedback never posts to a surface.
    analysis_worker_exec_feedback_max_chars: int = 16_384

    # Storage / queue
    database_url: str = (
        "postgresql://openloop:change-me@localhost:5432/openloop_agents"
    )
    redis_url: str = "redis://localhost:6379/0"

    # Cross-process coordination for multi-replica deploys — which lock backend
    # leads startup recovery:
    #   "auto"     (default) — follow memory_backend: Postgres advisory lock when
    #              memory_backend=postgres (no extra service), else in-process.
    #   "memory"   — force a process-local lock (single replica).
    #   "postgres" — force Postgres advisory locks (reuses database_url).
    #   "redis"    — force a Redis lock (needs redis_url + the `redis` extra).
    # An *explicit* postgres/redis that can't start logs loudly then degrades to
    # in-process; "auto" degrades quietly.
    lock_backend: str = "auto"
    # How often (seconds) to re-run the crash-recovery sweep under the lock, the
    # backstop that heals a recovery leader that died mid-sweep. 0 disables the
    # periodic retry (startup-only). Runs once at startup regardless.
    recovery_interval_seconds: int = 300

    # Runtime
    log_level: str = "info"

    # Where agent config-as-code lives
    agents_dir: str = "agents"

    # Memory
    # Backend: "memory" (process-local, default — runs without a DB) or
    # "postgres" (pgvector-backed, persistent).
    memory_backend: str = "memory"
    # Set to false to disable semantic recall (recency-only memory).
    embeddings_enabled: bool = True
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    @property
    def github_app_configured(self) -> bool:
        """True when all three GitHub App values are set."""
        return bool(
            self.github_app_id
            and self.github_app_private_key_path
            and self.github_app_installation_id
        )

    @property
    def github_app_repository_list(self) -> list[str]:
        """``github_app_repositories`` parsed into a list (empty when unset)."""
        if not self.github_app_repositories:
            return []
        return [
            repo.strip()
            for repo in self.github_app_repositories.split(",")
            if repo.strip()
        ]

    @property
    def embedding_provider(self) -> str:
        """LiteLLM-style provider prefix of the embedding model."""
        return self.embedding_model.split("/", 1)[0]

    @property
    def configured_providers(self) -> list[str]:
        """Provider prefixes (LiteLLM-style) that have a key set."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.gemini_api_key:
            providers.append("gemini")
        if self.openrouter_api_key:
            providers.append("openrouter")
        return providers


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
