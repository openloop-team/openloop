"""Runtime configuration loaded from environment / `.env`.

Mirrors the keys documented in `.env.example`. Only what the first vertical
slice needs is wired up here; more lands as the runtime grows.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from openloop.openhands.runtime_profile import DEFAULT_OPENHANDS_SERVER_IMAGE


# Temporary smoke-test default: the official multi-platform Python 3.12 slim
# image pinned by registry digest. It satisfies the sealed runtime contract
# (python + GNU timeout), but not the richer pandas/numpy/matplotlib contract of
# docker/analysis.Dockerfile; production deployments should override it with
# the purpose-built image's immutable digest.
DEFAULT_ANALYSIS_SANDBOX_IMAGE = (
    "python@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf"
)


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
    #   "claude"    — EXPERIMENTAL / personal use: drive the `claude` CLI in
    #               headless mode (`claude -p`), authenticating with whatever
    #               `claude` is logged into, INCLUDING a Pro/Max subscription.
    #               Host sandbox only; bounded by --max-turns + the deadline
    #               (its load-bearing fail-closed cap, since the subscription
    #               dollar signal is unreliable). See claude_worker.py for the
    #               ToS/reversibility caveats before enabling on a team surface.
    # FAIL-CLOSED: an unknown value disables the coding worker loudly; a typo
    # in a spend/safety setting must not select a different worker.
    coding_worker_backend: str = "builtin"
    # In-run iteration cap handed to the OpenHands conversation / claude
    # `--max-turns`. For openhands the budget cap is enforced by the worker-spend
    # ledger (per_task_usd), not in-run; for claude this is half the fail-closed
    # bound (the deadline is the other half).
    coding_worker_max_iterations: int = 100
    # Wall-clock ceiling for a single attempt. For OpenHands it is a soft deadline
    # checked between agent events (it cannot interrupt a truly-frozen single call
    # — that needs the docker sandbox to hard-kill the container); 0 disables it.
    # For the claude backend it is a HARD kill of the subprocess and its
    # load-bearing fail-closed bound, so a value > 0 is required there.
    coding_worker_deadline_seconds: float = 600.0
    # Path to the `claude` CLI for CODING_WORKER_BACKEND=claude.
    coding_worker_claude_bin: str = "claude"
    # Headless permission handling for the claude backend. "acceptEdits" (default)
    # auto-accepts file edits; "bypassPermissions" grants full autonomy (shell,
    # tests) at higher risk — recommended only inside a sandbox.
    coding_worker_claude_permission_mode: str = "acceptEdits"
    # Agent-server image for the OpenHands docker runtime
    # (CODING_WORKER_SANDBOX=docker + backend=openhands).
    coding_worker_openhands_image: str = DEFAULT_OPENHANDS_SERVER_IMAGE
    # Docker network for the OpenHands agent-server container. Unset = the
    # default bridge (the agent loop runs in-container and needs egress to
    # the model provider — "none" would break it). Point at an egress-proxy
    # network to move to an allowlist model.
    coding_worker_openhands_network: str | None = None
    # How the runtime reaches the agent-server:
    #   "loopback" (default) — publish 127.0.0.1:<port> on the Docker daemon
    #       host; correct when the runtime runs on that host.
    #   "network" — publish nothing; dial the container by name over
    #       CODING_WORKER_OPENHANDS_NETWORK (required — a user-defined network
    #       shared with the runtime container). Use for sibling-container
    #       Compose deployments, where the daemon host's loopback is
    #       unreachable from the runtime's network namespace.
    coding_worker_openhands_connect: str = "loopback"
    # Phase 0 cold-resume foundation. The root defaults beneath the system temp
    # directory; production may point it at storage shared by resume-capable
    # replicas. It always stays outside Git checkouts.
    coding_worker_openhands_state_dir: str | None = None
    # Dedicated base64-encoded 32-byte master key. Required for Docker OpenHands;
    # SecretStr keeps it out of Settings repr/logging. Never reuse another app,
    # Slack, GitHub, or provider secret here.
    coding_worker_openhands_state_master_key: SecretStr | None = None
    coding_worker_openhands_master_key_id: str = "key-v1"
    # OpenHands Docker runs park at confirmation boundaries and resume in a
    # fresh container by default. Set false only as an operational rollback;
    # the authenticated runtime and encrypted state foundation remain active.
    coding_worker_openhands_cold_resume_enabled: bool = True
    # --- Container broker (architecture step 4, first wiring slice) ----------
    # Master flag. When true, docker-mode OpenHands routes container lifecycle
    # through the co-process broker over its UDS RPC boundary instead of the
    # in-process HardenedDockerWorkspace. FAIL-CLOSED: when true but the broker
    # cannot be built (missing/invalid setting below), the coding worker is
    # DISABLED loudly — it never falls back to the direct launch path.
    coding_worker_openhands_broker_enabled: bool = False
    # Parent directory for the broker control UDS (<dir>/control.sock). Required
    # when the broker flag is on; the app must own and be able to bind it.
    broker_control_socket_dir: str | None = None
    # Durable-state root the broker's runtime driver owns (per-job isolation).
    # Required when the broker flag is on; stays outside Git checkouts.
    broker_state_root: str | None = None
    # Runtime scratch root for the Docker runtime driver. DockerRuntimeConfig
    # requires runtime_root and state_root to be DISJOINT (it validates this),
    # so this is a separate path from broker_state_root. Required when on.
    broker_runtime_root: str | None = None
    # Separated key topology — ONE root ring per trust domain (never a shared
    # master root; production splits client/broker/checkpoint-store into distinct
    # trust domains). Each domain is a VERSION -> base64-32-byte-secret map plus a
    # current_version, so keys rotate while old capabilities/durable digests/
    # receipts still verify (overlapping verification) across a restart — the
    # single-key form could not represent rotation. SecretStr keeps the values out
    # of repr/logs. build_broker fail-closed-validates when the flag is on:
    # non-empty map, current_version present, base64/32-byte, and NO reused root
    # bytes within or across domains (a reused root is fake rotation / a shared
    # trust line). Never reuse another app/Slack/GitHub/provider secret here.
    #
    # Identity (client-domain, workload-identity tokens): the Ed25519 keypair is
    # generated EPHEMERALLY per-process this slice (tokens live <=300s and never
    # outlive the process), so only the issuer/audience identifiers are config.
    broker_identity_issuer: str = "openloop-app"
    broker_identity_audience: str = "openloop-broker"
    # Capability (broker-domain, CapabilityRootRing). Broker-only; the client
    # receives per-job capabilities over RPC.
    broker_capability_roots: dict[str, SecretStr] = Field(default_factory=dict)
    broker_capability_current_version: str = "cap-key-v1"
    # Runtime/durable (broker-domain, RuntimeSecretRootRing -> the coordinator's
    # RuntimeSecretAuthority for session/relay/durable digests).
    broker_runtime_roots: dict[str, SecretStr] = Field(default_factory=dict)
    broker_runtime_current_version: str = "runtime-key-v1"
    # Receipt (checkpoint-store-domain): a SEPARATE Ed25519 signing key is derived
    # per version from these roots via RuntimeSecretRootRing under
    # broker_receipt_domain — the decision-2 split. current_version's private half
    # -> checkpoint store (signer); ALL versions' public halves -> broker verifier
    # (overlapping verification keys). Reuses the rotation plumbing, not key bytes.
    broker_receipt_roots: dict[str, SecretStr] = Field(default_factory=dict)
    broker_receipt_current_version: str = "receipt-key-v1"
    broker_receipt_domain: str = "broker-receipt"
    # Segment execution lease and absolute in-container generation deadline.
    # Both are bounded by the runtime driver's maximum lifetime (the coordinator
    # rejects a lease that exceeds it at construction).
    broker_execution_lease_seconds: int = 900
    broker_generation_deadline_seconds: int = 1800
    # --- Broker process split (phase 1): mode + trust-topology surfaces -----
    # Master switch. "coprocess" (default, unchanged today) runs the broker
    # graph in-process; "external" talks to a separate `openloop-broker`
    # process over the same UDS RPC boundary. A typo here must not silently
    # boot as coprocess, so this is validator-enforced below.
    broker_mode: str = "coprocess"  # "coprocess" | "external"
    # --- external mode, app side ---
    # base64 32-byte Ed25519 seed; the app signs workload-identity tokens with
    # it in external mode. SecretStr keeps it out of Settings repr/logging.
    broker_identity_private_key: SecretStr | None = None
    broker_identity_key_id: str = "identity-v1"
    # --- external mode, broker side ---
    # key_id -> base64 Ed25519 public. The broker verifies app-issued identity
    # tokens against these; it never holds the private half.
    broker_identity_public_keys: dict[str, str] = Field(default_factory=dict)
    # version -> base64 Ed25519 public. The broker verifies checkpoint-store
    # receipts against these; it never holds a receipt private half either.
    broker_receipt_public_keys: dict[str, str] = Field(default_factory=dict)
    # --- shared surfaces ---
    # Dedicated receipts subtree, BOTH sides (identical absolute path; RO
    # mount broker-side). REQUIRED app-side in external mode too: the client
    # dual-writes the dedicated sidecar here — without it only the legacy
    # in-artifact sidecar exists and broker recovery cannot locate receipts.
    # (Refines the spec's `broker_checkpoint_root`: only receipts/ crosses
    # the boundary, so the setting names exactly what gets mounted.)
    broker_checkpoint_receipt_root: str | None = None
    # App-owned sibling root; external-required, coprocess derives
    # runtime_root/.workspace-ingress when unset.
    broker_ingress_root: str | None = None
    # Numeric identities only — group/user *names* never cross the container
    # boundary.
    broker_shared_data_gid: int | None = None
    broker_expected_app_uid: int | None = None
    # 0/negative would busy-loop the periodic reconcile pass.
    broker_reconcile_interval_seconds: int = Field(default=300, gt=0)
    # Entrypoint only: in-memory repo/audit (tests/dev).
    broker_dev_in_memory: bool = False
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
    # host execution. The worker remains off by default; when enabled, it uses
    # a digest-pinned Python smoke image unless the operator supplies the richer
    # purpose-built image from docker/analysis.Dockerfile.
    analysis_worker_enabled: bool = False
    analysis_worker_backend: str = "builtin"
    analysis_worker_model: str = "anthropic/claude-sonnet-4-6"
    # Only ``docker`` is accepted. ``host`` is an explicit unsafe value and
    # disables the worker rather than weakening the execution boundary.
    analysis_worker_sandbox: str = "docker"
    # Must be a digest reference (contains ``@sha256:``); no mutable image tag
    # is allowed for arbitrary model-authored execution.
    analysis_worker_sandbox_image: str = DEFAULT_ANALYSIS_SANDBOX_IMAGE
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
    # Phase 4 provisioning caps. The merged cap is a decrementing budget the
    # orchestrator checks before every per-source fetch (once spent, no further
    # fetch starts); the per-source caps additionally bound each download in
    # flight. Repo-shaped inputs extract into tmpfs inside the sandbox, which
    # counts against container memory — raise ANALYSIS_WORKER_TMP_SIZE (and
    # memory) alongside these when provisioning repos.
    analysis_worker_max_input_bytes: int = 33_554_432  # 32 MiB merged manifest
    analysis_worker_github_max_bytes: int = 33_554_432  # 32 MiB per tarball
    analysis_worker_upload_max_bytes: int = 16_777_216  # 16 MiB per upload

    # Storage / queue
    database_url: str = (
        "postgresql://openloop:change-me@localhost:5432/openloop_agents"
    )
    # One ordinary-query pool per runtime process. The Postgres coordination
    # backend intentionally owns a separate small pool because advisory locks
    # hold connections for the lifetime of a lease.
    postgres_pool_min_size: int = Field(default=1, ge=0)
    postgres_pool_max_size: int = Field(default=10, ge=1)
    redis_url: str = "redis://localhost:6379/0"

    # Cross-process coordination for multi-replica deploys — which lock backend
    # leads startup recovery:
    #   "auto"     (default) — follow effective_storage_mode: Postgres advisory
    #              lock for auto/postgres storage, else in-process.
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
    # Canonical storage policy. ``None`` preserves the legacy MEMORY_BACKEND
    # input during migration; all runtime consumers use effective_storage_mode.
    storage_mode: Literal["auto", "postgres", "memory"] | None = None
    # Backend: "memory" (process-local, default — runs without a DB) or
    # "postgres" (pgvector-backed, persistent). Deprecated: prefer
    # STORAGE_MODE; retained as a derived-only compatibility input.
    memory_backend: str = "memory"
    # Set to false to disable semantic recall (recency-only memory).
    embeddings_enabled: bool = True
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    @field_validator("broker_mode")
    @classmethod
    def _validate_broker_mode(cls, value: str) -> str:
        allowed = {"coprocess", "external"}
        if value not in allowed:
            raise ValueError(
                f"broker_mode must be one of {sorted(allowed)}, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _validate_postgres_pool_size(self) -> Self:
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError(
                "postgres_pool_min_size must be less than or equal to "
                "postgres_pool_max_size"
            )
        return self

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

    @property
    def effective_storage_mode(self) -> Literal["auto", "postgres", "memory"]:
        """Resolve the canonical storage policy, including the legacy input."""
        if self.storage_mode is not None:
            return self.storage_mode
        return "auto" if self.memory_backend == "postgres" else "memory"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
