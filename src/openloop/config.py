"""Process-owned settings loaded from secrets, environment, or dotenv."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from openloop.openhands.runtime_profile import DEFAULT_OPENHANDS_SERVER_IMAGE

DEFAULT_EXTERNAL_BROKER_CONTAINER_ROOT = Path("/var/lib/openloop/broker")
DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR = (
    DEFAULT_EXTERNAL_BROKER_CONTAINER_ROOT / "control"
)
DEFAULT_EXTERNAL_BROKER_STATE_ROOT = (
    DEFAULT_EXTERNAL_BROKER_CONTAINER_ROOT / "state"
)
DEFAULT_EXTERNAL_BROKER_RUNTIME_ROOT = (
    DEFAULT_EXTERNAL_BROKER_CONTAINER_ROOT / "runtime"
)
DEFAULT_EXTERNAL_BROKER_INGRESS_ROOT = (
    DEFAULT_EXTERNAL_BROKER_CONTAINER_ROOT / "ingress"
)
DEFAULT_EXTERNAL_BROKER_CHECKPOINT_RECEIPT_ROOT = (
    DEFAULT_EXTERNAL_BROKER_CONTAINER_ROOT / "receipts"
)


class _OpenLoopSettings(BaseSettings):
    """Common source policy for each independently loaded process schema."""

    model_config = SettingsConfigDict(
        env_file=".runtime.env",
        env_file_encoding="utf-8",
        secrets_dir="/run/secrets",
        extra="ignore",
        hide_input_in_errors=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keep test overrides first, then prefer mounted production secrets.

        Pydantic's default order puts environment and dotenv values ahead of
        its Docker-secrets source. OpenLoop reverses that part deliberately:
        once Compose grants a secret file to the service, an old environment
        value must not silently replace it. The standard mount is optional so
        ordinary local runs do not warn when ``/run/secrets`` is absent.
        """
        del settings_cls
        configured = getattr(file_secret_settings, "secrets_dir", None)
        secret_dirs = (
            (configured,)
            if isinstance(configured, (str, Path))
            else tuple(configured or ())
        )
        external_sources: tuple[PydanticBaseSettingsSource, ...]
        if any(Path(path).expanduser().exists() for path in secret_dirs):
            external_sources = (
                file_secret_settings,
                env_settings,
                dotenv_settings,
            )
        else:
            external_sources = (env_settings, dotenv_settings)
        return (init_settings, *external_sources)


class RuntimeSettings(_OpenLoopSettings):
    """Unprivileged application-runtime settings.

    This schema owns provider, surface, storage, worker, and external
    broker-client inputs. Broker service authority is intentionally absent.
    """

    # Model providers. Wiring passes these explicitly to LiteLLM so mounted
    # secret files work without copying credentials into process environment.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
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
    # Optional long-lived Claude Code subscription token. Kept out of the
    # process environment when loaded from /run/secrets and exposed only to the
    # child `claude` process.
    claude_code_oauth_token: SecretStr | None = None
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
    # SecretStr keeps it out of RuntimeSettings repr/logging. Never reuse
    # another app, Slack, GitHub, or provider secret here.
    coding_worker_openhands_state_master_key: SecretStr | None = None
    coding_worker_openhands_master_key_id: str = "key-v1"
    # OpenHands Docker runs park at confirmation boundaries and resume in a
    # fresh container by default. Set false only as an operational rollback;
    # the authenticated runtime and encrypted state foundation remain active.
    coding_worker_openhands_cold_resume_enabled: bool = True
    # --- Container broker client ---------------------------------------------
    # Master flag. When true, docker-mode OpenHands routes container lifecycle
    # through the broker over its UDS RPC boundary. FAIL-CLOSED: when true but
    # the client cannot be built, the coding worker is disabled loudly.
    coding_worker_openhands_broker_enabled: bool = False
    # Runtime-side mount targets. Compose may bind any host source to these
    # stable paths; source and target paths are independent.
    broker_control_socket_dir: str = str(
        DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR
    )
    broker_ingress_root: str = str(DEFAULT_EXTERNAL_BROKER_INGRESS_ROOT)
    broker_checkpoint_receipt_root: str = str(
        DEFAULT_EXTERNAL_BROKER_CHECKPOINT_RECEIPT_ROOT
    )
    # Master switch for the client topology. A typo must fail validation.
    broker_mode: str = "coprocess"  # "coprocess" | "external"
    # External mode: the runtime signs short-lived workload identity tokens.
    broker_identity_issuer: str = "openloop-app"
    broker_identity_audience: str = "openloop-broker"
    broker_identity_private_key: SecretStr | None = None
    broker_identity_key_id: str = "identity-v1"
    # Receipt roots belong to the checkpoint-store/client trust domain. The
    # broker process receives public verification keys, never these roots.
    broker_receipt_roots: dict[str, SecretStr] = Field(default_factory=dict)
    broker_receipt_current_version: str = "receipt-key-v1"
    broker_receipt_domain: str = "broker-receipt"
    broker_shared_data_gid: int | None = None
    # Execution authority marker. ``host`` uses the app's plain subprocess
    # executor. ``docker`` is accepted only by containerized OpenHands and means
    # the external broker owns execution; the app never launches containers.
    coding_worker_sandbox: str = "host"
    # Where attempt workspaces are created (default: system tempdir). External
    # broker deployments set this to storage visible to the ingress handoff.
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

    # Storage / queue
    # Names the same database Compose provisions from POSTGRES_DB, so the bare
    # default and a Compose-provisioned Postgres agree instead of the app
    # quietly opening a second database.
    database_url: str = "postgresql://openloop:change-me@localhost:5432/openloop"
    # Allows the DSN to remain non-secret. Compose can grant the same mounted
    # password file to Postgres, runtime, and broker without exporting it.
    postgres_password: SecretStr | None = None
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
        if self.groq_api_key:
            providers.append("groq")
        if self.openrouter_api_key:
            providers.append("openrouter")
        return providers

    @property
    def effective_storage_mode(self) -> Literal["auto", "postgres", "memory"]:
        """Resolve the canonical storage policy, including the legacy input."""
        if self.storage_mode is not None:
            return self.storage_mode
        return "auto" if self.memory_backend == "postgres" else "memory"


class BrokerSettings(_OpenLoopSettings):
    """Privileged external-broker process settings.

    Runtime provider credentials, Slack/GitHub credentials, worker controls,
    and receipt signing roots are deliberately not part of this schema.
    """

    model_config = SettingsConfigDict(
        env_file=".broker.env",
        env_file_encoding="utf-8",
        secrets_dir="/run/secrets",
        extra="ignore",
        hide_input_in_errors=True,
    )

    # Stable container targets. Deployments choose host sources independently.
    broker_control_socket_dir: str = str(
        DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR
    )
    broker_state_root: str = str(DEFAULT_EXTERNAL_BROKER_STATE_ROOT)
    broker_runtime_root: str = str(DEFAULT_EXTERNAL_BROKER_RUNTIME_ROOT)
    broker_ingress_root: str = str(DEFAULT_EXTERNAL_BROKER_INGRESS_ROOT)
    broker_checkpoint_receipt_root: str = str(
        DEFAULT_EXTERNAL_BROKER_CHECKPOINT_RECEIPT_ROOT
    )

    broker_identity_issuer: str = "openloop-app"
    broker_identity_audience: str = "openloop-broker"
    broker_capability_roots: dict[str, SecretStr] = Field(default_factory=dict)
    broker_capability_current_version: str = "cap-key-v1"
    broker_runtime_roots: dict[str, SecretStr] = Field(default_factory=dict)
    broker_runtime_current_version: str = "runtime-key-v1"
    broker_identity_public_keys: dict[str, str] = Field(default_factory=dict)
    broker_receipt_public_keys: dict[str, str] = Field(default_factory=dict)
    broker_receipt_domain: str = "broker-receipt"

    broker_shared_data_gid: int | None = None
    broker_expected_app_uid: int | None = None
    broker_execution_lease_seconds: int = 900
    broker_generation_deadline_seconds: int = 1800
    broker_reconcile_interval_seconds: int = Field(default=300, gt=0)
    broker_dev_in_memory: bool = False

    database_url: str = "postgresql://openloop:change-me@localhost:5432/openloop"
    postgres_password: SecretStr | None = None
    postgres_pool_min_size: int = Field(default=1, ge=0)
    postgres_pool_max_size: int = Field(default=10, ge=1)
    log_level: str = "info"

    @model_validator(mode="after")
    def _validate_postgres_pool_size(self) -> Self:
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError(
                "postgres_pool_min_size must be less than or equal to "
                "postgres_pool_max_size"
            )
        return self


class CoprocessBrokerSettings(_OpenLoopSettings):
    """Broker-service authority loaded only for an explicit coprocess topology.

    Coprocess paths remain explicit so a local runtime cannot accidentally
    claim the external container's fixed ``/var/lib/openloop/broker`` targets.
    """

    broker_control_socket_dir: str | None = None
    broker_state_root: str | None = None
    broker_runtime_root: str | None = None
    broker_capability_roots: dict[str, SecretStr] = Field(default_factory=dict)
    broker_capability_current_version: str = "cap-key-v1"
    broker_runtime_roots: dict[str, SecretStr] = Field(default_factory=dict)
    broker_runtime_current_version: str = "runtime-key-v1"
    broker_identity_issuer: str = "openloop-app"
    broker_identity_audience: str = "openloop-broker"
    broker_receipt_domain: str = "broker-receipt"
    broker_shared_data_gid: int | None = None
    broker_execution_lease_seconds: int = 900
    broker_generation_deadline_seconds: int = 1800
