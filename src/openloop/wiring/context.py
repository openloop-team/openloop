"""Immutable application ownership records for the composition root."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openloop.agents.schema import Agent
from openloop.approvals import ApprovalStore
from openloop.checkpoints import CheckpointStore
from openloop.config import Settings
from openloop.coordination import DistributedLock
from openloop.memory import Embedder, MemoryStore
from openloop.runtime import Runtime
from openloop.sessions import SurfaceSessionStore, ThreadRecordStore
from openloop.tools import ToolGateway
from openloop.usage import TaskLimiter, UsageStore
from openloop.workflows import WorkflowEngine, WorkflowStore


@dataclass(frozen=True, slots=True)
class SettledStores:
    """Final store instances. Dependents may be built only from this bundle."""

    memory: MemoryStore
    usage: UsageStore
    approvals: ApprovalStore
    checkpoints: CheckpointStore
    workflows: WorkflowStore
    sessions: SurfaceSessionStore
    threads: ThreadRecordStore


@dataclass(slots=True)
class AgentRuntimes:
    """Lazy runtime registry preserving the current HTTP and Slack selectors."""

    loaded: dict[str, Agent]
    stores: SettledStores
    embedder: Embedder | None
    tools: ToolGateway
    engine: WorkflowEngine
    limiter: TaskLimiter
    model_gateway: Any | None = None
    _slack_runtime: Runtime | None = field(default=None, init=False, repr=False)

    @property
    def primary(self) -> Agent | None:
        """The first configured agent; reading it constructs nothing."""
        return next(iter(self.loaded.values()), None)

    @property
    def slack_agent(self) -> Agent | None:
        return next(
            (agent for agent in self.loaded.values() if agent.has_slack_surface()),
            None,
        )

    def slack_runtime(self) -> Runtime | None:
        """Construct exactly the one runtime the Slack surface needs, lazily."""
        agent = self.slack_agent
        if agent is None:
            return None
        if self._slack_runtime is None:
            self._slack_runtime = Runtime(
                agent,
                gateway=self.model_gateway,
                memory=self.stores.memory,
                embedder=self.embedder,
                usage=self.stores.usage,
                tools=self.tools,
                engine=self.engine,
                limiter=self.limiter,
            )
        return self._slack_runtime


@dataclass(frozen=True, slots=True)
class AppContext:
    """Fully composed application graph and the resources that own its lifetime."""

    settings: Settings
    agents: AgentRuntimes
    stores: SettledStores
    embedder: Embedder | None
    limiter: TaskLimiter
    engine: WorkflowEngine
    tools: ToolGateway
    coordinator: DistributedLock
    slack_app: Any | None
    session_runner: Any | None
    slack_handler: Any | None
    postgres_pool: Any | None
    recovery_task: Any | None = None
    warm_sweep_task: Any | None = None

    @property
    def memory(self) -> MemoryStore:
        return self.stores.memory

    @property
    def usage(self) -> UsageStore:
        return self.stores.usage

    @property
    def approvals(self) -> ApprovalStore:
        return self.stores.approvals

    @property
    def checkpoints(self) -> CheckpointStore:
        return self.stores.checkpoints

    @property
    def workflows(self) -> WorkflowStore:
        return self.stores.workflows

    @property
    def sessions(self) -> SurfaceSessionStore:
        return self.stores.sessions

    @property
    def threads(self) -> ThreadRecordStore:
        return self.stores.threads
