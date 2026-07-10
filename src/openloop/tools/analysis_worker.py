"""Single-shot sealed analysis worker (Phase 1).

The model authors a Python program in the trusted controller.  Only execution
of that program happens in the sealed Docker sandbox; it receives staged input
through a read-only mount and can write only the output mount.  The
orchestrator is deliberately the one owner of input materialization, spend
gates, read-out, and artifact persistence.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from openloop.analysis import ArtifactStore, InputStore
from openloop.sandbox import (
    Mount,
    ReadOutViolation,
    SandboxLimits,
    SandboxResult,
    SealedSpec,
    read_contained,
)
from openloop.tools.base import ActionSpec, ToolResult
from openloop.usage import WorkerBudgetExceeded

if TYPE_CHECKING:
    from openloop.agents.schema import Agent
    from openloop.usage.ledger import WorkerSpendLedger


ANALYSIS_WORKER_TOOL_NAME = "analysis"
ANALYSIS_REPORT_WRITE = "report:write"
_REPORT_NAME = "report.md"

StepCallback = Callable[["AnalysisState"], Awaitable[None]]


@dataclass(slots=True)
class AnalysisState:
    """Replay-safe identity and request metadata for one analysis attempt."""

    job_id: str
    input_ref: str
    instruction: str
    agent: str | None = None
    completed_steps: list[str] = field(default_factory=list)
    # Reserved for the Phase 3 iterative strategy. It remains transient so no
    # budget setting can be model-supplied through persisted tool args.
    budget_usd: float | None = None


@dataclass(slots=True)
class AnalysisRun:
    """Execution telemetry returned by the credential-free worker only."""

    # ``None`` means the model completion succeeded but execution could not
    # produce a sandbox result (for example an empty program or docker setup
    # failure). Its known model spend still has to reach the ledger.
    exit_code: int | None
    stdout: str
    stderr: str
    killed: bool
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    duration_seconds: float
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_sandbox(cls, result: SandboxResult, *, response) -> "AnalysisRun":
        return cls(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            killed=result.killed,
            timed_out=result.timed_out,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            duration_seconds=result.duration_seconds,
            cost_usd=response.cost_usd,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    def telemetry(self) -> dict:
        """Telemetry fit for a tool result, intentionally excluding stream bytes.

        Captured stdout/stderr are controller-side diagnostics. Returning them
        through ``ToolResult.data`` would make them tool-loop input to the model,
        which is the forbidden single-shot execution-feedback channel.
        """
        return {
            "exit_code": self.exit_code,
            "killed": self.killed,
            "timed_out": self.timed_out,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class AnalysisWorkerFailure(RuntimeError):
    """A post-completion worker failure that still carries paid telemetry.

    A model provider can return a successful completion before local validation,
    checkpointing, or sandbox startup fails.  The orchestrator must settle that
    observed spend before reporting the failure, but must not read outputs.
    """

    def __init__(self, message: str, *, response) -> None:
        super().__init__(message)
        self.run = AnalysisRun(
            exit_code=None,
            stdout="",
            stderr="",
            killed=False,
            timed_out=False,
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.0,
            cost_usd=response.cost_usd,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )


@dataclass(slots=True)
class AnalysisResult:
    """The orchestrator's outcome; report bytes never appear here."""

    job_id: str
    input_ref: str
    run: AnalysisRun | None = None
    artifact_ref: str | None = None
    prose_summary: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.artifact_ref is not None

    def data(self) -> dict:
        data = {
            "job_id": self.job_id,
            "input_ref": self.input_ref,
            "artifact_ref": self.artifact_ref,
            "prose_summary": self.prose_summary,
        }
        if self.run is not None:
            data.update(self.run.telemetry())
        if self.error is not None:
            data["error"] = self.error
        return data


@runtime_checkable
class AnalysisWorker(Protocol):
    """Generates and executes code against an already-prepared workspace.

    Implementations have neither input-store nor artifact-store access, and
    return execution telemetry only. That keeps the read-out/exfiltration
    boundary solely in :class:`SealedAnalysisOrchestrator`.
    """

    async def run(
        self,
        workspace: Path,
        state: AnalysisState,
        on_step: StepCallback | None = None,
    ) -> AnalysisRun: ...


@runtime_checkable
class AnalysisAttemptRunner(Protocol):
    async def run_analysis(
        self, state: AnalysisState, on_step: StepCallback | None = None
    ) -> AnalysisResult: ...


@runtime_checkable
class _Completer(Protocol):
    async def complete(self, model: str, messages: list[dict], **kwargs): ...


@runtime_checkable
class _SealedSandbox(Protocol):
    async def run(self, spec: SealedSpec) -> SandboxResult: ...


class BuiltinAnalysisWorker:
    """One controller completion plus one sealed Python execution.

    The worker has one Phase 1 strategy: it writes a whole program in one
    completion and streams it over stdin to ``python -``. Iteration and any
    model-visible execution feedback are deliberately deferred to Phase 3.
    """

    def __init__(
        self,
        model: str,
        sandbox: _SealedSandbox,
        *,
        gateway: _Completer | None = None,
        limits: SandboxLimits,
        output_cap_bytes: int,
        output_watch_interval_seconds: float = 2.0,
    ) -> None:
        self.model = model
        self.sandbox = sandbox
        self._gateway = gateway
        self.limits = limits
        self.output_cap_bytes = output_cap_bytes
        self.output_watch_interval_seconds = output_watch_interval_seconds

    def _completer(self) -> _Completer:
        if self._gateway is None:
            from openloop.models.gateway import ModelGateway

            self._gateway = ModelGateway()
        return self._gateway

    async def run(
        self,
        workspace: Path,
        state: AnalysisState,
        on_step: StepCallback | None = None,
    ) -> AnalysisRun:
        script, response = await self._generate(state, workspace / "inputs")
        try:
            if not script.strip():
                raise AnalysisWorkerFailure(
                    "analysis model returned an empty program", response=response
                )
            if not script.endswith("\n"):
                script += "\n"

            state.completed_steps.append("generate")
            if on_step is not None:
                await on_step(state)

            outputs = workspace / "outputs"
            result = await self.sandbox.run(
                SealedSpec(
                    job_id=state.job_id,
                    command=("python", "-"),
                    limits=self.limits,
                    mounts=(
                        Mount(workspace / "inputs", "/workspace/inputs", read_only=True),
                        Mount(outputs, "/workspace/outputs"),
                    ),
                    # Generated code is never placed in either mounted directory.
                    stdin=script,
                    watch_dir=outputs,
                    watch_max_bytes=self.output_cap_bytes,
                    watch_interval_seconds=self.output_watch_interval_seconds,
                )
            )
            state.completed_steps.append("execute")
            if on_step is not None:
                await on_step(state)
            return AnalysisRun.from_sandbox(result, response=response)
        except AnalysisWorkerFailure:
            raise
        except Exception as exc:
            raise AnalysisWorkerFailure(
                f"analysis execution setup failed: {exc}", response=response
            ) from exc

    async def _generate(self, state: AnalysisState, inputs: Path):
        inventory = []
        for path in sorted(inputs.iterdir()):
            if path.is_file():
                inventory.append(f"- {path.name} ({path.stat().st_size} bytes)")
        listed_inputs = "\n".join(inventory) or "- (no files staged)"
        response = await self._completer().complete(
            self.model,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a sealed data-analysis worker. Produce exactly one "
                        "complete Python program, with no Markdown fences or prose. "
                        "The program has no network, no credentials, and no package "
                        "installation. Read only from /workspace/inputs and write the "
                        "final UTF-8 Markdown report exactly to "
                        "/workspace/outputs/report.md. Do not write anywhere else. "
                        "You may use the preinstalled pandas, numpy, and matplotlib "
                        "(Agg backend). Inspect input contents at runtime as needed."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analysis request:\n{state.instruction}\n\n"
                        f"Staged input inventory (names and sizes only):\n{listed_inputs}"
                    ),
                },
            ],
        )
        return _strip_code_fence(response.text), response


class SealedAnalysisOrchestrator:
    """The sole provision-in / settle / read-out boundary for analysis.

    No generated code gets an input-store handle, an artifact-store handle, a
    credential, or a network. No report bytes are read until the per-task spend
    settlement succeeds.
    """

    def __init__(
        self,
        worker: AnalysisWorker,
        inputs: InputStore,
        artifacts: ArtifactStore,
        *,
        ledger: "WorkerSpendLedger | None" = None,
        workspace_root: Path | None = None,
        report_max_bytes: int = 1_000_000,
        summary_lines: int = 12,
    ) -> None:
        self.worker = worker
        self._inputs = inputs
        self._artifacts = artifacts
        self._ledger = ledger
        self._workspace_root = workspace_root
        self.report_max_bytes = report_max_bytes
        self.summary_lines = summary_lines

    def per_task_usd_for(self, agent: str | None) -> float | None:
        return self._ledger.per_task_usd_for(agent) if self._ledger else None

    async def run_analysis(
        self, state: AnalysisState, on_step: StepCallback | None = None
    ) -> AnalysisResult:
        async def step(name: str) -> None:
            state.completed_steps.append(name)
            if on_step is not None:
                await on_step(state)

        async def settle(run: AnalysisRun) -> WorkerBudgetExceeded | None:
            """Record known model spend once, before any possible read-out."""
            if self._ledger is None:
                return None
            try:
                await self._ledger.settle(
                    agent=state.agent,
                    job_id=state.job_id,
                    cost_usd=run.cost_usd,
                    prompt_tokens=run.prompt_tokens,
                    completion_tokens=run.completion_tokens,
                )
            except WorkerBudgetExceeded as exc:
                return exc
            return None

        # This gate precedes input-store access and workspace allocation. A
        # monthly refusal must not provision data or start a model call.
        try:
            if self._ledger is not None:
                await self._ledger.check_monthly(state.agent, job_id=state.job_id)
                state.budget_usd = self._ledger.per_task_usd_for(state.agent)
        except WorkerBudgetExceeded as exc:
            return _failed(state, str(exc))

        manifest = await self._inputs.get(state.job_id, state.input_ref)
        if manifest is None:
            return _failed(
                state,
                "no staged input matches this job_id and input_ref",
            )

        if self._workspace_root is not None:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(prefix="openloop-analysis-", dir=self._workspace_root)
        )
        run: AnalysisRun | None = None
        try:
            manifest.materialize(workspace / "inputs")
            (workspace / "outputs").mkdir()
            await step("materialize")

            try:
                run = await self.worker.run(workspace, state, on_step)
            except AnalysisWorkerFailure as exc:
                # A completion succeeded, so the provider's observed spend is
                # real even though local validation or sandbox setup failed.
                run = exc.run
                if budget_error := await settle(run):
                    return _failed(state, str(budget_error), run=run)
                return _failed(state, f"analysis worker failed before execution: {exc}", run=run)
            except Exception as exc:  # model/backend failures before telemetry
                return _failed(state, f"analysis worker failed before execution: {exc}")

            # Settlement happens for both success and execution failure; model
            # spend is real either way. A failed settlement is always terminal
            # before read-out.
            if budget_error := await settle(run):
                return _failed(state, str(budget_error), run=run)

            if run.exit_code != 0:
                return _failed(state, _execution_error(run), run=run)

            try:
                body, truncated = read_contained(
                    workspace / "outputs", _REPORT_NAME, max_bytes=self.report_max_bytes
                )
            except (FileNotFoundError, OSError, ReadOutViolation) as exc:
                return _failed(state, f"analysis report read-out refused: {exc}", run=run)
            if truncated:
                return _failed(
                    state,
                    f"analysis report exceeds the {self.report_max_bytes}-byte read-out cap",
                    run=run,
                )
            try:
                report = body.decode("utf-8")
            except UnicodeDecodeError:
                return _failed(
                    state, "analysis report must be UTF-8 Markdown", run=run
                )

            await step("read_out")
            artifact_ref = await self._artifacts.put(state.job_id, body)
            await step("store_artifact")
            return AnalysisResult(
                job_id=state.job_id,
                input_ref=state.input_ref,
                run=run,
                artifact_ref=artifact_ref,
                prose_summary=_mechanical_summary(report, self.summary_lines),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


class AnalysisWorkerConnector:
    """Maps ``analysis.report:write`` onto the sealed attempt boundary."""

    name = ANALYSIS_WORKER_TOOL_NAME
    # Product policy for Phase 1: an agent cannot silently opt this action out
    # of human approval. A future spend-only policy may make this per-agent.
    requires_approval = True

    def __init__(self, orchestrator: AnalysisAttemptRunner) -> None:
        self.orchestrator = orchestrator

    def supported_permissions(self) -> set[str]:
        return {ANALYSIS_REPORT_WRITE}

    def describe(self, permission: str) -> ActionSpec:
        if permission != ANALYSIS_REPORT_WRITE:
            raise ValueError(f"unsupported analysis permission {permission!r}")
        return ActionSpec(
            description=(
                "Run sealed analysis over a pre-staged input reference and return "
                "a report artifact reference. Requires human approval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "The analysis question to answer.",
                    },
                    "input_ref": {
                        "type": "string",
                        "description": "Reference to controller-staged input data.",
                    },
                },
                "required": ["instruction", "input_ref"],
                "additionalProperties": False,
            },
        )

    def prepare_args(
        self,
        permission: str,
        args: dict,
        agent: "Agent | None" = None,
        *,
        warm_key: str | None = None,
    ) -> dict:
        """Mint identity and strip anything other than Phase 1 safe fields.

        Creating a fresh dict is a backstop against direct API callers smuggling
        raw data into the approval/workflow record through unknown fields.
        ``warm_key`` is intentionally irrelevant: Phase 1 has no durable
        workspace reuse and must not retain provisioned data between runs.
        """
        if permission != ANALYSIS_REPORT_WRITE:
            return args
        return {
            "job_id": args.get("job_id") or uuid.uuid4().hex,
            "instruction": str(args.get("instruction") or "").strip(),
            "input_ref": (
                args.get("input_ref") if isinstance(args.get("input_ref"), str) else ""
            ),
            "agent": agent.metadata.name if agent is not None else None,
        }

    async def execute(self, permission: str, args: dict) -> ToolResult:
        if permission != ANALYSIS_REPORT_WRITE:
            return ToolResult(ok=False, summary=f"unsupported permission {permission}")
        # Gateway.invoke() has already called prepare_args before persisting an
        # approval request. Do not call it again here: doing so without the
        # trusted Agent object would erase the stamped invoking identity and
        # misattribute spend after approval.
        job_id = args.get("job_id") or uuid.uuid4().hex
        input_ref = args.get("input_ref") if isinstance(args.get("input_ref"), str) else ""
        instruction = str(args.get("instruction") or "").strip()
        agent = args.get("agent") if isinstance(args.get("agent"), str) else None
        state = AnalysisState(
            job_id=str(job_id),
            input_ref=input_ref,
            instruction=instruction,
            agent=agent,
        )
        if not state.instruction:
            return _tool_failure(_failed(state, "analysis instruction is required"))
        if not state.input_ref:
            return _tool_failure(_failed(state, "input_ref is required"))
        try:
            result = await self.orchestrator.run_analysis(state)
        except Exception as exc:  # connector boundary: never leak an exception past approval
            result = _failed(state, f"analysis orchestration failed: {exc}")
        if not result.ok:
            return _tool_failure(result)
        return ToolResult(
            ok=True,
            summary=result.prose_summary or "sealed analysis completed",
            data=result.data(),
        )


def _failed(
    state: AnalysisState, error: str, *, run: AnalysisRun | None = None
) -> AnalysisResult:
    return AnalysisResult(
        job_id=state.job_id,
        input_ref=state.input_ref,
        run=run,
        error=error,
    )


def _tool_failure(result: AnalysisResult) -> ToolResult:
    return ToolResult(
        ok=False,
        summary=f"sealed analysis job {result.job_id} failed: {result.error}",
        data=result.data(),
    )


def _execution_error(run: AnalysisRun) -> str:
    detail = f"analysis program exited with code {run.exit_code}"
    if run.timed_out:
        detail += " (timed out)"
    elif run.killed:
        detail += " (killed by sandbox resource watchdog)"
    if run.stdout_truncated or run.stderr_truncated:
        detail += "; diagnostic streams were truncated"
    return detail


def _mechanical_summary(report: str, max_lines: int) -> str:
    lines = report.splitlines()[:max_lines]
    summary = "\n".join(lines).strip()
    return summary or "Analysis completed; the report is available as an artifact."


def _strip_code_fence(script: str) -> str:
    """Tolerate a fenced model response without trying to parse Python."""
    stripped = script.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1])
    return script
