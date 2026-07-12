"""Sealed analysis worker — single-shot (Phase 1) and iterative (Phase 3).

The model authors a Python program in the trusted controller.  Only execution
of that program happens in the sealed Docker sandbox; it receives staged input
through a read-only mount and can write only the output mount.  The
orchestrator is deliberately the one owner of input materialization, spend
gates, read-out, and artifact persistence.

Strategies stay internal to :class:`BuiltinAnalysisWorker` (never sibling
classes).  ``single`` is one completion + one sealed run.  ``iterative`` (the
default) loops
generate → execute → feed capped stdout/stderr back to the in-controller model
(**exec_feedback** — governed by the in-run spend cap and hard truncation,
never posted to a surface) → refine, bounded by ``max_iterations`` and a
:class:`~openloop.tools.coding_worker.WorkerRunAborted` raised past the
per-task cap the orchestrator stamps into ``AnalysisState.budget_usd``.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import ValidationError

from openloop.analysis import (
    ANALYSIS_ARGS_VERSION,
    AnalysisAttemptStore,
    AnalysisReportArgs,
    ArtifactStore,
    ExecutableAnalysisRequest,
    InMemoryAnalysisAttemptStore,
    ProvisionError,
    Provisioner,
    RequestIdentity,
    UploadStore,
    materialize_inputs,
    provision_inputs,
)
from openloop.sandbox import (
    Mount,
    ReadOutViolation,
    SandboxLimits,
    SandboxResult,
    SealedSpec,
    read_contained,
)
from openloop.tools.base import ActionSpec, ToolResult, format_validation_error
from openloop.tools.coding_worker import WorkerRunAborted
from openloop.usage import WorkerBudgetExceeded

if TYPE_CHECKING:
    from openloop.agents.schema import Agent
    from openloop.usage.ledger import WorkerSpendLedger


ANALYSIS_WORKER_TOOL_NAME = "analysis"
ANALYSIS_REPORT_WRITE = "report:write"
_REPORT_NAME = "report.md"

StepCallback = Callable[["AnalysisState"], Awaitable[None]]
ChargeCallback = Callable[["AnalysisCharge"], Awaitable[None]]


@dataclass(slots=True)
class AnalysisState:
    """Replay-safe identity and request metadata for one analysis attempt.

    Identity (job/attempt/agent/scope) is lenient by construction so attempt
    reconciliation is never blocked; whether ``instruction``/``inputs`` can
    actually execute is decided by the orchestrator's
    :class:`~openloop.analysis.ExecutableAnalysisRequest` parse.
    """

    job_id: str
    instruction: str
    # Raw parsed-and-dumped ``inputs[]`` entries from the durable record; the
    # orchestrator re-parses them through the current args contract.
    inputs: list[dict] = field(default_factory=list)
    agent: str | None = None
    # The full thread-ownership tuple key stamped gateway-side from the
    # session context; None on scopeless paths (the direct tools API).
    scope_key: str | None = None
    # The args-contract version the durable record was written under; None is
    # the pre-version sentinel and always refuses execution.
    args_schema: int | None = None
    # Minted before approval in Phase 1b; direct harnesses can omit it and the
    # orchestrator will mint one before the first model call.
    attempt_id: str | None = None
    completed_steps: list[str] = field(default_factory=list)
    # The invoking agent's per-task cap, stamped by the orchestrator before the
    # worker runs so the worker can stop itself (WorkerRunAborted) instead of
    # spending past it. It remains transient so no budget setting can be
    # model-supplied through persisted tool args.
    budget_usd: float | None = None


@dataclass(slots=True, frozen=True)
class AnalysisCharge:
    """Observed provider usage for one successful completion."""

    cost_usd: float
    prompt_tokens: int
    completion_tokens: int

    @classmethod
    def from_response(cls, response) -> "AnalysisCharge":
        return cls(
            cost_usd=response.cost_usd,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    @classmethod
    def from_run(cls, run: "AnalysisRun") -> "AnalysisCharge":
        return cls(
            cost_usd=run.cost_usd,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
        )


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
    # Model completions this run consumed (1 for single-shot; the iterative
    # strategy reports how many refinement rounds actually ran).
    iterations: int = 1

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
            "iterations": self.iterations,
        }


class AnalysisWorkerFailure(RuntimeError):
    """A post-completion worker failure that still carries paid telemetry.

    A model provider can return a successful completion before local validation,
    checkpointing, or sandbox startup fails.  The orchestrator must settle that
    observed spend before reporting the failure, but must not read outputs.
    """

    def __init__(self, message: str, *, charge: AnalysisCharge) -> None:
        super().__init__(message)
        self.charge = charge
        self.run = _spend_only_run(
            cost_usd=charge.cost_usd,
            prompt_tokens=charge.prompt_tokens,
            completion_tokens=charge.completion_tokens,
        )


def _spend_only_run(
    *, cost_usd: float, prompt_tokens: int, completion_tokens: int
) -> AnalysisRun:
    """Telemetry for an attempt that never produced a sandbox result."""
    return AnalysisRun(
        exit_code=None,
        stdout="",
        stderr="",
        killed=False,
        timed_out=False,
        stdout_truncated=False,
        stderr_truncated=False,
        duration_seconds=0.0,
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


@dataclass(slots=True)
class AnalysisResult:
    """The orchestrator's outcome; report bytes never appear here."""

    job_id: str
    attempt_id: str | None = None
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
            "attempt_id": self.attempt_id,
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
        on_charge: ChargeCallback | None = None,
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
    """Controller-side generation plus sealed Python execution.

    ``strategy="iterative"`` (Phase 3, the default) loops generate → execute →
    **exec_feedback** (capped stdout/stderr shown to the in-controller model)
    → refine.  The loop ends mechanically — a run that exits 0 with
    ``outputs/report.md`` present — never on the model's say-so, and is
    bounded by ``max_iterations`` plus the in-run spend guard.

    ``strategy="single"`` (Phase 1) writes a whole program in one completion
    and streams it over stdin to ``python -``; captured stdout/stderr stay
    controller telemetry and are never fed back to the model.

    Both strategies report each completion's cumulative spend through
    ``on_charge`` **before** doing further work (durable retention), and stop
    themselves with :class:`WorkerRunAborted` once cumulative spend passes
    ``state.budget_usd`` — the ledger's post-run settle stays the fail-closed
    backstop.
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
        strategy: str = "iterative",
        max_iterations: int = 4,
        exec_feedback_max_chars: int = 16_384,
    ) -> None:
        if strategy not in ("single", "iterative"):
            raise ValueError(f"unknown analysis strategy {strategy!r}")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if exec_feedback_max_chars < 1:
            raise ValueError("exec_feedback_max_chars must be positive")
        self.model = model
        self.sandbox = sandbox
        self._gateway = gateway
        self.limits = limits
        self.output_cap_bytes = output_cap_bytes
        self.output_watch_interval_seconds = output_watch_interval_seconds
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.exec_feedback_max_chars = exec_feedback_max_chars

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
        on_charge: ChargeCallback | None = None,
    ) -> AnalysisRun:
        if self.strategy == "iterative":
            return await self._run_iterative(workspace, state, on_step, on_charge)
        return await self._run_single(workspace, state, on_step, on_charge)

    async def _run_single(
        self,
        workspace: Path,
        state: AnalysisState,
        on_step: StepCallback | None,
        on_charge: ChargeCallback | None,
    ) -> AnalysisRun:
        script, response = await self._generate(state, workspace / "inputs")
        charge = AnalysisCharge.from_response(response)
        try:
            # Retention happens before any generated-code validation, sandbox
            # startup, or output read. A later crash/retry sees the durable
            # attempt record, never a free completion; the one idempotent
            # ledger settle is the orchestrator's, after the run.
            if on_charge is not None:
                await on_charge(charge)
            self._abort_past_cap(state, charge)
            if not script.strip():
                raise AnalysisWorkerFailure(
                    "analysis model returned an empty program", charge=charge
                )
            if not script.endswith("\n"):
                script += "\n"

            state.completed_steps.append("generate")
            if on_step is not None:
                await on_step(state)

            result = await self.sandbox.run(self._sealed_spec(state, workspace, script))
            state.completed_steps.append("execute")
            if on_step is not None:
                await on_step(state)
            return AnalysisRun.from_sandbox(result, response=response)
        except (AnalysisWorkerFailure, WorkerRunAborted):
            raise
        except Exception as exc:
            raise AnalysisWorkerFailure(
                f"analysis execution setup failed: {exc}", charge=charge
            ) from exc

    async def _run_iterative(
        self,
        workspace: Path,
        state: AnalysisState,
        on_step: StepCallback | None,
        on_charge: ChargeCallback | None,
    ) -> AnalysisRun:
        report = workspace / "outputs" / _REPORT_NAME
        messages = self._iterative_prompt(state, workspace / "inputs")
        cost_usd = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        duration_seconds = 0.0
        result: SandboxResult | None = None
        rounds = 0
        for round_no in range(1, self.max_iterations + 1):
            rounds = round_no
            try:
                response = await self._completer().complete(self.model, messages)
            except Exception as exc:
                if round_no == 1:
                    # No completion has been observed yet: parity with the
                    # single-shot generation failure — the orchestrator's
                    # pre-telemetry handler applies, nothing to settle.
                    raise
                # Rounds 1..N-1 were billed and durably retained; this failure
                # must carry that cumulative total out so the orchestrator
                # settles it instead of leaving the attempt charged forever.
                raise AnalysisWorkerFailure(
                    f"analysis model call failed after {round_no - 1} "
                    f"completed round(s): {exc}",
                    charge=AnalysisCharge(
                        cost_usd=cost_usd,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    ),
                ) from exc
            cost_usd += response.cost_usd
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            # Every charge is the CUMULATIVE attempt total: the durable attempt
            # record then always holds exactly what a crash-resume must settle,
            # and the final ledger settle uses the same figure.
            charge = AnalysisCharge(
                cost_usd=cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            script = _strip_code_fence(response.text)
            try:
                if on_charge is not None:
                    await on_charge(charge)
                # After retention, before more spend or another sealed run: a
                # cumulative total past the cap can only fail the post-run
                # settle, so executing this round's program would be waste.
                self._abort_past_cap(state, charge)
                if not script.strip():
                    raise AnalysisWorkerFailure(
                        "analysis model returned an empty program", charge=charge
                    )
                if not script.endswith("\n"):
                    script += "\n"

                state.completed_steps.append("generate")
                if on_step is not None:
                    await on_step(state)

                # A report left by a failed earlier round is discarded so the
                # loop can only end on a report the succeeding run wrote.
                _discard_report(report)
                result = await self.sandbox.run(
                    self._sealed_spec(state, workspace, script)
                )
                duration_seconds += result.duration_seconds
                state.completed_steps.append("execute")
                if on_step is not None:
                    await on_step(state)

                report_written = report.is_file()
                if result.exit_code == 0 and report_written:
                    break
                if round_no < self.max_iterations:
                    messages.append(
                        {"role": "assistant", "content": response.text}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._exec_feedback(
                                result, round_no, report_written
                            ),
                        }
                    )
            except (AnalysisWorkerFailure, WorkerRunAborted):
                raise
            except Exception as exc:
                # The whole round body is guarded: once this round's charge is
                # retained, no failure (checkpoint callback included) may
                # escape without carrying the cumulative total to the settle.
                raise AnalysisWorkerFailure(
                    f"analysis execution setup failed: {exc}", charge=charge
                ) from exc
        # An exhausted loop returns the last run as-is; the orchestrator's
        # exit-code and read-out gates produce the truthful failure.
        assert result is not None  # max_iterations >= 1
        return AnalysisRun(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            killed=result.killed,
            timed_out=result.timed_out,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            duration_seconds=duration_seconds,
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            iterations=rounds,
        )

    def _abort_past_cap(self, state: AnalysisState, charge: AnalysisCharge) -> None:
        """Stop the run once cumulative spend passes the stamped per-task cap.

        First-hand in-run enforcement (the openhands guard's analog); the
        orchestrator's post-run ledger settle remains the fail-closed backstop
        should the stamped cap and the agent's config ever disagree.
        """
        if state.budget_usd is not None and charge.cost_usd > state.budget_usd:
            raise WorkerRunAborted(
                f"in-run spend ${charge.cost_usd:.4f} reached the "
                f"${state.budget_usd:.2f} per-task cap",
                cost_usd=charge.cost_usd,
                prompt_tokens=charge.prompt_tokens,
                completion_tokens=charge.completion_tokens,
            )

    def _sealed_spec(
        self, state: AnalysisState, workspace: Path, script: str
    ) -> SealedSpec:
        outputs = workspace / "outputs"
        return SealedSpec(
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

    async def _generate(self, state: AnalysisState, inputs: Path):
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
                {"role": "user", "content": self._request(state, inputs)},
            ],
        )
        return _strip_code_fence(response.text), response

    def _iterative_prompt(self, state: AnalysisState, inputs: Path) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a sealed data-analysis worker operating in a "
                    "refinement loop. In every round, produce exactly one "
                    "complete Python program, with no Markdown fences or prose. "
                    "The program has no network, no credentials, and no package "
                    "installation. Read only from /workspace/inputs and write "
                    "only under /workspace/outputs. You may use the preinstalled "
                    "pandas, numpy, and matplotlib (Agg backend). After each "
                    "round you are shown the program's exit code and truncated "
                    "stdout/stderr. Use early rounds to inspect the data (print "
                    "schemas, samples, aggregates). When the analysis is "
                    "complete, write the final UTF-8 Markdown report exactly to "
                    "/workspace/outputs/report.md and exit 0 — the loop ends as "
                    "soon as a run exits 0 with report.md present, so write "
                    "report.md only in your final round. A report.md left by a "
                    "failed run is discarded before the next round. You have "
                    f"{self.max_iterations} rounds in total."
                ),
            },
            {"role": "user", "content": self._request(state, inputs)},
        ]

    def _request(self, state: AnalysisState, inputs: Path) -> str:
        inventory = []
        for path in sorted(inputs.iterdir()):
            if path.is_file():
                inventory.append(f"- {path.name} ({path.stat().st_size} bytes)")
        listed_inputs = "\n".join(inventory) or "- (no files staged)"
        return (
            f"Analysis request:\n{state.instruction}\n\n"
            f"Staged input inventory (names and sizes only):\n{listed_inputs}"
        )

    def _exec_feedback(
        self, result: SandboxResult, round_no: int, report_written: bool
    ) -> str:
        """One round's execution feedback for the in-controller model.

        This is the exec_feedback channel: hard-truncated here, bounded
        overall by the in-run spend guard, and never posted to a surface.
        The output-was-cut notes are load-bearing — the model must know it is
        not seeing everything, whether the cut happened at capture time
        (stream cap) or here (feedback cap).
        """
        status = f"exited with code {result.exit_code}"
        if result.timed_out:
            status += " (killed at the wall-clock deadline)"
        elif result.killed:
            status += " (killed by the sandbox resource watchdog)"
        report_line = (
            "report.md was written, but a run must exit 0 for it to be "
            "accepted; it will be discarded before your next run"
            if report_written
            else "/workspace/outputs/report.md does not exist yet"
        )
        return "\n\n".join(
            (
                f"Round {round_no} of {self.max_iterations}: your program "
                f"{status}. {report_line}.",
                self._feedback_stream("stdout", result.stdout, result.stdout_truncated),
                self._feedback_stream("stderr", result.stderr, result.stderr_truncated),
                "Respond with the next complete Python program (no fences, no "
                "prose).",
            )
        )

    def _feedback_stream(self, name: str, text: str, capture_truncated: bool) -> str:
        cut = capture_truncated or len(text) > self.exec_feedback_max_chars
        shown = text[: self.exec_feedback_max_chars]
        if not shown:
            return f"--- {name}: (empty) ---"
        header = (
            f"--- {name} (output was cut; this is only the beginning) ---"
            if cut
            else f"--- {name} ---"
        )
        return f"{header}\n{shown}"


class SealedAnalysisOrchestrator:
    """The sole provision-in / settle / read-out boundary for analysis.

    No generated code gets a provisioner handle, an artifact-store handle, a
    credential, or a network. No report bytes are read until the per-task spend
    settlement succeeds. As the one credentialed boundary it also owns the
    provisioner seam (Phase 4): each ``inputs[]`` entry is materialized here —
    after the approval resolves and the monthly gate passes, never earlier —
    under one merged, incrementally-enforced byte budget.
    """

    def __init__(
        self,
        worker: AnalysisWorker,
        provisioners: "list[Provisioner]",
        artifacts: ArtifactStore,
        *,
        attempts: AnalysisAttemptStore | None = None,
        ledger: "WorkerSpendLedger | None" = None,
        workspace_root: Path | None = None,
        report_max_bytes: int = 1_000_000,
        summary_lines: int = 12,
        max_input_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self.worker = worker
        self._provisioners: dict[str, Provisioner] = {
            p.source: p for p in provisioners
        }
        self._artifacts = artifacts
        self._attempts = attempts or InMemoryAnalysisAttemptStore()
        self._ledger = ledger
        self._workspace_root = workspace_root
        self.report_max_bytes = report_max_bytes
        self.summary_lines = summary_lines
        self.max_input_bytes = max_input_bytes

    def per_task_usd_for(self, agent: str | None) -> float | None:
        return self._ledger.per_task_usd_for(agent) if self._ledger else None

    async def run_analysis(
        self, state: AnalysisState, on_step: StepCallback | None = None
    ) -> AnalysisResult:
        async def step(name: str) -> None:
            state.completed_steps.append(name)
            if on_step is not None:
                await on_step(state)

        async def record(charge: AnalysisCharge) -> None:
            """Durably retain the run's cumulative observed usage — no settle.

            The worker calls this after every completion (cumulatively), so a
            crash at any point leaves the attempt record holding exactly what
            reconciliation must settle. The ledger settle itself happens once,
            after the run: the usage store's idempotency key is insert-ignore,
            so per-completion settles would freeze the audit row at the first
            iteration's totals.
            """
            assert state.attempt_id is not None
            await self._attempts.charge(
                state.attempt_id,
                cost_usd=charge.cost_usd,
                prompt_tokens=charge.prompt_tokens,
                completion_tokens=charge.completion_tokens,
            )

        async def account(charge: AnalysisCharge) -> None:
            """Persist observed usage, settle it idempotently, then mark done."""
            await record(charge)
            budget_error: WorkerBudgetExceeded | None = None
            if self._ledger is not None:
                try:
                    await self._ledger.settle(
                        agent=state.agent,
                        job_id=state.job_id,
                        idempotency_key=state.attempt_id,
                        cost_usd=charge.cost_usd,
                        prompt_tokens=charge.prompt_tokens,
                        completion_tokens=charge.completion_tokens,
                    )
                except WorkerBudgetExceeded as exc:
                    # The usage row was written before the cap error. Mark the
                    # attempt settled, then keep the sandbox/read-out blocked.
                    budget_error = exc
            await self._attempts.settle(state.attempt_id)
            if budget_error is not None:
                raise budget_error

        async def settle_or_fail(
            charge: AnalysisCharge, run: AnalysisRun | None = None
        ) -> AnalysisResult | None:
            """Account known spend; a failure result if that ends the attempt."""
            try:
                await account(charge)
            except WorkerBudgetExceeded as budget_error:
                return _failed(state, str(budget_error), run=run)
            except Exception as accounting_error:
                await self._mark_unknown(state, accounting_error)
                return _failed(
                    state,
                    "known analysis spend could not be durably accounted: "
                    f"{accounting_error}",
                    run=run,
                )
            return None

        state.attempt_id = state.attempt_id or uuid.uuid4().hex
        existing = await self._attempts.get(state.attempt_id)
        if existing is not None:
            # A charged checkpoint contains enough provider telemetry to finish
            # its accounting safely. This is deliberately *not* a computation
            # retry: no inputs, model, sandbox, or report are touched.
            if existing.status == "charged":
                if (
                    existing.cost_usd is None
                    or existing.prompt_tokens is None
                    or existing.completion_tokens is None
                ):
                    error = RuntimeError(
                        f"analysis attempt {existing.attempt_id} is charged "
                        "without complete usage telemetry"
                    )
                    await self._mark_unknown(state, error)
                    return _failed(
                        state,
                        "known analysis spend has incomplete durable telemetry; "
                        "operator reconciliation is required",
                    )
                failure = await settle_or_fail(
                    AnalysisCharge(
                        cost_usd=existing.cost_usd,
                        prompt_tokens=existing.prompt_tokens,
                        completion_tokens=existing.completion_tokens,
                    )
                )
                if failure is not None:
                    return failure
                return _failed(
                    state,
                    f"analysis attempt {existing.attempt_id} was already charged; "
                    "spend was settled without re-executing computation",
                )
            return _failed(
                state,
                f"analysis attempt {existing.attempt_id} is already {existing.status}; "
                "automatic re-execution is refused pending reconciliation",
            )

        # The spend-boundary parse (typed-tool-args §3.5), replacing the old
        # hand-written instruction/input_ref guard: args can also reach this
        # boundary from persisted records (an approval or parked workflow
        # written before a schema change) or from future direct callers, and a
        # request that cannot execute must never reach the ledger, a
        # provisioner, or a model. Deliberately after the attempt
        # reconciliation above — settling already-observed spend is correct
        # whatever the args look like. The version gate refuses records
        # written under any other contract, including the NULL pre-version
        # sentinel, instead of running over misinterpreted args.
        if state.args_schema != ANALYSIS_ARGS_VERSION:
            return _failed(
                state,
                "this analysis record predates args schema "
                f"v{ANALYSIS_ARGS_VERSION}; please request the analysis again",
            )
        try:
            request = ExecutableAnalysisRequest.model_validate(
                {"instruction": state.instruction, "inputs": state.inputs}
            )
        except ValidationError as exc:
            return _failed(
                state,
                "analysis request is not executable: "
                f"{format_validation_error(exc)}",
            )

        # This gate precedes provisioning and workspace allocation. A monthly
        # refusal must not fetch data or start a model call.
        try:
            if self._ledger is not None:
                await self._ledger.check_monthly(state.agent, job_id=state.job_id)
                state.budget_usd = self._ledger.per_task_usd_for(state.agent)
        except WorkerBudgetExceeded as exc:
            return _failed(state, str(exc))

        # Provision each source through the seam, under one merged byte budget
        # decremented before every fetch. Pre-model-call, so a crash-resume
        # that re-provisions is safe; every failure is terminal and sanitized.
        try:
            files = await provision_inputs(
                self._provisioners,
                request.inputs,
                RequestIdentity(
                    job_id=state.job_id,
                    scope_key=state.scope_key,
                    agent=state.agent,
                ),
                max_total_bytes=self.max_input_bytes,
            )
        except ProvisionError as exc:
            return _failed(state, f"analysis input provisioning failed: {exc}")

        attempt, created = await self._attempts.begin(state.attempt_id, state.job_id)
        if not created:
            return _failed(
                state,
                f"analysis attempt {attempt.attempt_id} is already {attempt.status}; "
                "automatic re-execution is refused pending reconciliation",
            )

        if self._workspace_root is not None:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(prefix="openloop-analysis-", dir=self._workspace_root)
        )
        run: AnalysisRun | None = None
        try:
            materialize_inputs(files, workspace / "inputs")
            (workspace / "outputs").mkdir()
            await step("materialize")

            try:
                run = await self.worker.run(workspace, state, on_step, record)
            except WorkerRunAborted as aborted:
                # The worker stopped itself at the in-run spend ceiling. The
                # spend accrued before stopping is real: settle it (which
                # normally raises the per-task cap error this abort predicted),
                # then keep the attempt failed closed — no read-out.
                run = _spend_only_run(
                    cost_usd=aborted.cost_usd,
                    prompt_tokens=aborted.prompt_tokens,
                    completion_tokens=aborted.completion_tokens,
                )
                failure = await settle_or_fail(
                    AnalysisCharge.from_run(run), run=run
                )
                if failure is not None:
                    return failure
                return _failed(
                    state, f"analysis run aborted: {aborted.reason}", run=run
                )
            except AnalysisWorkerFailure as exc:
                # A completion succeeded, so the provider's observed spend is
                # real even though local validation or sandbox setup failed.
                run = exc.run
                failure = await settle_or_fail(exc.charge, run=run)
                if failure is not None:
                    return failure
                return _failed(
                    state, f"analysis worker failed before execution: {exc}", run=run
                )
            except Exception as exc:  # model/backend failures before telemetry
                return _failed(state, f"analysis worker failed before execution: {exc}")

            # The one idempotent ledger settle for this attempt (the worker's
            # on_charge calls only *retained* cumulative telemetry). Runs with
            # the run's totals whether or not the worker reported mid-run, and
            # fails the attempt closed before any read-out.
            failure = await settle_or_fail(AnalysisCharge.from_run(run), run=run)
            if failure is not None:
                return failure

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
                attempt_id=state.attempt_id,
                run=run,
                artifact_ref=artifact_ref,
                prose_summary=_mechanical_summary(report, self.summary_lines),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    async def _mark_unknown(self, state: AnalysisState, error: Exception) -> None:
        """Best-effort durable signal for operator/provider reconciliation."""
        if state.attempt_id is None:
            return
        try:
            await self._attempts.mark_unknown(state.attempt_id, str(error))
        except Exception:
            # The original accounting-store failure is the actionable error. A
            # second failure cannot make it safe to continue execution.
            pass


class AnalysisWorkerConnector:
    """Maps ``analysis.report:write`` onto the sealed attempt boundary."""

    name = ANALYSIS_WORKER_TOOL_NAME
    # Product policy for Phase 1: an agent cannot silently opt this action out
    # of human approval. A future spend-only policy may make this per-agent.
    requires_approval = True
    # Declaring a workflow makes the gateway run this action durably when an
    # engine is wired (approval = wait node, resolve() emits the event; see the
    # workflow in openloop.workflows.analysis_worker). Without one, execute()
    # below is the engine-less fallback through the same orchestrator.
    workflow = "analysis_worker"

    def __init__(
        self,
        orchestrator: AnalysisAttemptRunner,
        *,
        uploads: "UploadStore | None" = None,
        available_sources: "set[str] | None" = None,
    ) -> None:
        self.orchestrator = orchestrator
        # Local metadata for the pre-approval resolution step. None means the
        # deployment has no upload surface at all.
        self.uploads = uploads
        self.available_sources = available_sources or {"staged"}

    def supported_permissions(self) -> set[str]:
        return {ANALYSIS_REPORT_WRITE}

    def describe(self, permission: str) -> ActionSpec:
        if permission != ANALYSIS_REPORT_WRITE:
            raise ValueError(f"unsupported analysis permission {permission!r}")
        # The schema is GENERATED from the args model the gateway parses with,
        # so what the model sees IS what parsing enforces — including the
        # discriminated inputs union the subset validator cannot express.
        return ActionSpec(
            description=(
                "Run sealed analysis over provisioned inputs — operator-staged "
                "references, files shared in this conversation thread, or a "
                "GitHub repository archive — and return a report artifact "
                "reference. Requires human approval."
            ),
            parameters=AnalysisReportArgs.model_json_schema(),
            model=AnalysisReportArgs,
            version=ANALYSIS_ARGS_VERSION,
        )

    def prepare_args(
        self,
        permission: str,
        args: dict,
        agent: "Agent | None" = None,
        *,
        warm_key: str | None = None,
    ) -> dict:
        """Mint identity on the parsed args — nothing here is caller-suppliable.

        Runs AFTER the gateway's typed parse, so ``args`` holds exactly the
        model-facing contract; building a fresh dict is the backstop against
        anything else riding into the durable record. ``job_id``/``attempt_id``
        are always freshly minted (an args-supplied job_id used to be a soft
        binding hole; staged access is a capability ref now, so job identity
        is purely run identity). ``warm_key`` is the requesting thread's
        ownership-tuple key (`thread_scope_key`), stamped as the request scope
        that upload provisioning is checked against — model-supplied scope
        would void upload scoping, and scopeless paths stamp None.
        """
        if permission != ANALYSIS_REPORT_WRITE:
            return args
        return {
            "instruction": str(args.get("instruction") or ""),
            "inputs": list(args.get("inputs") or []),
            "job_id": uuid.uuid4().hex,
            "attempt_id": uuid.uuid4().hex,
            "agent": agent.metadata.name if agent is not None else None,
            "scope_key": warm_key,
        }

    async def resolve_args(
        self, permission: str, args: dict
    ) -> tuple[dict, str | None]:
        """Pre-approval, LOCAL-ONLY resolution of upload references.

        A DB read, never a surface fetch — approve-before-work holds. Verifies
        each upload against the trusted thread scope stamped in
        ``prepare_args`` and stamps display metadata (``upload_meta``) so the
        approval card can truthfully name the file; the model only ever knew
        the opaque ref. A violation refuses at invoke time — a human is never
        asked to approve a request policy forbids. The provisioner re-checks
        scope post-approval (TOCTOU).
        """
        if permission != ANALYSIS_REPORT_WRITE:
            return args, None
        inputs = args.get("inputs") or []
        for entry in inputs:
            source = entry.get("source") if isinstance(entry, dict) else None
            if source not in self.available_sources:
                return args, (
                    f"input source {source!r} is not available in this "
                    "deployment"
                )
        upload_meta: dict[str, dict] = {}
        scope = args.get("scope_key")
        for entry in inputs:
            if entry.get("source") != "upload":
                continue
            if self.uploads is None:
                return args, "upload inputs are not available in this deployment"
            if not scope:
                # /tools/invoke and other scopeless paths stamp no surface
                # scope; the direct API provisions via `staged`, never uploads.
                return args, (
                    "upload inputs are only usable from the conversation "
                    "thread the file was shared in"
                )
            ref = entry.get("upload_ref") or ""
            record = await self.uploads.get(ref)
            # Unknown ref and wrong thread share one message: whether a file
            # id exists in another thread must not leak here.
            if record is None or record.scope_key != scope:
                return args, (
                    f"no shared file {ref!r} is available in this "
                    "conversation thread"
                )
            upload_meta[ref] = {"name": record.name, "size": record.size}
        if upload_meta:
            args = {**args, "upload_meta": upload_meta}
        return args, None

    async def execute(self, permission: str, args: dict) -> ToolResult:
        if permission != ANALYSIS_REPORT_WRITE:
            return ToolResult(ok=False, summary=f"unsupported permission {permission}")
        # Gateway.invoke() has already called prepare_args before persisting an
        # approval request. Do not call it again here: doing so without the
        # trusted Agent object would erase the stamped invoking identity and
        # misattribute spend after approval.
        state = _state_from_record(args)
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


def _state_from_record(record: dict) -> AnalysisState:
    """Identity-lenient state from a persisted record (approval args or
    workflow state).

    Never raises on garbage — attempt reconciliation must be reachable for any
    record shape; executability is the orchestrator's re-parse."""
    inputs = record.get("inputs")
    scope_key = record.get("scope_key")
    args_schema = record.get("args_schema")
    return AnalysisState(
        job_id=str(record.get("job_id") or uuid.uuid4().hex),
        instruction=str(record.get("instruction") or ""),
        inputs=list(inputs) if isinstance(inputs, list) else [],
        agent=record.get("agent") if isinstance(record.get("agent"), str) else None,
        scope_key=scope_key if isinstance(scope_key, str) else None,
        args_schema=args_schema if isinstance(args_schema, int) else None,
        attempt_id=str(record.get("attempt_id") or uuid.uuid4().hex),
    )


def _failed(
    state: AnalysisState, error: str, *, run: AnalysisRun | None = None
) -> AnalysisResult:
    return AnalysisResult(
        job_id=state.job_id,
        attempt_id=state.attempt_id,
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


def _discard_report(report: Path) -> None:
    """Remove whatever model-authored code left at the report path.

    ``unlink`` covers regular files, symlinks, and FIFOs, but raises on a real
    directory — and a malformed round is one ``os.makedirs`` away from leaving
    one. A symlink is always unlinked itself, never followed (``rmtree``
    refuses symlinks-to-directories, so the order here matters).
    """
    if report.is_dir() and not report.is_symlink():
        shutil.rmtree(report)
    else:
        report.unlink(missing_ok=True)


def _strip_code_fence(script: str) -> str:
    """Tolerate a fenced model response without trying to parse Python."""
    stripped = script.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1])
    return script
