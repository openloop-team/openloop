"""Phase 1 sealed-analysis worker and orchestration tests (no Docker)."""

from pathlib import Path

import pytest

from openloop.analysis import (
    InMemoryAnalysisAttemptStore,
    InMemoryArtifactStore,
    InMemoryInputStore,
    InputFile,
    InputManifest,
)
from openloop.agents import load_agent
from openloop.agents.schema import Tool
from openloop.models.gateway import ModelResponse
from openloop.sandbox import SandboxLimits, SandboxResult
from openloop.testing import FakeGateway
from openloop.tools.analysis_worker import (
    ANALYSIS_REPORT_WRITE,
    AnalysisResult,
    AnalysisRun,
    AnalysisState,
    AnalysisWorkerConnector,
    BuiltinAnalysisWorker,
    SealedAnalysisOrchestrator,
)
from openloop.tools import ToolGateway
from openloop.usage import InMemoryUsageStore, UsageRecord, WorkerSpendLedger

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _state(job_id="job-1"):
    return AnalysisState(
        job_id=job_id,
        input_ref="upload:one",
        instruction="summarize the sales data",
    )


def _run(*, exit_code=0, cost=0.25, stdout="", stderr=""):
    return AnalysisRun(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        killed=False,
        timed_out=False,
        stdout_truncated=False,
        stderr_truncated=False,
        duration_seconds=0.1,
        cost_usd=cost,
        prompt_tokens=100,
        completion_tokens=25,
    )


class _FakeSealedSandbox:
    def __init__(self, result=None):
        self.result = result or SandboxResult(
            exit_code=0,
            stdout="controller telemetry",
            stderr="",
            killed=False,
            timed_out=False,
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.1,
        )
        self.specs = []

    async def run(self, spec):
        self.specs.append(spec)
        return self.result


class _PaidGateway:
    """A successful completion carrying spend for paid-failure tests."""

    def __init__(self, text, *, cost=0.25, prompt_tokens=100, completion_tokens=25):
        self.response = ModelResponse(
            text=text,
            model="analysis-model",
            cost_usd=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def complete(self, model, messages, **kwargs):
        return self.response


class _FailingSealedSandbox:
    async def run(self, spec):
        raise RuntimeError("docker daemon disconnected")


class _ReportWorker:
    def __init__(self, run=None, body=b"# Findings\nRevenue increased.\n"):
        self.result = run or _run()
        self.body = body
        self.workspaces = []

    async def run(self, workspace, state, on_step=None, on_charge=None):
        self.workspaces.append(workspace)
        (workspace / "outputs" / "report.md").write_bytes(self.body)
        return self.result


class _TrackingInputStore(InMemoryInputStore):
    def __init__(self):
        super().__init__()
        self.gets = 0

    async def get(self, job_id, input_ref):
        self.gets += 1
        return await super().get(job_id, input_ref)


def _ledger(*, per_task=0.50, monthly=None):
    agent = load_agent(AGENT_YAML)
    agent.spec.budget.per_task_usd = per_task
    agent.spec.budget.monthly_usd = monthly
    usage = InMemoryUsageStore()
    return WorkerSpendLedger(
        usage=usage,
        model="analysis-model",
        agents={agent.metadata.name: agent},
        default_agent=agent.metadata.name,
        task_kind="analysis_worker",
    ), usage, agent


async def test_builtin_worker_generates_in_controller_and_streams_script_to_sealed_stdin(
    tmp_path,
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    # Content must not enter the model prompt; the worker gives it only a
    # filename/size inventory and lets the sealed script inspect it at runtime.
    (inputs / "sales.csv").write_text("top-secret,total\nwest,42\n")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    sandbox = _FakeSealedSandbox()
    gateway = FakeGateway(
        "open('/workspace/outputs/report.md', 'w').write('# report')"
    )
    worker = BuiltinAnalysisWorker(
        "m",
        sandbox,
        gateway=gateway,
        limits=SandboxLimits(timeout_seconds=30),
        output_cap_bytes=1024,
    )

    run = await worker.run(tmp_path, _state())

    (spec,) = sandbox.specs
    assert spec.command == ("python", "-")
    assert spec.stdin.startswith("open('/workspace/outputs/report.md'")
    assert spec.mounts[0].read_only is True
    assert spec.mounts[0].target == "/workspace/inputs"
    assert spec.mounts[1].target == "/workspace/outputs"
    prompt = "\n".join(m["content"] for m in gateway.last_messages)
    assert "sales.csv" in prompt
    assert "top-secret" not in prompt
    assert run.stdout == "controller telemetry"


async def test_orchestrator_settles_before_readout_and_persists_only_ref():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        job_id=state.job_id,
        input_ref=state.input_ref,
        files=(InputFile("sales.csv", b"amount\n42\n"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger()
    worker = _ReportWorker()
    orchestrator = SealedAnalysisOrchestrator(worker, inputs, artifacts, ledger=ledger)
    state.agent = agent.metadata.name

    result = await orchestrator.run_analysis(state)

    assert result.ok
    assert result.artifact_ref == "analysis://job-1/report.md"
    assert result.prose_summary == "# Findings\nRevenue increased."
    assert (await artifacts.get(result.artifact_ref)).body.startswith(b"# Findings")
    # The public result carries telemetry only — never stdout/stderr bytes.
    assert "stdout" not in result.data()
    assert "stderr" not in result.data()
    assert usage.records[0].task_kind == "analysis_worker"
    assert usage.records[0].outcome == "ok"
    assert not worker.workspaces[0].exists()


async def test_monthly_gate_runs_before_input_provisioning():
    inputs = _TrackingInputStore()
    ledger, usage, agent = _ledger(monthly=1.0)
    await usage.record(UsageRecord(
        scope_key="ws:acme:agent:dev-platform",
        workspace="acme",
        agent="dev-platform",
        model="m",
        cost_usd=1.0,
    ))
    state = _state()
    state.agent = agent.metadata.name
    result = await SealedAnalysisOrchestrator(
        _ReportWorker(), inputs, InMemoryArtifactStore(), ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "monthly budget" in result.error
    assert inputs.gets == 0


async def test_empty_instruction_fails_before_any_gate_store_or_spend():
    # Fail-closed backstop behind the gateway's invoke-time schema validation:
    # a stale persisted record (approval/workflow written before the seam) or a
    # future direct caller must not buy a model call with no instruction.
    inputs = _TrackingInputStore()
    ledger, usage, agent = _ledger()
    attempts = InMemoryAnalysisAttemptStore()
    state = _state()
    state.instruction = "   "
    state.agent = agent.metadata.name

    result = await SealedAnalysisOrchestrator(
        _ReportWorker(), inputs, InMemoryArtifactStore(),
        attempts=attempts, ledger=ledger,
    ).run_analysis(state)

    assert not result.ok
    assert "instruction is required" in result.error
    assert inputs.gets == 0  # no provisioning
    assert usage.records == []  # no spend recorded or settled
    assert await attempts.get(state.attempt_id) is None  # no attempt begun


async def test_empty_input_ref_fails_before_input_lookup():
    inputs = _TrackingInputStore()
    ledger, usage, agent = _ledger()
    state = _state()
    state.input_ref = ""
    state.agent = agent.metadata.name

    result = await SealedAnalysisOrchestrator(
        _ReportWorker(), inputs, InMemoryArtifactStore(), ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "input_ref is required" in result.error
    assert inputs.gets == 0
    assert usage.records == []


async def test_over_cap_settlement_blocks_report_readout_and_artifact_write():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        job_id=state.job_id, input_ref=state.input_ref,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger(per_task=0.10)
    state.agent = agent.metadata.name
    result = await SealedAnalysisOrchestrator(
        _ReportWorker(run=_run(cost=0.25)), inputs, artifacts, ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "per-task budget" in result.error
    assert result.artifact_ref is None
    assert await artifacts.get("analysis://job-1/report.md") is None
    assert usage.records[0].outcome == "over_task_budget"


async def test_nonzero_execution_settles_spend_without_reading_report():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        job_id=state.job_id, input_ref=state.input_ref,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    result = await SealedAnalysisOrchestrator(
        _ReportWorker(run=_run(exit_code=7, stderr="sensitive input")),
        inputs,
        artifacts,
        ledger=ledger,
    ).run_analysis(state)

    assert not result.ok
    assert result.run.stderr == "sensitive input"
    assert "sensitive input" not in result.error
    assert await artifacts.get("analysis://job-1/report.md") is None
    assert usage.records[0].outcome == "ok"


async def test_empty_generated_program_still_settles_known_completion_spend():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        job_id=state.job_id, input_ref=state.input_ref,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    attempts = InMemoryAnalysisAttemptStore()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    sandbox = _FakeSealedSandbox()
    worker = BuiltinAnalysisWorker(
        "analysis-model",
        sandbox,
        gateway=_PaidGateway("", cost=0.31, prompt_tokens=120, completion_tokens=30),
        limits=SandboxLimits(timeout_seconds=30),
        output_cap_bytes=1_024,
    )

    result = await SealedAnalysisOrchestrator(
        worker, inputs, artifacts, attempts=attempts, ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "empty program" in result.error
    assert result.run.exit_code is None
    assert result.run.cost_usd == 0.31
    assert sandbox.specs == []  # no sealed execution and no report read-out
    assert await artifacts.get("analysis://job-1/report.md") is None
    (record,) = usage.records
    assert (record.cost_usd, record.prompt_tokens, record.completion_tokens) == (0.31, 120, 30)
    attempt = await attempts.get(result.attempt_id)
    assert attempt.status == "settled"
    assert attempt.cost_usd == 0.31


async def test_sandbox_setup_failure_still_settles_known_completion_spend():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        job_id=state.job_id, input_ref=state.input_ref,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    worker = BuiltinAnalysisWorker(
        "analysis-model",
        _FailingSealedSandbox(),
        gateway=_PaidGateway("print('analysis')", cost=0.19),
        limits=SandboxLimits(timeout_seconds=30),
        output_cap_bytes=1_024,
    )

    result = await SealedAnalysisOrchestrator(
        worker, inputs, artifacts, ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "execution setup failed" in result.error
    assert result.run.exit_code is None
    assert result.run.cost_usd == 0.19
    assert await artifacts.get("analysis://job-1/report.md") is None
    assert usage.records[0].cost_usd == 0.19


async def test_reused_attempt_id_refuses_another_model_execution():
    inputs = InMemoryInputStore()
    state = _state()
    state.attempt_id = "attempt-1"
    await inputs.stage(InputManifest(
        job_id=state.job_id,
        input_ref=state.input_ref,
        files=(InputFile("sales.csv", b"x"),),
    ))
    worker = _ReportWorker()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    orchestrator = SealedAnalysisOrchestrator(
        worker,
        inputs,
        InMemoryArtifactStore(),
        attempts=InMemoryAnalysisAttemptStore(),
        ledger=ledger,
    )

    first = await orchestrator.run_analysis(state)
    # The approved attempt identity, not job identity, controls replay. Reuse it
    # explicitly here to simulate a retry after a controller crash.
    retry = await orchestrator.run_analysis(
        AnalysisState(
            job_id=state.job_id,
            input_ref=state.input_ref,
            instruction=state.instruction,
            agent=agent.metadata.name,
            attempt_id=state.attempt_id,
        )
    )

    assert first.ok
    assert not retry.ok
    assert "automatic re-execution is refused" in retry.error
    assert len(worker.workspaces) == 1
    assert len(usage.records) == 1


async def test_charged_attempt_reconciles_usage_without_reexecuting_computation():
    state = _state()
    state.attempt_id = "attempt-needs-settlement"
    attempts = InMemoryAnalysisAttemptStore()
    await attempts.begin(state.attempt_id, state.job_id)
    await attempts.charge(
        state.attempt_id,
        cost_usd=0.25,
        prompt_tokens=100,
        completion_tokens=25,
    )
    # A spent monthly budget would reject a fresh attempt. Reconciliation still
    # has to account for this already-observed charge before it returns.
    ledger, usage, agent = _ledger(monthly=0.0)
    state.agent = agent.metadata.name
    worker = _ReportWorker()

    result = await SealedAnalysisOrchestrator(
        worker,
        InMemoryInputStore(),
        InMemoryArtifactStore(),
        attempts=attempts,
        ledger=ledger,
    ).run_analysis(state)

    assert not result.ok
    assert "spend was settled without re-executing" in result.error
    assert (await attempts.get(state.attempt_id)).status == "settled"
    assert len(usage.records) == 1
    assert usage.records[0].idempotency_key == state.attempt_id
    assert worker.workspaces == []


async def test_accounting_failure_marks_observed_charge_unknown_and_skips_readout():
    class _BrokenUsageStore(InMemoryUsageStore):
        async def record(self, usage):
            raise RuntimeError("usage store unavailable")

    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        job_id=state.job_id,
        input_ref=state.input_ref,
        files=(InputFile("sales.csv", b"x"),),
    ))
    agent = load_agent(AGENT_YAML)
    agent.spec.budget.per_task_usd = 0.50
    ledger = WorkerSpendLedger(
        usage=_BrokenUsageStore(),
        model="analysis-model",
        agents={agent.metadata.name: agent},
        default_agent=agent.metadata.name,
        task_kind="analysis_worker",
    )
    state.agent = agent.metadata.name
    attempts = InMemoryAnalysisAttemptStore()
    artifacts = InMemoryArtifactStore()

    result = await SealedAnalysisOrchestrator(
        _ReportWorker(),
        inputs,
        artifacts,
        attempts=attempts,
        ledger=ledger,
    ).run_analysis(state)

    assert not result.ok
    assert "could not be durably accounted" in result.error
    attempt = await attempts.get(result.attempt_id)
    assert attempt.status == "unknown"
    assert attempt.cost_usd == 0.25
    assert await artifacts.get("analysis://job-1/report.md") is None


def test_connector_prepare_args_strips_inline_payload_and_stamps_agent():
    connector = AnalysisWorkerConnector(object())
    agent = load_agent(AGENT_YAML)
    args = connector.prepare_args(
        ANALYSIS_REPORT_WRITE,
        {
            "instruction": "summarize",
            "input_ref": "upload:123",
            "raw_data": "must never persist",
            "agent": "spoofed",
        },
        agent,
    )

    assert args["agent"] == "dev-platform"
    assert args["input_ref"] == "upload:123"
    assert "raw_data" not in args
    assert args["job_id"]


async def test_analysis_is_always_held_for_approval_and_keeps_stamped_agent():
    class _Runner:
        def __init__(self):
            self.state = None

        async def run_analysis(self, state, on_step=None):
            self.state = state
            return AnalysisResult(
                job_id=state.job_id,
                input_ref=state.input_ref,
                run=_run(),
                artifact_ref=f"analysis://{state.job_id}/report.md",
                prose_summary="# Done",
            )

    runner = _Runner()
    connector = AnalysisWorkerConnector(runner)
    agent = load_agent(AGENT_YAML)
    agent.spec.tools.append(Tool(name="analysis", type="native", permissions=["report:write"]))
    # Deliberately do not add the action to require_for: the connector's Phase 1
    # policy still has to hold it for human approval.
    gateway = ToolGateway(tools=[connector])

    pending = await gateway.invoke(
        agent,
        "analysis.report:write",
        {"instruction": "summarize", "input_ref": "upload:1", "inline": "secret"},
    )

    assert pending.status == "pending_approval"
    assert pending.approval.args["agent"] == "dev-platform"
    assert "inline" not in pending.approval.args
    assert "configured spend limits" in pending.approval.summary

    completed = await gateway.resolve(
        pending.approval.id, "@maciag.artur", approve=True
    )
    assert completed.result.ok
    assert runner.state.agent == "dev-platform"
