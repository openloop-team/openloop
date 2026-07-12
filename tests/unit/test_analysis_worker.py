"""Sealed-analysis worker and orchestration tests (no Docker).

Covers the Phase 1 single-shot strategy plus the Phase 3 iterative strategy
(exec_feedback loop, cumulative charge retention, in-run spend abort).
"""

from pathlib import Path

import pytest

from openloop.analysis import (
    ANALYSIS_ARGS_VERSION,
    InMemoryAnalysisAttemptStore,
    InMemoryArtifactStore,
    InMemoryInputStore,
    InputFile,
    InputManifest,
    StagedProvisioner,
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
from openloop.tools.coding_worker import WorkerRunAborted
from openloop.usage import InMemoryUsageStore, UsageRecord, WorkerSpendLedger

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"
_INPUT_REF = "staged:one"


def _state(job_id="job-1"):
    return AnalysisState(
        job_id=job_id,
        instruction="summarize the sales data",
        inputs=[{"source": "staged", "input_ref": _INPUT_REF}],
        args_schema=ANALYSIS_ARGS_VERSION,
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


class _NoReportWorker:
    """Exits 0 but writes no report — the relative-path / stdout-only miss."""

    def __init__(self, run=None):
        self.result = run or _run()
        self.workspaces = []

    async def run(self, workspace, state, on_step=None, on_charge=None):
        self.workspaces.append(workspace)
        # Deliberately writes nothing to outputs/report.md.
        return self.result


class _TrackingInputStore(InMemoryInputStore):
    def __init__(self):
        super().__init__()
        self.gets = 0

    async def get(self, input_ref):
        self.gets += 1
        return await super().get(input_ref)


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
        strategy="single",
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
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"amount\n42\n"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger()
    worker = _ReportWorker()
    orchestrator = SealedAnalysisOrchestrator(worker, [StagedProvisioner(inputs)], artifacts, ledger=ledger)
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
        _ReportWorker(), [StagedProvisioner(inputs)], InMemoryArtifactStore(), ledger=ledger
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
        _ReportWorker(), [StagedProvisioner(inputs)], InMemoryArtifactStore(),
        attempts=attempts, ledger=ledger,
    ).run_analysis(state)

    assert not result.ok
    assert "not executable" in result.error
    assert "instruction" in result.error
    assert inputs.gets == 0  # no provisioning
    assert usage.records == []  # no spend recorded or settled
    assert await attempts.get(state.attempt_id) is None  # no attempt begun


async def test_empty_inputs_fails_before_input_lookup():
    inputs = _TrackingInputStore()
    ledger, usage, agent = _ledger()
    state = _state()
    state.inputs = []  # the executable re-parse rejects a request with no inputs
    state.agent = agent.metadata.name

    result = await SealedAnalysisOrchestrator(
        _ReportWorker(), [StagedProvisioner(inputs)], InMemoryArtifactStore(), ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "not executable" in result.error
    assert inputs.gets == 0
    assert usage.records == []


async def test_pre_version_record_refuses_before_gate_store_or_spend():
    # A record written before args versioning (the scalar-input_ref era) has a
    # NULL args_schema; it must refuse cleanly instead of running over args the
    # current contract would read differently.
    inputs = _TrackingInputStore()
    ledger, usage, agent = _ledger()
    attempts = InMemoryAnalysisAttemptStore()
    state = _state()
    state.args_schema = None
    state.agent = agent.metadata.name

    result = await SealedAnalysisOrchestrator(
        _ReportWorker(), [StagedProvisioner(inputs)], InMemoryArtifactStore(),
        attempts=attempts, ledger=ledger,
    ).run_analysis(state)

    assert not result.ok
    assert "predates args schema" in result.error
    assert inputs.gets == 0
    assert usage.records == []
    assert await attempts.get(state.attempt_id) is None


async def test_unknown_staged_ref_fails_provisioning_after_gate():
    inputs = _TrackingInputStore()  # nothing staged
    ledger, usage, agent = _ledger()
    state = _state()
    state.agent = agent.metadata.name

    result = await SealedAnalysisOrchestrator(
        _ReportWorker(), [StagedProvisioner(inputs)], InMemoryArtifactStore(), ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "provisioning failed" in result.error
    assert inputs.gets == 1  # the lookup ran (and missed)
    assert usage.records == []


async def test_over_cap_settlement_blocks_report_readout_and_artifact_write():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger(per_task=0.10)
    state.agent = agent.metadata.name
    result = await SealedAnalysisOrchestrator(
        _ReportWorker(run=_run(cost=0.25)), [StagedProvisioner(inputs)], artifacts, ledger=ledger
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
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    result = await SealedAnalysisOrchestrator(
        _ReportWorker(run=_run(exit_code=7, stderr="sensitive input")),
        [StagedProvisioner(inputs)],
        artifacts,
        ledger=ledger,
    ).run_analysis(state)

    assert not result.ok
    assert result.run.stderr == "sensitive input"
    assert "sensitive input" not in result.error
    assert await artifacts.get("analysis://job-1/report.md") is None
    assert usage.records[0].outcome == "ok"


async def test_clean_exit_without_report_is_a_job_outcome_not_a_readout_refusal():
    # A program that exits 0 but leaves no report.md is a benign outcome, not a
    # containment refusal: read_contained raises FileNotFoundError (never a
    # ReadOutViolation), so the message must say the run produced no report and
    # must NOT read like a security block. Spend still settles before read-out.
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    result = await SealedAnalysisOrchestrator(
        _NoReportWorker(run=_run(exit_code=0)),
        [StagedProvisioner(inputs)],
        artifacts,
        ledger=ledger,
    ).run_analysis(state)

    assert not result.ok
    assert "produced no report" in result.error
    assert "/workspace/outputs/report.md" in result.error
    assert "refused" not in result.error  # not a containment refusal
    assert await artifacts.get("analysis://job-1/report.md") is None
    assert usage.records[0].outcome == "ok"  # spend settled before read-out


async def test_empty_generated_program_still_settles_known_completion_spend():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
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
        strategy="single",
    )

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], artifacts, attempts=attempts, ledger=ledger
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
        input_ref=_INPUT_REF,
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
        strategy="single",
    )

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], artifacts, ledger=ledger
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
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    worker = _ReportWorker()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    orchestrator = SealedAnalysisOrchestrator(
        worker,
        [StagedProvisioner(inputs)],
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
            instruction=state.instruction,
            inputs=list(state.inputs),
            args_schema=ANALYSIS_ARGS_VERSION,
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
        [StagedProvisioner(InMemoryInputStore())],
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
        input_ref=_INPUT_REF,
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
        [StagedProvisioner(inputs)],
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


def test_connector_prepare_args_mints_identity_and_stamps_agent_and_scope():
    # prepare_args runs AFTER the gateway's typed parse, so it sees only the
    # model-facing contract; it mints fresh identity (a caller-supplied job_id
    # is never honored — a soft binding hole) and stamps the trusted invoking
    # agent + the request scope from the warm_key.
    connector = AnalysisWorkerConnector(object())
    agent = load_agent(AGENT_YAML)
    args = connector.prepare_args(
        ANALYSIS_REPORT_WRITE,
        {
            "instruction": "summarize",
            "inputs": [{"source": "staged", "input_ref": "staged:123"}],
            "job_id": "attacker-chosen",
            "agent": "spoofed",
        },
        agent,
        warm_key="slack\x1facme\x1fdev-platform\x1fC1\x1fT1",
    )

    assert args["agent"] == "dev-platform"
    assert args["inputs"] == [{"source": "staged", "input_ref": "staged:123"}]
    assert args["job_id"] and args["job_id"] != "attacker-chosen"
    assert args["attempt_id"]
    assert args["scope_key"] == "slack\x1facme\x1fdev-platform\x1fC1\x1fT1"


async def test_analysis_is_always_held_for_approval_and_keeps_stamped_agent():
    class _Runner:
        def __init__(self):
            self.state = None

        async def run_analysis(self, state, on_step=None):
            self.state = state
            return AnalysisResult(
                job_id=state.job_id,
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
        # `inline` is rejected by the typed parse (extra="forbid"); the model
        # can never smuggle raw data into the durable record.
        {
            "instruction": "summarize",
            "inputs": [{"source": "staged", "input_ref": "staged:1"}],
        },
    )

    assert pending.status == "pending_approval"
    assert pending.approval.args["agent"] == "dev-platform"
    assert pending.approval.args_schema == ANALYSIS_ARGS_VERSION
    assert "inline" not in pending.approval.args
    assert "configured spend limits" in pending.approval.summary

    completed = await gateway.resolve(
        pending.approval.id, "@maciag.artur", approve=True
    )
    assert completed.result.ok
    assert runner.state.agent == "dev-platform"


# --- Phase 3: iterative strategy (exec_feedback under the in-run cap) --------


def _response(text, *, cost=0.20, prompt_tokens=100, completion_tokens=25):
    return ModelResponse(
        text=text,
        model="analysis-model",
        cost_usd=cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _sandbox_result(
    *,
    exit_code=0,
    stdout="",
    stderr="",
    stdout_truncated=False,
    stderr_truncated=False,
    timed_out=False,
    killed=False,
):
    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        killed=killed,
        timed_out=timed_out,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        duration_seconds=0.1,
    )


class _SequencedGateway:
    """Bills a scripted sequence of completions (or raises) per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def complete(self, model, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _ScriptedSealedSandbox:
    """Plays scripted rounds; a round may write the report like a real run.

    Each round is ``(SandboxResult, body)`` or an exception to raise; ``body``
    is report bytes, a callable given the outputs dir (for planting hostile
    filesystem shapes), or None. Records whether report.md pre-existed at each
    round so tests can pin the stale-report discard.
    """

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.specs = []
        self.report_existed_at_start: list[bool] = []

    async def run(self, spec):
        self.specs.append(spec)
        outputs = spec.mounts[1].source
        self.report_existed_at_start.append((outputs / "report.md").exists())
        round_ = self._rounds.pop(0)
        if isinstance(round_, Exception):
            raise round_
        result, body = round_
        if callable(body):
            body(outputs)
        elif body is not None:
            (outputs / "report.md").write_bytes(body)
        return result


def _iterative_worker(sandbox, gateway, *, max_iterations=3, feedback_chars=500):
    return BuiltinAnalysisWorker(
        "analysis-model",
        sandbox,
        gateway=gateway,
        limits=SandboxLimits(timeout_seconds=30),
        output_cap_bytes=4_096,
        strategy="iterative",
        max_iterations=max_iterations,
        exec_feedback_max_chars=feedback_chars,
    )


def _workspace(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "sales.csv").write_text("a,b\n1,2\n")
    (tmp_path / "outputs").mkdir()
    return tmp_path


async def test_iterative_worker_feeds_exec_feedback_and_stops_on_clean_report(tmp_path):
    workspace = _workspace(tmp_path)
    gateway = _SequencedGateway([
        _response("print(open('/workspace/inputs/sales.csv').read())", cost=0.10),
        _response("open('/workspace/outputs/report.md','w').write('# done')", cost=0.15),
    ])
    sandbox = _ScriptedSealedSandbox([
        (_sandbox_result(exit_code=1, stdout="columns: a,b", stderr="KeyError: 'c'"), None),
        (_sandbox_result(exit_code=0), b"# Findings\n"),
    ])
    worker = _iterative_worker(sandbox, gateway)

    run = await worker.run(workspace, _state())

    assert run.exit_code == 0
    assert run.iterations == 2
    assert run.cost_usd == pytest.approx(0.25)
    assert run.prompt_tokens == 200
    assert len(sandbox.specs) == 2
    # Round 2's prompt carries round 1's program plus its execution feedback.
    second_call = gateway.calls[1]
    assert second_call[-2]["role"] == "assistant"
    assert "sales.csv" in second_call[-2]["content"]
    feedback = second_call[-1]["content"]
    assert second_call[-1]["role"] == "user"
    assert "Round 1 of 3" in feedback
    assert "exited with code 1" in feedback
    assert "columns: a,b" in feedback
    assert "KeyError: 'c'" in feedback
    assert "no report was captured at /workspace/outputs/report.md" in feedback


async def test_final_round_feedback_demands_the_report_be_written(tmp_path):
    # The observed failure: the model spends its last round inspecting/printing
    # and never writes the report. The feedback for the round before the last
    # must make the write obligation unmissable.
    workspace = _workspace(tmp_path)
    gateway = _SequencedGateway([
        _response("print(open('/workspace/inputs/sales.csv').read())"),
        _response("open('/workspace/outputs/report.md','w').write('# done')"),
    ])
    sandbox = _ScriptedSealedSandbox([
        (_sandbox_result(exit_code=0), None),  # round 1: inspected, no report
        (_sandbox_result(exit_code=0), b"# done\n"),  # round 2 (final): writes it
    ])
    worker = _iterative_worker(sandbox, gateway, max_iterations=2)

    await worker.run(workspace, _state())

    final_round_prompt = gateway.calls[1][-1]["content"]
    assert "FINAL round" in final_round_prompt
    assert "you MUST write" in final_round_prompt
    assert "/workspace/outputs/report.md" in final_round_prompt


async def test_exec_feedback_is_hard_truncated_and_says_so(tmp_path):
    workspace = _workspace(tmp_path)
    gateway = _SequencedGateway([
        _response("print('x' * 999)"),
        _response("open('/workspace/outputs/report.md','w').write('# r')"),
    ])
    sandbox = _ScriptedSealedSandbox([
        # stderr is short but was already cut at capture time (stream cap):
        # the model must be told either way.
        (
            _sandbox_result(
                exit_code=1, stdout="x" * 50, stderr="tail lost",
                stderr_truncated=True,
            ),
            None,
        ),
        (_sandbox_result(exit_code=0), b"# r\n"),
    ])
    worker = _iterative_worker(sandbox, gateway, feedback_chars=10)

    await worker.run(workspace, _state())

    feedback = gateway.calls[1][-1]["content"]
    assert "x" * 10 in feedback
    assert "x" * 11 not in feedback
    stdout_part, stderr_part = feedback.split("--- stderr")
    assert "output was cut" in stdout_part
    assert "output was cut" in stderr_part


async def test_iterative_worker_discards_a_failed_rounds_report(tmp_path):
    workspace = _workspace(tmp_path)
    gateway = _SequencedGateway([
        _response("round one"), _response("round two"), _response("round three"),
    ])
    sandbox = _ScriptedSealedSandbox([
        # Writes a report but fails: the loop must not accept it, and the next
        # round must start without it.
        (_sandbox_result(exit_code=1), b"# partial\n"),
        # Exits clean but writes nothing: a stale report must not end the loop.
        (_sandbox_result(exit_code=0), None),
        (_sandbox_result(exit_code=0), b"# final\n"),
    ])
    worker = _iterative_worker(sandbox, gateway)

    run = await worker.run(workspace, _state())

    assert run.iterations == 3
    assert sandbox.report_existed_at_start == [False, False, False]
    assert (workspace / "outputs" / "report.md").read_bytes() == b"# final\n"
    assert "discarded" in gateway.calls[1][-1]["content"]
    assert "no report was captured" in gateway.calls[2][-1]["content"]


async def test_iterative_worker_returns_the_last_run_when_rounds_are_exhausted(tmp_path):
    workspace = _workspace(tmp_path)
    gateway = _SequencedGateway([_response("one"), _response("two")])
    sandbox = _ScriptedSealedSandbox([
        (_sandbox_result(exit_code=1, stderr="boom"), None),
        (_sandbox_result(exit_code=3, stderr="still boom"), None),
    ])
    worker = _iterative_worker(sandbox, gateway, max_iterations=2)

    run = await worker.run(workspace, _state())

    assert run.exit_code == 3
    assert run.iterations == 2
    assert run.cost_usd == pytest.approx(0.40)
    assert len(gateway.calls) == 2
    assert len(sandbox.specs) == 2


async def test_iterative_worker_reports_cumulative_charges_and_stops_at_the_cap(tmp_path):
    workspace = _workspace(tmp_path)
    gateway = _SequencedGateway([_response("one"), _response("two")])
    sandbox = _ScriptedSealedSandbox([(_sandbox_result(exit_code=1), None)])
    worker = _iterative_worker(sandbox, gateway)
    state = _state()
    state.budget_usd = 0.30

    charges = []

    async def on_charge(charge):
        charges.append((charge.cost_usd, charge.prompt_tokens))

    with pytest.raises(WorkerRunAborted) as aborted:
        await worker.run(workspace, state, on_charge=on_charge)

    # Both completions were durably reported cumulatively; the second crossed
    # the cap, so its program never reached the sandbox.
    assert charges == [(pytest.approx(0.20), 100), (pytest.approx(0.40), 200)]
    assert len(sandbox.specs) == 1
    assert aborted.value.cost_usd == pytest.approx(0.40)
    assert aborted.value.prompt_tokens == 200
    assert "per-task cap" in aborted.value.reason


async def test_iterative_abort_settles_cumulative_spend_and_blocks_readout():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    attempts = InMemoryAnalysisAttemptStore()
    ledger, usage, agent = _ledger(per_task=0.30)
    state.agent = agent.metadata.name
    gateway = _SequencedGateway([_response("one"), _response("two")])
    sandbox = _ScriptedSealedSandbox([(_sandbox_result(exit_code=1), None)])
    worker = _iterative_worker(sandbox, gateway)

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], artifacts, attempts=attempts, ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "per-task budget" in result.error
    assert result.run.cost_usd == pytest.approx(0.40)
    assert len(sandbox.specs) == 1
    assert await artifacts.get("analysis://job-1/report.md") is None
    (record,) = usage.records
    assert record.cost_usd == pytest.approx(0.40)
    assert record.outcome == "over_task_budget"
    attempt = await attempts.get(result.attempt_id)
    assert attempt.status == "settled"
    assert attempt.cost_usd == pytest.approx(0.40)


async def test_later_round_model_failure_settles_the_earlier_rounds_spend():
    # Round 1 was billed and durably retained; round 2's provider call dies.
    # The failure must carry the cumulative total out so it settles — the
    # attempt must never be left charged-but-unsettled (a failed workflow
    # instance is terminal, so nothing would ever reconcile it).
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    attempts = InMemoryAnalysisAttemptStore()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    gateway = _SequencedGateway([_response("one"), RuntimeError("provider 500")])
    sandbox = _ScriptedSealedSandbox([(_sandbox_result(exit_code=1), None)])
    worker = _iterative_worker(sandbox, gateway)

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], InMemoryArtifactStore(), attempts=attempts, ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "model call failed after 1 completed round" in result.error
    (record,) = usage.records
    assert record.cost_usd == pytest.approx(0.20)
    attempt = await attempts.get(result.attempt_id)
    assert attempt.status == "settled"
    assert attempt.cost_usd == pytest.approx(0.20)


async def test_first_round_model_failure_stays_pre_telemetry():
    # Nothing was billed yet, so the pre-telemetry posture applies (parity
    # with a single-shot generation failure): no usage row, attempt left
    # started for reconciliation.
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    attempts = InMemoryAnalysisAttemptStore()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    gateway = _SequencedGateway([RuntimeError("provider down")])
    worker = _iterative_worker(_ScriptedSealedSandbox([]), gateway)

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], InMemoryArtifactStore(), attempts=attempts, ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "failed before execution" in result.error
    assert usage.records == []
    assert (await attempts.get(result.attempt_id)).status == "started"


async def test_checkpoint_failure_after_execution_still_settles_spend():
    # The post-execution on_step (a checkpoint write) can fail too; once this
    # round's charge is retained, no failure may escape without settling.
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    attempts = InMemoryAnalysisAttemptStore()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    gateway = _SequencedGateway([_response("one"), _response("two")])
    sandbox = _ScriptedSealedSandbox([(_sandbox_result(exit_code=1), None)])
    worker = _iterative_worker(sandbox, gateway)

    async def flaky_checkpoint(astate):
        if astate.completed_steps[-1] == "execute":
            raise RuntimeError("checkpoint store unavailable")

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], InMemoryArtifactStore(), attempts=attempts, ledger=ledger
    ).run_analysis(state, on_step=flaky_checkpoint)

    assert not result.ok
    (record,) = usage.records
    assert record.cost_usd == pytest.approx(0.20)
    assert (await attempts.get(result.attempt_id)).status == "settled"


async def test_hostile_report_shapes_left_by_a_round_are_discarded(tmp_path):
    # A malformed round is one os.makedirs / os.symlink away from leaving a
    # directory or a symlink at the report path; cleanup must remove either
    # (never following the symlink) and let the loop keep refining.
    workspace = _workspace(tmp_path)

    def plant_dir(outputs):
        (outputs / "report.md").mkdir()

    def plant_symlink(outputs):
        (outputs / "report.md").symlink_to(outputs.parent / "inputs")

    gateway = _SequencedGateway([
        _response("one"), _response("two"), _response("three"),
    ])
    sandbox = _ScriptedSealedSandbox([
        (_sandbox_result(exit_code=1), plant_dir),
        (_sandbox_result(exit_code=1), plant_symlink),
        (_sandbox_result(exit_code=0), b"# final\n"),
    ])
    worker = _iterative_worker(sandbox, gateway)

    run = await worker.run(workspace, _state())

    assert run.exit_code == 0
    assert run.iterations == 3
    assert (workspace / "outputs" / "report.md").read_bytes() == b"# final\n"
    # The symlink target must have survived the discard untouched.
    assert (workspace / "inputs" / "sales.csv").exists()


async def test_uncapped_agent_runs_iterative_to_completion_and_settles():
    # The default posture: no per-task cap, so the in-run guard never fires
    # and spend is bounded structurally (max_iterations, capped feedback,
    # human approval per run). The full cumulative total still settles.
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    artifacts = InMemoryArtifactStore()
    ledger, usage, agent = _ledger(per_task=None)
    state.agent = agent.metadata.name
    gateway = _SequencedGateway([_response("one"), _response("two")])
    sandbox = _ScriptedSealedSandbox([
        (_sandbox_result(exit_code=1), None),
        (_sandbox_result(exit_code=0), b"# Findings\n"),
    ])
    worker = _iterative_worker(sandbox, gateway)

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], artifacts, ledger=ledger
    ).run_analysis(state)

    assert result.ok, result.error
    assert state.budget_usd is None
    assert result.run.iterations == 2
    assert result.run.cost_usd == pytest.approx(0.40)
    (record,) = usage.records
    assert record.cost_usd == pytest.approx(0.40)
    assert record.outcome == "ok"


async def test_single_shot_over_cap_completion_never_reaches_the_sandbox():
    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    ledger, usage, agent = _ledger(per_task=0.10)
    state.agent = agent.metadata.name
    sandbox = _FakeSealedSandbox()
    worker = BuiltinAnalysisWorker(
        "analysis-model",
        sandbox,
        gateway=_PaidGateway("print('analysis')", cost=0.25),
        limits=SandboxLimits(timeout_seconds=30),
        output_cap_bytes=1_024,
        strategy="single",
    )

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], InMemoryArtifactStore(), ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "per-task budget" in result.error
    assert sandbox.specs == []  # the in-run guard fired before execution
    (record,) = usage.records
    assert record.cost_usd == 0.25
    assert record.outcome == "over_task_budget"


async def test_iterative_mid_loop_failure_settles_cumulative_spend():
    class _TrackingAttempts(InMemoryAnalysisAttemptStore):
        def __init__(self):
            super().__init__()
            self.charges: list[float] = []

        async def charge(self, attempt_id, *, cost_usd, prompt_tokens, completion_tokens):
            self.charges.append(cost_usd)
            return await super().charge(
                attempt_id,
                cost_usd=cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    inputs = InMemoryInputStore()
    state = _state()
    await inputs.stage(InputManifest(
        input_ref=_INPUT_REF,
        files=(InputFile("sales.csv", b"x"),),
    ))
    attempts = _TrackingAttempts()
    ledger, usage, agent = _ledger()
    state.agent = agent.metadata.name
    gateway = _SequencedGateway([_response("one"), _response("two")])
    sandbox = _ScriptedSealedSandbox([
        (_sandbox_result(exit_code=1), None),
        RuntimeError("docker daemon disconnected"),
    ])
    worker = _iterative_worker(sandbox, gateway)

    result = await SealedAnalysisOrchestrator(
        worker, [StagedProvisioner(inputs)], InMemoryArtifactStore(), attempts=attempts, ledger=ledger
    ).run_analysis(state)

    assert not result.ok
    assert "execution setup failed" in result.error
    # Round 1 and round 2 each retained the growing cumulative total before
    # the crash; the final account re-charges the same figure idempotently.
    assert attempts.charges == [
        pytest.approx(0.20), pytest.approx(0.40), pytest.approx(0.40),
    ]
    (record,) = usage.records
    assert record.cost_usd == pytest.approx(0.40)
    assert record.outcome == "ok"
    attempt = await attempts.get(result.attempt_id)
    assert attempt.status == "settled"
    assert attempt.cost_usd == pytest.approx(0.40)
