"""Real-Docker happy paths for the single-shot and iterative workers.

The tests use Python's slim base rather than the deployment image so they stay
small; the generated scripts only need the standard library. The worker still
uses the same sealed execution API, stdin code path, mount split, read-out, and
artifact boundary as production.
"""

import shutil
import subprocess

import pytest

from openloop.analysis import InMemoryArtifactStore, InMemoryInputStore, InputFile, InputManifest
from openloop.sandbox import DockerSandbox, SandboxLimits
from openloop.testing import FakeGateway
from openloop.tools.analysis_worker import AnalysisState, BuiltinAnalysisWorker, SealedAnalysisOrchestrator


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _docker_usable(), reason="no usable docker daemon")
IMAGE = "python:3.12-slim"


@pytest.fixture(scope="module", autouse=True)
def _pull_image():
    subprocess.run(["docker", "pull", "-q", IMAGE], check=True, timeout=600)


async def test_single_shot_worker_round_trips_a_sealed_report():
    state = AnalysisState(
        job_id="analysis-live",
        input_ref="fixture:csv",
        instruction="total the values",
    )
    inputs = InMemoryInputStore()
    await inputs.stage(InputManifest(
        job_id=state.job_id,
        input_ref=state.input_ref,
        files=(InputFile("values.csv", b"value\n2\n3\n"),),
    ))
    worker = BuiltinAnalysisWorker(
        "m",
        DockerSandbox(IMAGE, kind="analysis"),
        gateway=FakeGateway(
            "values = [int(line) for line in open('/workspace/inputs/values.csv').read().splitlines()[1:]]\n"
            "open('/workspace/outputs/report.md', 'w').write(f'# Total\\n{sum(values)}\\n')"
        ),
        limits=SandboxLimits(timeout_seconds=30, memory="256m", pids_limit=64),
        output_cap_bytes=100_000,
        strategy="single",
    )
    artifacts = InMemoryArtifactStore()
    result = await SealedAnalysisOrchestrator(worker, inputs, artifacts).run_analysis(state)

    assert result.ok, result.error
    assert result.prose_summary == "# Total\n5"
    assert (await artifacts.get(result.artifact_ref)).body == b"# Total\n5\n"


class _SequencedGateway:
    def __init__(self, replies):
        from openloop.models.gateway import ModelResponse

        self._replies = [
            ModelResponse(text=r, model="m", cost_usd=0.01) for r in replies
        ]
        self.calls = []

    async def complete(self, model, messages, **kwargs):
        self.calls.append(list(messages))
        return self._replies.pop(0)


async def test_iterative_worker_refines_after_real_exec_feedback():
    """Round 1 explores (prints, exits 1); round 2 writes the report — with the
    real sealed sandbox producing the feedback the loop feeds back."""
    state = AnalysisState(
        job_id="analysis-live-iter",
        input_ref="fixture:csv",
        instruction="total the values",
    )
    inputs = InMemoryInputStore()
    await inputs.stage(InputManifest(
        job_id=state.job_id,
        input_ref=state.input_ref,
        files=(InputFile("values.csv", b"value\n2\n3\n"),),
    ))
    gateway = _SequencedGateway([
        # Exploration round: inspect the input, then fail on purpose so the
        # loop must continue.
        "print(open('/workspace/inputs/values.csv').read())\n"
        "raise SystemExit(9)",
        "values = [int(line) for line in open('/workspace/inputs/values.csv').read().splitlines()[1:]]\n"
        "open('/workspace/outputs/report.md', 'w').write(f'# Total\\n{sum(values)}\\n')",
    ])
    worker = BuiltinAnalysisWorker(
        "m",
        DockerSandbox(IMAGE, kind="analysis"),
        gateway=gateway,
        limits=SandboxLimits(timeout_seconds=30, memory="256m", pids_limit=64),
        output_cap_bytes=100_000,
        strategy="iterative",
        max_iterations=3,
    )
    artifacts = InMemoryArtifactStore()
    result = await SealedAnalysisOrchestrator(worker, inputs, artifacts).run_analysis(state)

    assert result.ok, result.error
    assert result.run.iterations == 2
    assert (await artifacts.get(result.artifact_ref)).body == b"# Total\n5\n"
    # The second completion saw the first run's real stdout and exit code.
    feedback = gateway.calls[1][-1]["content"]
    assert "exited with code 9" in feedback
    assert "value\n2\n3" in feedback
