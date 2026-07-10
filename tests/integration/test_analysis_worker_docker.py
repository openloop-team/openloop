"""Real-Docker happy path for the Phase 1 single-shot worker.

The test uses Python's slim base rather than the deployment image so it stays
small; the generated script only needs the standard library. The worker still
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
    )
    artifacts = InMemoryArtifactStore()
    result = await SealedAnalysisOrchestrator(worker, inputs, artifacts).run_analysis(state)

    assert result.ok, result.error
    assert result.prose_summary == "# Total\n5"
    assert (await artifacts.get(result.artifact_ref)).body == b"# Total\n5\n"
