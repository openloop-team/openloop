"""Linux-side driver for the opt-in Phase 5 real-Docker canary."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from openhands.sdk import Agent, LLM, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

from openloop.config import Settings
from openloop.tools.coding_worker import WorkerState
from openloop.tools.openhands_resume import ResumeDecision, WorkerPaused
from openloop.wiring.broker import build_broker
from openloop.wiring.builders import build_coding_worker


def _root(seed: int) -> str:
    return base64.b64encode(bytes([seed]) * 32).decode()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(root: Path) -> tuple[Path, str]:
    source = root / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.email", "canary@example.invalid")
    _git(source, "config", "user.name", "Phase 5 Canary")
    (source / "tracked.txt").write_text("base\n")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "base")
    base_commit = _git(source, "rev-parse", "HEAD")
    (source / "prepause.txt").write_text("survives parking\n")
    return source, base_commit


def _settings(root: Path, socket_root: Path) -> Settings:
    broker_state = root / "s"
    broker_runtime = root / "x"
    artifact_state = root / "a"
    for path in (broker_state, broker_runtime, artifact_state):
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    return Settings(
        coding_worker_backend="openhands",
        coding_worker_sandbox="docker",
        coding_worker_model="openai/gpt-4o-mini",
        coding_worker_max_iterations=10,
        coding_worker_deadline_seconds=180,
        coding_worker_openhands_broker_enabled=True,
        coding_worker_openhands_state_dir=str(artifact_state),
        coding_worker_openhands_state_master_key=base64.b64encode(b"y" * 32).decode(),
        broker_control_socket_dir=str(socket_root),
        broker_state_root=str(broker_state),
        broker_runtime_root=str(broker_runtime),
        broker_capability_roots={"cap-key-v1": _root(1)},
        broker_runtime_roots={"runtime-key-v1": _root(2)},
        broker_receipt_roots={"receipt-key-v1": _root(3)},
        broker_execution_lease_seconds=300,
        broker_generation_deadline_seconds=600,
    )


def _agent() -> Agent:
    port = int(os.environ["OPENLOOP_CANARY_MODEL_PORT"])
    llm = LLM(
        model="openai/gpt-4o-mini",
        api_key="proof-only",
        base_url=f"http://host.docker.internal:{port}/v1",
        num_retries=0,
        timeout=30,
        input_cost_per_token=0,
        output_cost_per_token=0,
    )
    return Agent(
        llm=llm,
        tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
    )


def _docker_ids(resource: str, broker_job_id: str) -> list[str]:
    command = [
        "docker",
        resource,
        "ls",
        "-q",
        "--filter",
        f"label=openloop.runtime.job={broker_job_id}",
    ]
    if resource == "container":
        command.insert(3, "-a")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Docker discovery failed")
    return result.stdout.split()


def _cleanup_runtime(broker_job_id: str | None) -> None:
    if broker_job_id is None:
        return
    uuid.UUID(broker_job_id)
    containers = _docker_ids("container", broker_job_id)
    if containers:
        subprocess.run(
            ["docker", "rm", "-f", *containers],
            check=False,
            capture_output=True,
        )
    networks = _docker_ids("network", broker_job_id)
    if networks:
        subprocess.run(
            ["docker", "network", "rm", *networks],
            check=False,
            capture_output=True,
        )


async def _run() -> dict[str, object]:
    shared = Path(os.environ["OPENLOOP_CANARY_SHARED_ROOT"])
    root = shared / "r"
    root.mkdir(mode=0o700)
    socket_root = Path(tempfile.mkdtemp(prefix="olp5-control-", dir="/tmp"))
    socket_root.chmod(0o700)
    broker_job_id: str | None = None
    try:
        source, base_commit = _repository(root)
        settings = _settings(root, socket_root)
        async with AsyncExitStack() as stack:
            handle = await build_broker(settings, stack)
            assert handle is not None
            worker = build_coding_worker(settings, broker_handle=handle)
            assert worker is not None
            assert handle.checkpoint_store is not None
            assert handle.reconciler is not None
            worker._build_agent = _agent

            state = WorkerState(
                job_id="phase5-canary-0001",
                repo="acme/phase5-canary",
                instruction="Create the requested proof files.",
                base="main",
                branch="openloop/phase5-canary-0001",
                requester_id="U-CANARY",
            )
            persisted_generations: list[int | None] = []
            persisted_statuses: list[str] = []

            async def persist(current: WorkerState) -> None:
                nonlocal broker_job_id
                resume = current.openhands_resume
                if resume is not None:
                    broker_job_id = resume.broker_job_id
                    persisted_generations.append(resume.broker_generation)
                    persisted_statuses.append(resume.status)

            paused = await worker.run(source, state, persist)
            assert isinstance(paused, WorkerPaused)
            resume = state.openhands_resume
            assert resume is not None
            assert resume.status == "parking"
            assert resume.broker_job_id is not None
            assert resume.broker_generation == 1
            broker_job_id = resume.broker_job_id
            assert 1 in persisted_generations
            assert await asyncio.to_thread(
                worker._docker_adapter.is_parked,
                state.job_id,
                resume.broker_job_id,
                resume.broker_generation,
            )
            report = await handle.reconciler.run_pass()
            assert report.failed_closed == 0
            assert report.error == 0

            resume.transition_to("parked")
            decision = ResumeDecision(
                "accept", paused.decision_id, "event-canary-1", "U-CANARY"
            )
            resume.transition_to(
                "resuming",
                segment_id=uuid.uuid4().hex,
                resolved_event_id=decision.event_id,
                resolved_decision=decision,
            )
            edit = await worker.run(source, state, persist)

            assert resume.status == "terminal"
            assert resume.broker_generation == 2
            assert edit.title == "Cold resume proof"
            assert edit.body == "Recovered after container removal."
            assert edit.workspace_artifact is not None
            assert edit.workspace_artifact.artifact.identity.kind == "checkpoint"
            assert (source / "prepause.txt").read_text() == "survives parking\n"
            assert (source / "resumed.txt").read_text() == "accepted-once\n"
            assert not (source / "OPENLOOP_PR.md").exists()

            with worker.artifact_store.open_verified(
                edit.workspace_artifact.artifact,
                edit.workspace_artifact.artifact.identity,
            ) as verified:
                assert verified.manifest.base_commit == base_commit
                assert verified.manifest.pr_title == "Cold resume proof"

            adapter_state = worker._docker_adapter._jobs[state.job_id]
            inspected = await handle.client.inspect_job(
                adapter_state.broker_job_id, adapter_state.capability
            )
            assert inspected.snapshot.state.value == "terminal"

        assert _docker_ids("container", broker_job_id) == []
        assert _docker_ids("network", broker_job_id) == []
        assert not (root / "x" / broker_job_id).exists()
        assert not (
            root / "x" / ".workspace-ingress" / broker_job_id
        ).exists()
        return {
            "broker_job_id": broker_job_id,
            "generations": persisted_generations,
            "statuses": persisted_statuses,
            "status": "terminal",
        }
    finally:
        _cleanup_runtime(broker_job_id)
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(socket_root, ignore_errors=True)


if __name__ == "__main__":
    print("PHASE5_CANARY_OK " + json.dumps(asyncio.run(_run()), sort_keys=True))
