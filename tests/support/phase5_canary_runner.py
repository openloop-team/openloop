"""Linux-side driver for the opt-in Phase 5 real-Docker canary."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from openhands.sdk import Agent, LLM, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

from openloop.config import Settings
from openloop.tools.coding_worker import WorkerState
from openloop.tools.openhands_resume import ResumeDecision, WorkerPaused
from openloop.wiring.broker import _derive_receipt_key, build_broker
from openloop.wiring.builders import build_coding_worker
from tests.support.processes import cleanup_process


_BROKER_TOPOLOGIES = {"coprocess", "subprocess", "managed"}
_IDENTITY_SEED = bytes(range(32))
_RECEIPT_ROOT = bytes([3]) * 32


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
    topology = os.environ.get("OPENLOOP_CANARY_BROKER_MODE", "coprocess")
    if topology not in _BROKER_TOPOLOGIES:
        raise RuntimeError(f"unsupported canary broker topology: {topology}")
    external = topology != "coprocess"
    identity_seed = _IDENTITY_SEED
    receipt_root_bytes = _RECEIPT_ROOT
    if topology == "managed":
        identity_seed = base64.b64decode(
            os.environ["BROKER_IDENTITY_PRIVATE_KEY"], validate=True
        )
        encoded_receipts = json.loads(os.environ["BROKER_RECEIPT_ROOTS"])
        receipt_root_bytes = base64.b64decode(
            encoded_receipts["receipt-key-v1"], validate=True
        )
    artifact_state = root / "a"
    artifact_state.mkdir(mode=0o700)
    artifact_state.chmod(0o700)
    if topology == "managed":
        broker_state = root / "unused-state"
        broker_runtime = Path(os.environ["BROKER_RUNTIME_ROOT"])
        broker_ingress = Path(os.environ["BROKER_INGRESS_ROOT"])
        receipt_root = Path(os.environ["BROKER_CHECKPOINT_RECEIPT_ROOT"])
        shared_gid = int(os.environ["BROKER_SHARED_DATA_GID"])
    else:
        broker_state = root / "s"
        broker_runtime = root / "x"
        broker_ingress = root / "i"
        receipt_root = root / "q"
        shared_gid = os.getgid()
    for path in (broker_state, broker_runtime, broker_ingress, receipt_root):
        if topology == "managed" and path != broker_state:
            continue
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    values = dict(
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
        broker_receipt_roots={
            "receipt-key-v1": base64.b64encode(receipt_root_bytes).decode()
        },
        broker_execution_lease_seconds=300,
        broker_generation_deadline_seconds=600,
    )
    if external:
        identity_private = Ed25519PrivateKey.from_private_bytes(identity_seed)
        receipt_private = _derive_receipt_key(
            receipt_root_bytes, "broker-receipt", "receipt-key-v1"
        )
        if topology != "managed":
            socket_root.chmod(0o750)
            broker_runtime.chmod(0o750)
            broker_ingress.chmod(0o2750)
            receipt_root.chmod(0o2750)
            for path in (socket_root, broker_runtime, broker_ingress, receipt_root):
                os.chown(path, -1, shared_gid)
        values.update(
            broker_mode="external",
            broker_dev_in_memory=True,
            broker_ingress_root=str(broker_ingress),
            broker_checkpoint_receipt_root=str(receipt_root),
            broker_shared_data_gid=shared_gid,
            broker_expected_app_uid=os.getuid(),
            broker_identity_private_key=base64.b64encode(identity_seed).decode(),
            broker_identity_public_keys={
                "identity-v1": base64.b64encode(
                    identity_private.public_key().public_bytes_raw()
                ).decode()
            },
            broker_receipt_public_keys={
                "receipt-key-v1": base64.b64encode(
                    receipt_private.public_key().public_bytes_raw()
                ).decode()
            },
            broker_reconcile_interval_seconds=60,
        )
    return Settings(**values)


def _secret_map(values) -> dict[str, str]:
    return {name: secret.get_secret_value() for name, secret in values.items()}


def _broker_environment(settings: Settings) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "BROKER_IDENTITY_PRIVATE_KEY",
        "BROKER_RECEIPT_ROOTS",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_SIGNING_SECRET",
        "SLACK_APP_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "BROKER_MODE": "external",
            "BROKER_DEV_IN_MEMORY": "1",
            "BROKER_CONTROL_SOCKET_DIR": settings.broker_control_socket_dir,
            "BROKER_STATE_ROOT": settings.broker_state_root,
            "BROKER_RUNTIME_ROOT": settings.broker_runtime_root,
            "BROKER_INGRESS_ROOT": settings.broker_ingress_root,
            "BROKER_CHECKPOINT_RECEIPT_ROOT": (
                settings.broker_checkpoint_receipt_root
            ),
            "BROKER_SHARED_DATA_GID": str(settings.broker_shared_data_gid),
            "BROKER_EXPECTED_APP_UID": str(settings.broker_expected_app_uid),
            "BROKER_CAPABILITY_ROOTS": json.dumps(
                _secret_map(settings.broker_capability_roots)
            ),
            "BROKER_RUNTIME_ROOTS": json.dumps(
                _secret_map(settings.broker_runtime_roots)
            ),
            "BROKER_IDENTITY_PUBLIC_KEYS": json.dumps(
                settings.broker_identity_public_keys
            ),
            "BROKER_RECEIPT_PUBLIC_KEYS": json.dumps(
                settings.broker_receipt_public_keys
            ),
            "BROKER_EXECUTION_LEASE_SECONDS": str(
                settings.broker_execution_lease_seconds
            ),
            "BROKER_GENERATION_DEADLINE_SECONDS": str(
                settings.broker_generation_deadline_seconds
            ),
            "BROKER_RECONCILE_INTERVAL_SECONDS": "60",
        }
    )
    return environment


def _wait_for_broker(settings: Settings, process: subprocess.Popen | None) -> None:
    socket_path = Path(settings.broker_control_socket_dir) / "control.sock"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            output, error = process.communicate()
            raise RuntimeError(
                f"broker exited {process.returncode} before healthy:\n"
                f"{(error + output)[-8000:]}"
            )
        probe = socket.socket(socket.AF_UNIX)
        try:
            probe.settimeout(0.25)
            probe.connect(os.fspath(socket_path))
        except OSError:
            time.sleep(0.05)
        else:
            return
        finally:
            probe.close()
    raise RuntimeError("broker did not become healthy")


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
    topology = os.environ.get("OPENLOOP_CANARY_BROKER_MODE", "coprocess")
    if topology not in _BROKER_TOPOLOGIES:
        raise RuntimeError(f"unsupported canary broker topology: {topology}")
    shared = Path(os.environ["OPENLOOP_CANARY_SHARED_ROOT"])
    root = shared / "r"
    root.mkdir(mode=0o700)
    managed = topology == "managed"
    if managed:
        socket_root = Path(os.environ["BROKER_CONTROL_SOCKET_DIR"])
    else:
        socket_root = Path(tempfile.mkdtemp(prefix="olp5-control-", dir="/tmp"))
        socket_root.chmod(0o700)
    broker_job_id: str | None = None
    broker_process: subprocess.Popen | None = None
    try:
        source, base_commit = _repository(root)
        settings = _settings(root, socket_root)
        if topology == "subprocess":
            broker_process = subprocess.Popen(
                [sys.executable, "-m", "openloop.broker_main"],
                env=_broker_environment(settings),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _wait_for_broker(settings, broker_process)
        elif topology == "managed":
            _wait_for_broker(settings, None)
        async with AsyncExitStack() as stack:
            handle = await build_broker(settings, stack)
            assert handle is not None
            worker = build_coding_worker(settings, broker_handle=handle)
            assert worker is not None
            assert handle.checkpoint_store is not None
            if topology == "coprocess":
                assert handle.reconciler is not None
            else:
                assert handle.reconciler is None
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
            if handle.reconciler is not None:
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

        if topology != "managed":
            assert _docker_ids("container", broker_job_id) == []
            assert _docker_ids("network", broker_job_id) == []
        assert not (root / "x" / broker_job_id).exists()
        if topology == "managed":
            ingress = Path(settings.broker_ingress_root)
        else:
            ingress = root / (
                "i" if topology != "coprocess" else "x/.workspace-ingress"
            )
        assert not (ingress / broker_job_id).exists()
        return {
            "broker_job_id": broker_job_id,
            "generations": persisted_generations,
            "statuses": persisted_statuses,
            "status": "terminal",
            "topology": topology,
        }
    finally:
        if topology != "managed":
            _cleanup_runtime(broker_job_id)
        cleanup_process(broker_process)
        shutil.rmtree(root, ignore_errors=True)
        if not managed:
            shutil.rmtree(socket_root, ignore_errors=True)


if __name__ == "__main__":
    print("PHASE5_CANARY_OK " + json.dumps(asyncio.run(_run()), sort_keys=True))
