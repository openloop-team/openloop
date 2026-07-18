"""Opt-in proof of the fixed broker-owned Docker generation profile.

Run on a Linux Docker host with both pinned images already present:

    OPENLOOP_BROKER_RUNTIME_LIVE=1 pytest -q \
        tests/integration/test_broker_runtime_docker_live.py
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from openloop.broker_runtime import (
    DockerOpenHandsRuntimeDriver,
    DockerRuntimeConfig,
    OpenHandsGenerationSpec,
    RuntimeDriverError,
    RuntimeResourceState,
)
from openloop.broker_runtime.docker_policy import derive_generation_names


LIVE_ENABLED = os.environ.get("OPENLOOP_BROKER_RUNTIME_LIVE") == "1"


def _docker_usable() -> bool:
    if not LIVE_ENABLED or shutil.which("docker") is None:
        return False
    completed = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    return completed.returncode == 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_ENABLED,
        reason="set OPENLOOP_BROKER_RUNTIME_LIVE=1 for the real Docker proof",
    ),
    pytest.mark.skipif(sys.platform != "linux", reason="requires a Linux Docker host"),
    pytest.mark.skipif(not _docker_usable(), reason="no usable Docker daemon"),
]


def _inspect_container(name: str) -> dict[str, object]:
    completed = subprocess.run(
        ["docker", "container", "inspect", name],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    documents = json.loads(completed.stdout)
    assert isinstance(documents, list) and len(documents) == 1
    return documents[0]


def _image_present(image: str) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    return completed.returncode == 0


@pytest.mark.asyncio
async def test_real_generation_is_private_healthy_and_self_expiring():
    root = Path(tempfile.mkdtemp(prefix="olr-live-", dir="/tmp"))
    runtime_root = root / "r"
    state_root = root / "s"
    runtime_root.mkdir(mode=0o700)
    state_root.mkdir(mode=0o700)
    config = DockerRuntimeConfig(
        runtime_root,
        state_root,
        maximum_lifetime_seconds=90,
        kill_after_seconds=2,
        reconciliation_grace_seconds=1,
    )
    driver = DockerOpenHandsRuntimeDriver(config)
    deadline = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(
        seconds=45
    )
    spec = OpenHandsGenerationSpec(
        operation_id=uuid4(),
        job_id=uuid4(),
        conversation_id=uuid4(),
        generation=1,
        deadline=deadline,
        relay_capability=secrets.token_urlsafe(32),
        session_api_key=secrets.token_urlsafe(32),
        conversation_secret=secrets.token_urlsafe(32),
    )
    names = derive_generation_names(spec.identity)

    try:
        for image in (config.resolved_agent_image, config.relay_image):
            if not await asyncio.to_thread(_image_present, image):
                pytest.skip(f"immutable live-test image is unavailable: {image}")
        ensured = await driver.ensure(spec)
        assert ensured.observation.complete

        for name in (names.agent, names.relay):
            document = await asyncio.to_thread(_inspect_container, name)
            host = document["HostConfig"]
            network = document["NetworkSettings"]
            assert host["PortBindings"] in (None, {})
            ports = network["Ports"]
            assert ports in (None, {}) or all(
                bindings in (None, []) for bindings in ports.values()
            )

        stop = time.monotonic() + max(
            0.0,
            (deadline - datetime.now(timezone.utc)).total_seconds(),
        ) + config.kill_after_seconds + 15
        statuses = ("running", "running")
        while time.monotonic() < stop:
            observed_statuses = []
            for name in (names.agent, names.relay):
                document = await asyncio.to_thread(_inspect_container, name)
                observed_statuses.append(str(document["State"]["Status"]))
            statuses = tuple(observed_statuses)
            if statuses == ("exited", "exited"):
                break
            await asyncio.sleep(0.25)
        assert statuses == ("exited", "exited")

        observed = await driver.inspect(spec.identity)
        assert observed.agent is RuntimeResourceState.EXITED
        assert observed.relay is RuntimeResourceState.EXITED
        assert observed.expired

        first = await driver.release(spec.identity)
        second = await driver.release(spec.identity)
        assert first == second
    finally:
        with suppress(RuntimeDriverError):
            await driver.release(spec.identity)
        shutil.rmtree(root, ignore_errors=True)
