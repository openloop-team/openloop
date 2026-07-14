"""Policy tests for the hardened OpenHands Docker adapter (no Docker needed)."""

from __future__ import annotations

import base64
import io
from contextlib import contextmanager

import pytest

from openloop.tools.openhands_docker import (
    CONVERSATION_LEASE_TTL_SECONDS,
    DEFAULT_OPENHANDS_SERVER_IMAGE,
    HardenedDockerLaunch,
    HardenedDockerWorkspace,
    HardenedDockerWorkspaceError,
    require_immutable_server_image,
)
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout


def _keys():
    encoded = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    return OpenHandsKeyDeriver.from_base64(encoded, master_key_id="key-v1")


def test_default_image_is_an_immutable_multiplatform_digest():
    assert require_immutable_server_image(DEFAULT_OPENHANDS_SERVER_IMAGE) == (
        DEFAULT_OPENHANDS_SERVER_IMAGE
    )


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/openhands/agent-server:latest-python",
        "ghcr.io/openhands/agent-server:1.31.0-python",
        "ghcr.io/openhands/agent-server@sha256:short",
        "",
    ],
)
def test_mutable_or_malformed_images_are_rejected(image):
    with pytest.raises(HardenedDockerWorkspaceError, match="digest"):
        require_immutable_server_image(image)


def test_launch_is_loopback_authenticated_and_mount_limited(tmp_path):
    workspace = tmp_path / "checkout"
    state = tmp_path / "state"
    workspace.mkdir()
    state.mkdir()
    launch = HardenedDockerLaunch(
        image=DEFAULT_OPENHANDS_SERVER_IMAGE,
        workspace=workspace,
        state_dir=state,
        host_port=32123,
        session_api_key="session-secret",
        conversation_secret="conversation-secret",
        network="egress-proxy",
    )

    command = launch.command(container_name="agent-server-test")
    rendered = " ".join(command)
    assert "127.0.0.1:32123:8000" in command
    assert f"{workspace}:/workspace:rw" in command
    assert f"{state}:/openhands-state:rw" in command
    assert "--network egress-proxy" in rendered
    assert command[-3:] == ["--host", "0.0.0.0", "--port", "8000"][-3:]
    assert "session-secret" not in rendered
    assert "conversation-secret" not in rendered
    assert set(launch.environment()) == {
        "OH_SESSION_API_KEYS_0",
        "OH_SECRET_KEY",
        "OH_CONVERSATIONS_PATH",
        "OH_LEASE_TTL_SECONDS",
    }
    assert launch.environment()["OH_LEASE_TTL_SECONDS"] == (
        CONVERSATION_LEASE_TTL_SECONDS
    )
    assert "redacted" in repr(launch)
    assert "session-secret" not in repr(launch)


def test_adapter_builds_per_job_launch_without_exposing_artifacts(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    captured = []
    layout = OpenHandsStateLayout(tmp_path / "state-root")
    adapter = HardenedDockerWorkspace(
        layout=layout,
        keys=_keys(),
        workspace_factory=lambda launch: captured.append(launch) or object(),
        port_allocator=lambda: 32123,
    )

    adapter.create(checkout, "job-1")

    launch = captured[0]
    paths = layout.for_job("job-1")
    assert launch.state_dir == paths.agent_server
    assert launch.state_dir != paths.artifacts
    assert str(paths.artifacts) not in " ".join(
        launch.command(container_name="agent-server-test")
    )
    assert launch.conversation_secret == _keys().conversation_secret("job-1")


def test_adapter_rejects_state_root_inside_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    adapter = HardenedDockerWorkspace(
        layout=OpenHandsStateLayout(checkout / ".openhands-state"),
        keys=_keys(),
        workspace_factory=lambda launch: object(),
        port_allocator=lambda: 32123,
    )

    with pytest.raises(HardenedDockerWorkspaceError, match="disjoint"):
        adapter.create(checkout, "job-1")


class _Response:
    headers = {"X-Archive-Base-Commit": "a" * 40}

    def raise_for_status(self):
        return None

    def iter_bytes(self):
        yield b"part-1"
        yield b"part-2"


class _Client:
    def __init__(self):
        self.requests = []

    @contextmanager
    def stream(self, method, path, params):
        self.requests.append((method, path, params))
        yield _Response()


class _Workspace:
    api_key = "session-key"

    def __init__(self):
        self.client = _Client()


def test_archive_stream_uses_authenticated_client_and_explicit_base(tmp_path):
    adapter = HardenedDockerWorkspace(
        layout=OpenHandsStateLayout(tmp_path / "state"),
        keys=_keys(),
        workspace_factory=lambda launch: object(),
    )
    workspace = _Workspace()
    sink = io.BytesIO()

    result = adapter.stream_git_delta(workspace, sink, base_ref="deadbeef")

    assert sink.getvalue() == b"part-1part-2"
    assert result.base_commit == "a" * 40
    assert result.bytes_written == len(sink.getvalue())
    assert workspace.client.requests == [
        (
            "GET",
            "/file/archive",
            {"path": "/workspace", "format": "git-delta", "base_ref": "deadbeef"},
        )
    ]


def test_archive_stream_refuses_unauthenticated_workspace(tmp_path):
    adapter = HardenedDockerWorkspace(
        layout=OpenHandsStateLayout(tmp_path / "state"),
        keys=_keys(),
        workspace_factory=lambda launch: object(),
    )
    workspace = _Workspace()
    workspace.api_key = None

    with pytest.raises(HardenedDockerWorkspaceError, match="authenticated"):
        adapter.stream_git_delta(workspace, io.BytesIO(), base_ref="main")
