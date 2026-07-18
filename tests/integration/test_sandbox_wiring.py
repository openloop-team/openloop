"""Integration: sandbox selection wiring — fail-closed, never weaker-by-typo.

The sandbox is a security boundary, so its wiring policy differs from the
stores' degrade-don't-fail posture: a requested docker sandbox that can't run
(or an unrecognized value) disables the coding worker rather than silently
executing model-generated edits on the host.
"""

import openloop.app as appmod
from openloop.config import Settings
from openloop.sandbox import DockerSandbox, HostSandbox, SandboxUnavailable
from openloop.wiring import builders


def test_default_is_host_sandbox():
    sandbox = builders.build_worker_sandbox(Settings())
    assert isinstance(sandbox, HostSandbox)


def test_docker_sandbox_built_when_probe_passes(monkeypatch):
    monkeypatch.setattr(
        DockerSandbox, "probe", lambda self, workspace_root=None: None
    )
    sandbox = builders.build_worker_sandbox(
        Settings(
            coding_worker_sandbox="docker",
            coding_worker_sandbox_image="img",
            coding_worker_sandbox_network="egress-proxy",
        )
    )
    assert isinstance(sandbox, DockerSandbox)
    assert sandbox.image == "img"
    assert sandbox.network == "egress-proxy"


def test_probe_receives_the_configured_workspace_root(monkeypatch, tmp_path):
    """What boot verifies must be exactly what attempts later use: the probe
    gets the same workspace root the orchestrator is built with."""
    seen = {}

    def spy(self, workspace_root=None):
        seen["root"] = workspace_root

    monkeypatch.setattr(DockerSandbox, "probe", spy)
    builders.build_worker_sandbox(
        Settings(
            coding_worker_sandbox="docker",
            coding_worker_workspace_dir=str(tmp_path / "ws"),
        )
    )
    assert seen["root"] == tmp_path / "ws"


def test_docker_probe_failure_fails_closed(monkeypatch, caplog):
    def boom(self, workspace_root=None):
        raise SandboxUnavailable("no docker")

    monkeypatch.setattr(DockerSandbox, "probe", boom)
    with caplog.at_level("ERROR"):
        sandbox = builders.build_worker_sandbox(
            Settings(coding_worker_sandbox="docker")
        )
    assert sandbox is None  # caller disables the worker; no host fallback
    assert "probe failed" in caplog.text


def test_unknown_sandbox_value_fails_closed(caplog):
    with caplog.at_level("ERROR"):
        sandbox = builders.build_worker_sandbox(
            Settings(coding_worker_sandbox="dokcer")
        )
    assert sandbox is None
    assert "unknown CODING_WORKER_SANDBOX" in caplog.text


def test_worker_not_registered_when_sandbox_unavailable(monkeypatch, caplog):
    """End to end through create_app: enabled worker + broken docker sandbox
    → the coding_worker tool is absent and the disable is loud."""

    def boom(self, workspace_root=None):
        raise SandboxUnavailable("no docker")

    monkeypatch.setattr(DockerSandbox, "probe", boom)
    monkeypatch.setattr(
        appmod,
        "get_settings",
        lambda: Settings(
            coding_worker_enabled=True,
            coding_worker_sandbox="docker",
            github_token="t",
        ),
    )
    with caplog.at_level("ERROR"):
        app = appmod.create_app()

    from fastapi.testclient import TestClient

    with TestClient(app):
        assert "coding_worker" not in app.state.ctx.tools._tools
        # The rest of the gateway is unaffected (github still registered).
        assert "github" in app.state.ctx.tools._tools
    assert "CODING WORKER DISABLED" in caplog.text
