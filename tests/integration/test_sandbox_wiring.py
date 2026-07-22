"""Host-only application sandbox selection."""

from openloop.config import Settings
from openloop.sandbox import HostSandbox
from openloop.wiring import builders


def test_default_is_host_sandbox() -> None:
    assert isinstance(builders.build_worker_sandbox(Settings()), HostSandbox)


def test_builtin_worker_rejects_docker_marker(caplog) -> None:
    with caplog.at_level("ERROR"):
        sandbox = builders.build_worker_sandbox(
            Settings(coding_worker_sandbox="docker")
        )

    assert sandbox is None
    assert "host-only" in caplog.text


def test_unknown_sandbox_value_fails_closed(caplog) -> None:
    with caplog.at_level("ERROR"):
        sandbox = builders.build_worker_sandbox(
            Settings(coding_worker_sandbox="dokcer")
        )

    assert sandbox is None
    assert "unknown CODING_WORKER_SANDBOX" in caplog.text
