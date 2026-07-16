"""Static contract for the opt-in sibling-container Compose override."""

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
OVERRIDE = ROOT / "docker-compose.sandbox.yml"
SANDBOX_ROOT = (
    "${OPENLOOP_SANDBOX_ROOT:?Set OPENLOOP_SANDBOX_ROOT to a pre-created "
    "absolute host path}"
)


def _compose(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_sandbox_override_exposes_docker_and_preserves_same_path_root():
    runtime = _compose(OVERRIDE)["services"]["runtime"]

    assert runtime["group_add"] == ["${DOCKER_GID:-0}"]

    mounts = {mount["target"]: mount for mount in runtime["volumes"]}
    socket = mounts["/var/run/docker.sock"]
    assert socket == {
        "type": "bind",
        "source": "${DOCKER_SOCKET:-/var/run/docker.sock}",
        "target": "/var/run/docker.sock",
    }

    workspace = mounts[SANDBOX_ROOT]
    assert workspace["type"] == "bind"
    assert workspace["source"] == workspace["target"] == SANDBOX_ROOT


def test_both_worker_roots_are_isolated_children_of_the_shared_mount():
    environment = _compose(OVERRIDE)["services"]["runtime"]["environment"]

    assert environment["CODING_WORKER_WORKSPACE_DIR"] == f"{SANDBOX_ROOT}/coding"
    assert environment["ANALYSIS_WORKER_WORKSPACE_DIR"] == f"{SANDBOX_ROOT}/analysis"
    assert environment["CODING_WORKER_OPENHANDS_STATE_DIR"] == (
        f"{SANDBOX_ROOT}/openhands-state"
    )


def test_base_stacks_remain_safe_and_document_the_opt_in_override():
    for filename in ("docker-compose.yml", "docker-compose.deploy.yml"):
        path = ROOT / filename
        text = path.read_text()
        runtime = _compose(path)["services"]["runtime"]

        assert all(
            mount.get("target") != "/var/run/docker.sock"
            if isinstance(mount, dict)
            else "/var/run/docker.sock" not in mount
            for mount in runtime.get("volumes", [])
        )
        assert "docker-compose.sandbox.yml" in text


def test_example_environment_does_not_enable_a_worker_without_the_override():
    example = (ROOT / ".env.example").read_text()

    assert "CODING_WORKER_ENABLED=false" in example
    assert "ANALYSIS_WORKER_ENABLED=false" in example
