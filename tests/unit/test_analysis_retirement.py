"""Regression contract for retiring the dedicated analysis-worker product."""

from pathlib import Path

from openloop.config import Settings


ROOT = Path(__file__).resolve().parents[2]

RETIRED_PATHS = (
    ROOT / "src/openloop/tools/analysis_worker.py",
    ROOT / "src/openloop/workflows/analysis_worker.py",
    ROOT / "src/openloop/surfaces/slack_files.py",
    ROOT / "docker/analysis.Dockerfile",
    ROOT / "docker-compose.sandbox.yml",
)

RETIRED_RUNTIME_MARKERS = (
    "analysis_worker",
    "analysis.report:write",
    "analysis://",
    "analysis_staged_inputs",
    "analysis_uploads",
    "analysis_artifacts",
    "analysis_attempts",
    "openloop.tools.openhands_docker",
)

RETIRED_GUIDANCE_MARKERS = (
    "ANALYSIS_WORKER_",
    "analysis.report:write",
    "openloop analysis stage",
    "docker-compose.sandbox.yml",
    "DockerSandbox",
    "HardenedDockerWorkspace",
    "openhands_docker.py",
)


def test_analysis_product_modules_are_absent() -> None:
    assert [str(path.relative_to(ROOT)) for path in RETIRED_PATHS if path.exists()] == []
    assert not list((ROOT / "src/openloop/analysis").glob("*.py"))


def test_analysis_settings_are_absent() -> None:
    assert not {
        name for name in Settings.model_fields if name.startswith("analysis_worker")
    }


def test_analysis_runtime_markers_are_absent_from_product_source() -> None:
    product = "\n".join(
        path.read_text()
        for path in sorted((ROOT / "src/openloop").rglob("*.py"))
    )
    assert [marker for marker in RETIRED_RUNTIME_MARKERS if marker in product] == []


def test_app_sandbox_has_no_docker_backend() -> None:
    import openloop.sandbox as sandbox

    assert not hasattr(sandbox, "DockerSandbox")


def test_active_operator_guidance_has_no_retired_paths() -> None:
    active = "\n".join(
        (ROOT / name).read_text() for name in ("README.md", ".env.example")
    )
    assert [marker for marker in RETIRED_GUIDANCE_MARKERS if marker in active] == []


def test_destructive_sql_inlines_operator_safety_contract() -> None:
    sql = (
        ROOT / "ops/postgres/2026-07-22-retire-analysis-worker.sql"
    ).read_text()

    assert "docs/operations/retire-analysis-worker.md" not in sql
    for requirement in (
        "ON_ERROR_STOP=1",
        "take and verify the normal PostgreSQL backup",
        "Drain every old application replica",
        "2026-07-22-audit-analysis-worker.sql",
        "A missing shared table is a stop condition",
        "require every dedicated and shared category count to be",
        "After the DROP, rollback requires",
    ):
        assert requirement in sql
