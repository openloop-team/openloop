"""Supply-chain pins that must not rot: image bases and the Claude CLI."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
DOCKERFILE = ROOT / "Dockerfile"

DIGEST = re.compile(r"@sha256:[0-9a-f]{64}\b")
# A bare build-stage name has no registry, tag, or digest punctuation.
STAGE_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _lines() -> list[str]:
    return DOCKERFILE.read_text().splitlines()


def test_every_from_is_digest_pinned():
    refs = [line.split()[1] for line in _lines() if line.startswith("FROM ")]
    assert refs, "no FROM instructions found"
    for ref in refs:
        assert DIGEST.search(ref), f"unpinned base image: {ref}"


def test_every_copy_from_image_is_digest_pinned():
    refs = re.findall(r"--from=(\S+)", DOCKERFILE.read_text())
    assert refs, "no COPY --from instructions found"
    for ref in refs:
        if STAGE_NAME.fullmatch(ref):
            continue
        assert DIGEST.search(ref), f"unpinned copied image: {ref}"


def test_claude_cli_version_is_concrete():
    versions = [
        line.split("=", 1)[1]
        for line in _lines()
        if line.startswith("ARG CLAUDE_CODE_VERSION=")
    ]
    assert len(versions) == 1, "expected exactly one ARG CLAUDE_CODE_VERSION"
    assert SEMVER.fullmatch(versions[0]), (
        f"Claude CLI must be pinned to X.Y.Z, got {versions[0]!r}"
    )


def test_claude_installer_uses_the_pinned_version():
    installer = [line for line in _lines() if "install.sh" in line]
    assert len(installer) == 1, "expected exactly one Claude installer line"
    assert "${CLAUDE_CODE_VERSION}" in installer[0], (
        "installer must consume ARG CLAUDE_CODE_VERSION, not a channel name"
    )
    for channel in (" stable", " latest"):
        assert channel not in installer[0], f"channel install: {installer[0]}"


def test_project_is_installed_non_editably_into_the_system_environment():
    text = DOCKERFILE.read_text()
    assert "ENV UV_PROJECT_ENVIRONMENT=/usr/local" in text, (
        "uv would otherwise sync into /app/.venv, which is on no PATH"
    )
    sync = [line for line in _lines() if "uv sync" in line]
    assert len(sync) == 1, "expected exactly one uv sync line"
    for flag in ("--locked", "--inexact", "--no-editable"):
        assert flag in sync[0], f"missing {flag}: {sync[0]}"
    assert "pip install" not in text, "pip install ignores uv.lock"


WORKFLOWS = ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
ACTION_SHA = re.compile(r"^[\w.-]+/[\w.-]+(?:/[^@]+)?@[0-9a-f]{40}$")
EXPRESSION = re.compile(r"\$\{\{")


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _steps(path: Path, job: str) -> list[dict]:
    return _workflow(path)["jobs"][job]["steps"]


def test_publish_job_pins_every_action_to_a_commit_sha():
    steps = _steps(CI, "publish")
    uses = [step["uses"] for step in steps if "uses" in step]
    assert uses, "publish job runs no actions"
    for ref in uses:
        assert ACTION_SHA.fullmatch(ref), f"unpinned action: {ref}"


def test_publish_job_declares_every_permission_it_needs():
    permissions = _workflow(CI)["jobs"]["publish"]["permissions"]
    assert permissions == {
        "contents": "read",
        "packages": "write",
        "id-token": "write",
        "attestations": "write",
    }


def test_publish_job_disables_the_artifact_storage_record():
    attest = [
        step for step in _steps(CI, "publish") if "attest" in step.get("uses", "")
    ]
    assert len(attest) == 1, "expected exactly one attestation step"
    # Defaults to true under push-to-registry and would need
    # artifact-metadata: write, a scope the enumerated block leaves at none.
    assert attest[0]["with"]["create-storage-record"] is False


def test_publish_job_keeps_expressions_out_of_shell():
    for step in _steps(CI, "publish"):
        assert not EXPRESSION.search(step.get("run", "")), (
            f"expression interpolated into run: {step.get('name')}"
        )
