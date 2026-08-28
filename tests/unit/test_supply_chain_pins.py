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


def _publish_step(name: str) -> dict:
    steps = [s for s in _steps(CI, "publish") if s.get("name") == name]
    assert len(steps) == 1, f"expected exactly one {name!r} step"
    return steps[0]


def test_publish_job_builds_for_amd64_only():
    steps = _steps(CI, "publish")
    runs = "\n".join(step.get("run", "") for step in steps)
    platforms = re.findall(r"--platform[ =](\S+)", runs)
    assert platforms, "no --platform flag found in publish job"
    # A second, differing --platform anywhere in the job would mean some
    # step targets an architecture other than what was smoke tested.
    assert platforms == ["linux/amd64"], f"unexpected --platform values: {platforms}"


def test_publish_job_build_step_attests_provenance_and_sbom():
    run = _publish_step("Build and push to a staging tag")["run"]
    assert "--provenance=mode=max" in run, "build must emit max-mode provenance"
    assert "--sbom=true" in run, "build must emit an SBOM"


def test_publish_job_build_args_match_dockerfile_arg_defaults():
    run = _publish_step("Build and push to a staging tag")["run"]
    for name in ("OPENLOOP_BROKER_UID", "OPENLOOP_DATA_GID"):
        defaults = [
            line.split("=", 1)[1]
            for line in _lines()
            if line.startswith(f"ARG {name}=")
        ]
        assert len(defaults) == 1, f"expected exactly one ARG {name} in Dockerfile"
        default = defaults[0]
        # Read the default from the Dockerfile instead of hardcoding it here
        # so the two sides can't silently drift apart. Anchor the match to require
        # the value to end at a token boundary (whitespace or end of string) to
        # catch drift where a longer value is used (e.g. 100025 vs 10002).
        pattern = re.escape(f"--build-arg {name}={default}") + r"(?:\s|$)"
        assert re.search(pattern, run), (
            f"--build-arg {name} does not match Dockerfile default {default!r}"
        )


def test_publish_job_staging_tag_is_scoped_to_run_and_attempt():
    staging_tag = _publish_step("Build and push to a staging tag")["env"]["STAGING_TAG"]
    # run_id alone is stable across re-runs; without run_attempt, a re-run
    # would push a different build over a tag an earlier attempt already
    # smoke tested.
    assert "github.run_id" in staging_tag, f"staging tag missing run_id: {staging_tag}"
    assert "github.run_attempt" in staging_tag, (
        f"staging tag missing run_attempt: {staging_tag}"
    )


def test_publish_job_smoke_test_runs_broker_probe_as_broker_user():
    run = _publish_step("Smoke test the pushed digest")["run"]
    # The image's default USER would not exercise /nonexistent as home;
    # the broker composition always runs as this uid:gid.
    assert "--user 10002:10777" in run, "broker probe must run as the broker uid:gid"


def test_publish_job_smoke_test_reads_claude_version_from_dockerfile_at_run_time():
    run = _publish_step("Smoke test the pushed digest")["run"]
    assert "Dockerfile" in run, (
        "expected Claude version must be sourced from the Dockerfile"
    )
    assert "CLAUDE_CODE_VERSION" in run, "must read the ARG CLAUDE_CODE_VERSION default"
    # A literal X.Y.Z here would mean the step hardcodes the version instead
    # of reading it, letting it silently drift from the Dockerfile pin.
    assert not re.search(r"\d+\.\d+\.\d+", run), f"hardcoded version literal in: {run}"


def test_publish_job_tags_commit_sha_only_after_smoke_test():
    names = [step.get("name") for step in _steps(CI, "publish")]
    smoke_idx = names.index("Smoke test the pushed digest")
    publish_idx = names.index("Publish the commit tag")
    assert publish_idx > smoke_idx, (
        "commit tag must not be published before the image is smoke tested"
    )


def test_publish_job_commit_tag_handles_absent_matching_and_differing_digest():
    run = _publish_step("Publish the commit tag")["run"]
    assert 'if [ -z "$existing" ]' in run, "missing branch: create when tag is absent"
    assert "docker buildx imagetools create" in run, "absent branch must create the tag"
    assert 'elif [ "$existing" != "$DIGEST" ]' in run, (
        "missing branch: fail when digest differs"
    )
    assert "exit 1" in run, "differing-digest branch must fail the job"
    # No explicit else: a matching digest falls through as a no-op, which is
    # the third branch of the present/absent/different check.


PROMOTE = WORKFLOWS / "promote-image.yml"


def test_promote_workflow_pins_every_action_to_a_commit_sha():
    jobs = _workflow(PROMOTE)["jobs"]
    uses = [
        step["uses"] for job in jobs.values() for step in job["steps"] if "uses" in step
    ]
    assert uses, "promotion workflow runs no actions"
    for ref in uses:
        assert ACTION_SHA.fullmatch(ref), f"unpinned action: {ref}"


def test_promote_workflow_keeps_expressions_out_of_shell():
    # A Git tag may contain shell metacharacters — v$(id) is creatable and
    # push-triggers this workflow. Interpolation happens before the shell
    # parses, so the tag must arrive as an environment variable.
    jobs = _workflow(PROMOTE)["jobs"]
    for job in jobs.values():
        for step in job["steps"]:
            assert not EXPRESSION.search(step.get("run", "")), (
                f"expression interpolated into run: {step.get('name')}"
            )


def test_promote_workflow_serializes_by_version_tag():
    concurrency = _workflow(PROMOTE)["concurrency"]
    assert concurrency["group"] == "promote-${{ github.ref_name }}"
    assert concurrency["cancel-in-progress"] is False
    # queue: single would cancel a pending run instead of letting it observe
    # the first run's digest and fail on the mismatch.
    assert concurrency["queue"] == "max"
