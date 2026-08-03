"""Deployment surfaces that make a release the only way to run production.

`test_release.py` pins the record format; these pin the places that have to
agree with it — the image that must not carry configuration, the bundles that
must cover the files a deployment actually reads, and the units that must
refuse an unpinned selection.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from openloop.release import (
    DEPLOY_BUNDLE_PATTERNS,
    RUNTIME_CONFIG_BUNDLE_PATTERNS,
    Bundle,
    ReleaseRecord,
    bundle_files,
)


ROOT = Path(__file__).parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DEPLOY = ROOT / "docker-compose.deploy.yml"
OVERRIDE = ROOT / "docker-compose.broker.yml"
UNITS = sorted((ROOT / "ops/systemd").glob("*.service"))
RELEASE_ENV = "/etc/openloop/release.env"


def test_the_image_carries_no_agent_definitions_or_settings():
    """An agent edit is a configuration revision, not a rebuild.

    Baked agents would also let one digest behave two ways: mounted files on a
    deployment, image copies anywhere the mount is missing.
    """
    dockerfile = DOCKERFILE.read_text()

    copied = re.findall(r"(?m)^COPY\s+(?!--from)(.+)$", dockerfile)
    sources = [line.split()[0] for line in copied]

    assert "agents" not in sources
    assert not any(source.startswith("configs") for source in sources)
    assert not any(source.endswith(".env") for source in sources)
    assert sorted(sources) == ["pyproject.toml", "src"]


def test_the_deployment_mounts_the_agents_the_image_no_longer_carries():
    runtime = yaml.safe_load(DEPLOY.read_text())["services"]["runtime"]

    assert "./agents:/app/agents:ro" in runtime["volumes"]
    assert runtime["env_file"] == ["./configs/prd/runtime.env"]


def test_the_runtime_reports_the_release_that_selected_it():
    runtime = yaml.safe_load(DEPLOY.read_text())["services"]["runtime"]

    assert runtime["environment"]["RELEASE_ID"] == "${OPENLOOP_RELEASE_ID:-}"
    assert "release_id" in (ROOT / "src/openloop/config.py").read_text()
    assert '"release": settings.release_id' in (
        ROOT / "src/openloop/app.py"
    ).read_text()


def test_every_unit_refuses_an_unpinned_or_missing_selection():
    assert UNITS, "no systemd units to check"
    for unit in UNITS:
        text = unit.read_text()

        # No `-` prefix: a host with no selected release fails activation
        # rather than starting on whatever image is local.
        assert f"EnvironmentFile={RELEASE_ENV}\n" in text
        assert f"ExecStartPre=/usr/bin/test -r {RELEASE_ENV}\n" in text
        gate = next(
            line
            for line in text.splitlines()
            if line.startswith("ExecStartPre=/usr/bin/grep")
        )
        assert gate.endswith(RELEASE_ENV)
        assert "@sha256:[0-9a-f]{64}" in gate
        # -x anchors the whole line without a `$`, which systemd would read as
        # the start of a variable expansion.
        assert "-Exq" in gate
        assert "--no-build" in text and "--pull never" in text


def test_the_release_gate_accepts_only_a_digest_selection(tmp_path):
    """The unit's own gate, run through the same grep the unit invokes.

    The pattern is a POSIX ERE with a `[:space:]` class, so it is checked by
    grep rather than by `re` — which would read the class as a plain character
    set and quietly agree with a wrong pattern.
    """
    grep = shutil.which("grep")
    if grep is None:  # pragma: no cover - grep is present on deployment hosts
        pytest.skip("grep is required to exercise the unit gate")
    gate = next(
        line
        for line in UNITS[0].read_text().splitlines()
        if line.startswith("ExecStartPre=/usr/bin/grep")
    )
    pattern = gate.split('"')[1]
    digest = "a" * 64

    def selects(selection: str) -> bool:
        candidate = tmp_path / "release.env"
        candidate.write_text(f"# generated\n{selection}\nOPENLOOP_RELEASE_ID=x\n")
        return (
            subprocess.run(
                [grep, "-Exq", pattern, str(candidate)], check=False
            ).returncode
            == 0
        )

    assert selects(f"OPENLOOP_IMAGE=ghcr.io/o/openloop@sha256:{digest}")
    assert selects(f"OPENLOOP_IMAGE=registry:5000/openloop@sha256:{digest}")
    assert not selects("OPENLOOP_IMAGE=openloop:local")
    assert not selects("OPENLOOP_IMAGE=ghcr.io/o/openloop:2026-08-03")
    assert not selects(f"OPENLOOP_IMAGE=o@sha256:{digest[:63]}")
    assert not selects(f"#OPENLOOP_IMAGE=o@sha256:{digest}")
    assert not selects(f"OPENLOOP_IMAGE=o@sha256:{digest} extra")


def test_the_bundles_cover_what_a_deployment_actually_reads():
    """Every unit, Compose file, and tracked config file lands in a bundle.

    A file a deployment reads but no bundle digests would change production
    without changing any revision.
    """
    deploy = bundle_files(ROOT, DEPLOY_BUNDLE_PATTERNS)
    runtime_config = bundle_files(ROOT, RUNTIME_CONFIG_BUNDLE_PATTERNS, env="prd")

    assert {"docker-compose.deploy.yml", "docker-compose.broker.yml"} <= set(
        deploy
    )
    for unit in UNITS + [ROOT / "ops/systemd/openloop.target"]:
        assert unit.relative_to(ROOT).as_posix() in deploy

    for path in sorted((ROOT / "configs/prd").glob("*.env")):
        assert path.relative_to(ROOT).as_posix() in runtime_config
    for path in sorted((ROOT / "agents").glob("*.yaml")):
        assert path.relative_to(ROOT).as_posix() in runtime_config

    # Neither bundle claims the other's files, and the dev-only composition
    # belongs to neither.
    assert set(deploy).isdisjoint(runtime_config)
    assert "docker-compose.yml" not in deploy
    assert "docker-compose.build.yml" not in deploy


def test_this_checkout_records_and_re_selects_a_release():
    """The end-to-end property: record here, verify here, select from here."""
    record = ReleaseRecord.from_checkout(
        ROOT,
        image=f"ghcr.io/openloop-team/openloop@sha256:{'a' * 64}",
        doppler={
            "openloop-deploy": "prd",
            "openloop-runtime": "prd",
            "openloop-broker": "prd",
        },
        source_commit="0" * 40,
    )
    reloaded = ReleaseRecord.from_json(record.to_json())

    assert reloaded.release_id == record.release_id
    assert reloaded.verify(ROOT) == []
    assert reloaded.selection()["OPENLOOP_IMAGE"] == record.image

    # A host whose deploy bundle no longer matches is reported per file, so an
    # operator learns what to restore instead of that "something" changed.
    replaced = Bundle(
        name="deploy",
        files={"docker-compose.deploy.yml": "sha256:" + "b" * 64},
    )
    assert record.deploy.drift(replaced) == sorted(
        ["deploy: docker-compose.deploy.yml differs"]
        + [
            f"deploy: {path} is missing"
            for path in record.deploy.files
            if path != "docker-compose.deploy.yml"
        ]
    )
