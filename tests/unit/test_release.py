"""The release tuple: what it pins, what it refuses, what it never stores."""

from __future__ import annotations

import json

import pytest

from openloop.release import (
    DEPLOY_BUNDLE,
    DEPLOY_BUNDLE_PATTERNS,
    RUNTIME_CONFIG_BUNDLE,
    RUNTIME_CONFIG_BUNDLE_PATTERNS,
    Bundle,
    ReleaseError,
    ReleaseRecord,
    bundle_files,
    doppler_environments,
    image_reference,
)


DIGEST = "a" * 64
IMAGE = f"ghcr.io/openloop-team/openloop@sha256:{DIGEST}"
OTHER_IMAGE = f"ghcr.io/openloop-team/openloop@sha256:{'b' * 64}"
DOPPLER = {
    "openloop-broker": "prd",
    "openloop-deploy": "prd",
    "openloop-runtime": "prd",
}


def _checkout(root, *, agent: str = "name: dev\n", unit: str = "[Unit]\n"):
    """A minimal tree holding one file for every bundle pattern."""
    (root / "ops/systemd").mkdir(parents=True)
    (root / "ops/docker-socket-adapter").mkdir(parents=True)
    (root / "agents").mkdir()
    (root / "configs/prd").mkdir(parents=True)
    (root / "docker-compose.deploy.yml").write_text("services: {}\n")
    (root / "docker-compose.broker.yml").write_text("services: {}\n")
    (root / "ops/docker-socket-adapter/haproxy.cfg").write_text("global\n")
    (root / "ops/systemd/openloop-runtime.service").write_text(unit)
    (root / "ops/systemd/openloop.target").write_text("[Unit]\n")
    (root / "agents/dev.yaml").write_text(agent)
    (root / "configs/prd/runtime.env").write_text("LOG_LEVEL=info\n")
    return root


def _record(root, **overrides) -> ReleaseRecord:
    arguments = {"image": IMAGE, "doppler": DOPPLER, **overrides}
    return ReleaseRecord.from_checkout(root, **arguments)


def test_a_tag_is_not_a_release():
    for unpinned in (
        "openloop:local",
        "ghcr.io/openloop-team/openloop:2026-08-03",
        "ghcr.io/openloop-team/openloop",
        f"ghcr.io/openloop-team/openloop@sha256:{'a' * 63}",
        f"ghcr.io/openloop-team/openloop@md5:{DIGEST}",
    ):
        with pytest.raises(ReleaseError, match="not pinned"):
            image_reference(unpinned)

    assert image_reference(f"  {IMAGE}  ") == IMAGE
    assert image_reference(f"registry:5000/openloop@sha256:{DIGEST}")


def test_bundles_are_addressed_independently(tmp_path):
    root = _checkout(tmp_path)
    before = _record(root)

    (root / "agents/dev.yaml").write_text("name: dev\nmodel: sonnet\n")
    after = _record(root)

    assert after.deploy.revision == before.deploy.revision
    assert after.runtime_config.revision != before.runtime_config.revision
    assert after.release_id != before.release_id


def test_the_same_tuple_records_the_same_release(tmp_path):
    root = _checkout(tmp_path)

    first = _record(root, recorded_at="2026-08-03T10:00:00Z")
    second = _record(root, recorded_at="2026-08-04T11:30:00Z")

    # Identity is the tuple, not when someone wrote it down.
    assert first.release_id == second.release_id
    assert _record(root, image=OTHER_IMAGE).release_id != first.release_id
    assert (
        _record(root, doppler={**DOPPLER, "openloop-runtime": "stg"}).release_id
        != first.release_id
    )


def test_a_release_records_names_and_revisions_but_no_values(tmp_path):
    root = _checkout(tmp_path)
    (root / "configs/prd/runtime.env").write_text(
        "LOG_LEVEL=info\nGITHUB_APP_ID=4223747\n"
    )

    document = json.loads(_record(root).to_json())

    # Every leaf is a name, a digest, a timestamp, or the pinned image.
    def leaves(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaves(key)
                yield from leaves(value)
        else:
            yield str(node)

    assert "4223747" not in json.dumps(document)
    for leaf in leaves(document["deploy"]["files"]):
        assert leaf.startswith("sha256:") or "/" in leaf or leaf.endswith(
            (".yml", ".cfg", ".service", ".target", "files", "revision")
        )
    assert document["doppler"] == DOPPLER


def test_a_doppler_token_is_not_a_doppler_environment():
    with pytest.raises(ReleaseError, match="never its token"):
        doppler_environments(
            [
                "openloop-deploy=dp.st.prd.AbC123",
                "openloop-runtime=prd",
                "openloop-broker=prd",
            ]
        )
    with pytest.raises(ReleaseError, match="must be <project>=<config>"):
        doppler_environments(["openloop-deploy"])


def test_a_release_names_every_environment_that_injects_a_secret():
    with pytest.raises(ReleaseError, match="openloop-broker"):
        doppler_environments(
            ["openloop-deploy=prd", "openloop-runtime=prd"]
        )
    assert doppler_environments(
        ["openloop-runtime=prd", "openloop-broker=prd", "openloop-deploy=prd"]
    ) == DOPPLER


def test_an_incomplete_checkout_is_not_a_bundle(tmp_path):
    root = _checkout(tmp_path)
    (root / "docker-compose.broker.yml").unlink()

    with pytest.raises(ReleaseError, match="matched no file"):
        bundle_files(root, DEPLOY_BUNDLE_PATTERNS)


def test_a_bundle_never_digests_a_local_credential_file(tmp_path):
    root = _checkout(tmp_path)
    (root / ".env").write_text("POSTGRES_PASSWORD=hunter2\n")

    with pytest.raises(ReleaseError, match="local credential files"):
        bundle_files(root, (".env",))
    with pytest.raises(ReleaseError, match="local credential files"):
        Bundle(name=DEPLOY_BUNDLE, files={".env": f"sha256:{DIGEST}"})


def test_a_record_vouches_for_itself(tmp_path):
    root = _checkout(tmp_path)
    document = json.loads(_record(root).to_json())

    assert ReleaseRecord.from_json(json.dumps(document)).image == IMAGE

    edited = {**document, "image": OTHER_IMAGE}
    with pytest.raises(ReleaseError, match="release id"):
        ReleaseRecord.from_dict(edited)

    edited = json.loads(json.dumps(document))
    edited["runtime_config"]["revision"] = f"sha256:{'c' * 64}"
    with pytest.raises(ReleaseError, match="does not match the file digests"):
        ReleaseRecord.from_dict(edited)

    edited = json.loads(json.dumps(document))
    edited["deploy"]["files"]["docker-compose.deploy.yml"] = f"sha256:{DIGEST}"
    with pytest.raises(ReleaseError, match="does not match"):
        ReleaseRecord.from_dict(edited)

    with pytest.raises(ReleaseError, match="unknown release fields"):
        ReleaseRecord.from_dict({**document, "postgres_password": "hunter2"})
    with pytest.raises(ReleaseError, match="unsupported release schema"):
        ReleaseRecord.from_dict({**document, "schema": "openloop.release/v9"})


def test_verify_names_every_file_that_drifted(tmp_path):
    root = _checkout(tmp_path)
    record = _record(root)
    assert record.verify(root) == []

    (root / "agents/dev.yaml").write_text("name: dev\nmodel: opus\n")
    (root / "agents/second.yaml").write_text("name: second\n")
    (root / "ops/systemd/openloop.target").write_text("[Unit]\n# edited\n")

    assert sorted(record.verify(root)) == [
        "deploy: ops/systemd/openloop.target differs",
        "runtime-config: agents/dev.yaml differs",
        "runtime-config: agents/second.yaml is not recorded",
    ]


def test_selection_carries_the_image_and_the_release(tmp_path):
    record = _record(_checkout(tmp_path), source_commit="0123abc")

    selection = record.render_selection()
    values = dict(
        line.split("=", 1)
        for line in selection.splitlines()
        if line and not line.startswith("#")
    )

    assert values == {
        "OPENLOOP_IMAGE": IMAGE,
        "OPENLOOP_RELEASE_ID": record.release_id,
    }
    # systemd EnvironmentFile syntax: assignments and comments only.
    assert all(
        line.startswith("#") or "=" in line
        for line in selection.splitlines()
    )
    assert f"# source       {record.source_commit}" in selection
    assert "# doppler      openloop-runtime=prd" in selection


def test_bundle_membership_is_the_documented_split():
    assert "agents/*.yaml" in RUNTIME_CONFIG_BUNDLE_PATTERNS
    assert "configs/{env}/*.env" in RUNTIME_CONFIG_BUNDLE_PATTERNS
    assert not any(
        pattern.startswith(("agents/", "configs/"))
        for pattern in DEPLOY_BUNDLE_PATTERNS
    )
    assert RUNTIME_CONFIG_BUNDLE != DEPLOY_BUNDLE


def test_a_config_env_selects_its_own_runtime_bundle(tmp_path):
    root = _checkout(tmp_path)
    (root / "configs/stg").mkdir()
    (root / "configs/stg/runtime.env").write_text("LOG_LEVEL=debug\n")

    production = _record(root)
    staging = _record(root, config_env="stg")

    assert production.runtime_config.files.keys() == {
        "agents/dev.yaml",
        "configs/prd/runtime.env",
    }
    assert staging.runtime_config.files.keys() == {
        "agents/dev.yaml",
        "configs/stg/runtime.env",
    }
    assert staging.release_id != production.release_id
    assert staging.deploy.revision == production.deploy.revision


def test_a_release_records_the_checkout_it_was_taken_from(tmp_path):
    root = _checkout(tmp_path)

    with pytest.raises(ReleaseError, match="not a git commit"):
        _record(root, source_commit="HEAD~1")
    with pytest.raises(ReleaseError, match="configs/<env> directory name"):
        _record(root, config_env="../../etc")

    assert _record(root, source_commit=None).to_dict().get("source_commit") is None
