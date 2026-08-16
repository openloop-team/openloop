"""CLI tests — `openloop release` (record, show, verify, select).

The commands are the operator-facing half of the release contract: recording
must refuse anything a deployment could not be held to later, and selecting
must refuse a checkout that no longer holds what was recorded.
"""

import json

import pytest

from openloop.cli import main


DIGEST = "a" * 64
IMAGE = f"ghcr.io/openloop-team/openloop@sha256:{DIGEST}"
DOPPLER = [
    "--doppler",
    "openloop-deploy=prd",
    "--doppler",
    "openloop-runtime=prd",
    "--doppler",
    "openloop-broker=prd",
]


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "openloop"
    (root / "ops/systemd").mkdir(parents=True)
    (root / "ops/docker-socket-adapter").mkdir(parents=True)
    (root / "agents").mkdir()
    (root / "configs/prd").mkdir(parents=True)
    (root / "docker-compose.deploy.yml").write_text("services: {}\n")
    (root / "docker-compose.broker.yml").write_text("services: {}\n")
    (root / "ops/docker-socket-adapter/haproxy.cfg").write_text("global\n")
    (root / "ops/systemd/openloop-runtime.service").write_text("[Unit]\n")
    (root / "ops/systemd/openloop.target").write_text("[Unit]\n")
    (root / "agents/dev.yaml").write_text("name: dev\n")
    (root / "configs/prd/runtime.env").write_text("LOG_LEVEL=info\n")
    return root


def _record(checkout, output, *extra):
    return main(
        [
            "release",
            "record",
            "--image",
            IMAGE,
            *DOPPLER,
            "--root",
            str(checkout),
            "--output",
            str(output),
            *extra,
        ]
    )


def test_record_writes_the_tuple_and_prints_it_when_asked(
    checkout, tmp_path, capsys
):
    record = tmp_path / "releases/2026-08-03.json"

    assert _record(checkout, record, "--source-commit", "0" * 40) == 0
    assert "recorded release" in capsys.readouterr().out

    document = json.loads(record.read_text())
    assert document["image"] == IMAGE
    assert document["doppler"] == {
        "openloop-broker": "prd",
        "openloop-deploy": "prd",
        "openloop-runtime": "prd",
    }
    assert document["config_env"] == "prd"
    assert "agents/dev.yaml" in document["runtime_config"]["files"]

    assert main(["release", "show", str(record)]) == 0
    shown = capsys.readouterr().out
    assert document["release_id"] in shown
    assert IMAGE in shown
    assert "openloop-broker=prd" in shown


def test_record_refuses_a_tag_and_a_missing_environment(
    checkout, tmp_path, capsys
):
    record = tmp_path / "release.json"

    assert (
        main(
            [
                "release",
                "record",
                "--image",
                "openloop:local",
                *DOPPLER,
                "--root",
                str(checkout),
            ]
        )
        == 1
    )
    assert "not pinned" in capsys.readouterr().err

    assert (
        main(
            [
                "release",
                "record",
                "--image",
                IMAGE,
                "--doppler",
                "openloop-runtime=prd",
                "--root",
                str(checkout),
                "--output",
                str(record),
            ]
        )
        == 1
    )
    assert "openloop-broker" in capsys.readouterr().err
    assert not record.exists()


def test_a_recorded_release_is_never_overwritten(checkout, tmp_path, capsys):
    record = tmp_path / "release.json"
    assert _record(checkout, record) == 0
    original = record.read_text()

    # Same tuple: idempotent, and the first record survives.
    assert _record(checkout, record) == 0
    assert "already recorded" in capsys.readouterr().out
    assert record.read_text() == original

    # Different tuple: refused rather than silently replacing history.
    (checkout / "agents/dev.yaml").write_text("name: dev\nmodel: opus\n")
    assert _record(checkout, record) == 1
    assert "immutable" in capsys.readouterr().err
    assert record.read_text() == original


def test_select_refuses_a_checkout_that_drifted(checkout, tmp_path, capsys):
    record = tmp_path / "release.json"
    selection = tmp_path / "etc/release.env"
    assert _record(checkout, record, "--source-commit", "abc1234") == 0

    assert (
        main(
            [
                "release",
                "select",
                str(record),
                "--root",
                str(checkout),
                "--output",
                str(selection),
            ]
        )
        == 0
    )
    written = selection.read_text()
    assert f"OPENLOOP_IMAGE={IMAGE}" in written
    assert "OPENLOOP_RELEASE_ID=" in written

    (checkout / "configs/prd/runtime.env").write_text("LOG_LEVEL=debug\n")
    selection.unlink()
    assert (
        main(
            [
                "release",
                "select",
                str(record),
                "--root",
                str(checkout),
                "--output",
                str(selection),
            ]
        )
        == 1
    )
    errors = capsys.readouterr().err
    assert "drift: runtime-config: configs/prd/runtime.env differs" in errors
    assert "check out abc1234 first" in errors
    assert not selection.exists()


def test_verify_reports_the_running_image_too(checkout, tmp_path, capsys):
    record = tmp_path / "release.json"
    assert _record(checkout, record) == 0

    assert (
        main(["release", "verify", str(record), "--root", str(checkout)]) == 0
    )
    assert "holds release" in capsys.readouterr().out

    assert (
        main(
            [
                "release",
                "verify",
                str(record),
                "--root",
                str(checkout),
                "--image",
                f"ghcr.io/openloop-team/openloop@sha256:{'b' * 64}",
            ]
        )
        == 1
    )
    assert "is not the recorded" in capsys.readouterr().err


def test_a_tampered_record_is_not_selectable(checkout, tmp_path, capsys):
    record = tmp_path / "release.json"
    assert _record(checkout, record) == 0

    document = json.loads(record.read_text())
    document["image"] = f"ghcr.io/openloop-team/openloop@sha256:{'c' * 64}"
    record.write_text(json.dumps(document))

    assert main(["release", "show", str(record)]) == 1
    assert "release id" in capsys.readouterr().err
    assert (
        main(["release", "select", str(record), "--root", str(checkout)]) == 1
    )
