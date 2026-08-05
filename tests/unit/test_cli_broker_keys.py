"""CLI tests — `openloop broker keys` (broker-side public key material).

`.broker.env` must carry the PUBLIC halves of two app-side secrets that live in
`.runtime.env`: the identity signing seed, and the per-version receipt keys
HKDF-derived from the receipt roots. Neither is free-choice — a random value
passes every startup check and then silently fails at verification time — so
this command derives and prints them.

Expectations here recompute the derivation independently (explicit HKDF) rather
than calling the shipped helper, so the test would catch a change to it.
"""

import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from openloop.cli import main

IDENTITY_SEED = b"\x11" * 32
RECEIPT_ROOT = b"\x22" * 32
OTHER_ROOT = b"\x33" * 32


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _public_of(seed: bytes) -> str:
    return _b64(
        Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
    )


def _expected_receipt_public(root: bytes, domain: str, version: str) -> str:
    seed = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"{domain}:{version}".encode(),
    ).derive(root)
    return _public_of(seed)


def _emitted_map(output: str, name: str) -> dict[str, str]:
    """Pull one `NAME={json}` assignment out of the emitted dotenv lines."""
    for line in output.splitlines():
        if line.startswith(f"{name}="):
            return json.loads(line.split("=", 1)[1])
    raise AssertionError(f"{name} was not emitted:\n{output}")


@pytest.fixture
def app_material(monkeypatch, tmp_path):
    """App-side `.runtime.env` values, with cwd moved off the real repo file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BROKER_IDENTITY_PRIVATE_KEY", _b64(IDENTITY_SEED))
    monkeypatch.setenv("BROKER_IDENTITY_KEY_ID", "identity-v1")
    monkeypatch.setenv(
        "BROKER_RECEIPT_ROOTS", json.dumps({"receipt-key-v1": _b64(RECEIPT_ROOT)})
    )
    monkeypatch.setenv("BROKER_RECEIPT_CURRENT_VERSION", "receipt-key-v1")
    monkeypatch.setenv("BROKER_RECEIPT_DOMAIN", "broker-receipt")
    return monkeypatch


def test_emits_public_maps_derived_from_app_side_material(app_material, capsys):
    assert main(["broker", "keys"]) == 0

    out = capsys.readouterr().out
    assert _emitted_map(out, "BROKER_IDENTITY_PUBLIC_KEYS") == {
        "identity-v1": _public_of(IDENTITY_SEED)
    }
    assert _emitted_map(out, "BROKER_RECEIPT_PUBLIC_KEYS") == {
        "receipt-key-v1": _expected_receipt_public(
            RECEIPT_ROOT, "broker-receipt", "receipt-key-v1"
        )
    }


def test_emits_every_receipt_version_not_only_the_current_one(app_material, capsys):
    """A rotation leaves older versions verifiable, so all publics are emitted."""
    app_material.setenv(
        "BROKER_RECEIPT_ROOTS",
        json.dumps(
            {"receipt-key-v1": _b64(RECEIPT_ROOT), "receipt-key-v2": _b64(OTHER_ROOT)}
        ),
    )
    app_material.setenv("BROKER_RECEIPT_CURRENT_VERSION", "receipt-key-v2")

    assert main(["broker", "keys"]) == 0

    assert _emitted_map(capsys.readouterr().out, "BROKER_RECEIPT_PUBLIC_KEYS") == {
        "receipt-key-v1": _expected_receipt_public(
            RECEIPT_ROOT, "broker-receipt", "receipt-key-v1"
        ),
        "receipt-key-v2": _expected_receipt_public(
            OTHER_ROOT, "broker-receipt", "receipt-key-v2"
        ),
    }


def test_receipt_publics_follow_the_configured_domain(app_material, capsys):
    """The domain is HKDF info, so a changed domain must change the output."""
    app_material.setenv("BROKER_RECEIPT_DOMAIN", "other-domain")

    assert main(["broker", "keys"]) == 0

    assert _emitted_map(capsys.readouterr().out, "BROKER_RECEIPT_PUBLIC_KEYS") == {
        "receipt-key-v1": _expected_receipt_public(
            RECEIPT_ROOT, "other-domain", "receipt-key-v1"
        )
    }


def test_refuses_when_the_identity_seed_is_unset(app_material, capsys):
    app_material.delenv("BROKER_IDENTITY_PRIVATE_KEY")

    assert main(["broker", "keys"]) == 1

    captured = capsys.readouterr()
    assert "BROKER_IDENTITY_PRIVATE_KEY" in captured.err
    assert ".runtime.env" in captured.err
    assert captured.out == ""


def test_refuses_when_the_receipt_roots_are_unset(app_material, capsys):
    app_material.delenv("BROKER_RECEIPT_ROOTS")

    assert main(["broker", "keys"]) == 1

    captured = capsys.readouterr()
    assert "BROKER_RECEIPT_ROOTS" in captured.err
    assert ".runtime.env" in captured.err
    assert captured.out == ""


def test_refuses_malformed_app_side_material_without_a_traceback(app_material, capsys):
    app_material.setenv("BROKER_IDENTITY_PRIVATE_KEY", _b64(b"\x11" * 31))

    assert main(["broker", "keys"]) == 1

    assert "32 bytes" in capsys.readouterr().err


def test_refuses_a_malformed_receipt_root_instead_of_deriving_from_it(
    app_material, capsys
):
    """HKDF happily accepts any input length, so an unvalidated root would
    derive a plausible-looking key that verifies nothing. Refuse instead."""
    app_material.setenv(
        "BROKER_RECEIPT_ROOTS", json.dumps({"receipt-key-v1": _b64(b"\x22" * 16)})
    )

    assert main(["broker", "keys"]) == 1

    captured = capsys.readouterr()
    assert "32 bytes" in captured.err
    assert captured.out == ""


def test_refuses_when_the_current_receipt_version_is_absent_from_the_roots(
    app_material, capsys
):
    app_material.setenv("BROKER_RECEIPT_CURRENT_VERSION", "receipt-key-v9")

    assert main(["broker", "keys"]) == 1

    assert "receipt-key-v9" in capsys.readouterr().err


def test_never_prints_private_material(app_material, capsys):
    assert main(["broker", "keys"]) == 0

    captured = capsys.readouterr()
    written = captured.out + captured.err
    assert _b64(IDENTITY_SEED) not in written
    assert _b64(RECEIPT_ROOT) not in written
