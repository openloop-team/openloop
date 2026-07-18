import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openloop.broker_rpc.keys import (
    KeyFileProblem,
    VerificationKeySet,
    load_ed25519_private_key,
    load_ed25519_public_key,
)


def _write_keys(directory: Path, name: str):
    private_key = Ed25519PrivateKey.generate()
    private_path = directory / f"{name}-private.pem"
    public_path = directory / f"{name}-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    private_path.chmod(0o400)
    public_path.chmod(0o444)
    return private_key, private_path, public_path


def test_safe_key_loader_accepts_regular_mode_restricted_ed25519_files(tmp_path):
    expected, private_path, public_path = _write_keys(tmp_path, "good")
    private = load_ed25519_private_key(private_path, expected_uid=os.getuid())
    public = load_ed25519_public_key(public_path, expected_uid=os.getuid())
    message = b"broker-identity-test"
    assert public.verify(private.sign(message), message) is None
    assert public.verify(expected.sign(message), message) is None


def test_safe_key_loader_rejects_symlink_and_writable_file(tmp_path):
    _, private_path, public_path = _write_keys(tmp_path, "unsafe")
    symlink = tmp_path / "public-link.pem"
    symlink.symlink_to(public_path)
    with pytest.raises(KeyFileProblem):
        load_ed25519_public_key(symlink, expected_uid=os.getuid())
    private_path.chmod(0o640)
    with pytest.raises(KeyFileProblem):
        load_ed25519_private_key(private_path, expected_uid=os.getuid())


def test_verification_key_set_reload_is_atomic(tmp_path):
    _, _, first_path = _write_keys(tmp_path, "first")
    _, _, second_path = _write_keys(tmp_path, "second")
    keys = VerificationKeySet.load(
        {"issuer-v1": first_path}, expected_uid=os.getuid()
    )
    first = keys.snapshot()
    bad_path = tmp_path / "bad.pem"
    bad_path.write_text("not a key", encoding="utf-8")
    bad_path.chmod(0o444)
    with pytest.raises(KeyFileProblem):
        keys.reload({"issuer-v2": bad_path}, expected_uid=os.getuid())
    assert keys.snapshot() == first
    keys.reload({"issuer-v2": second_path}, expected_uid=os.getuid())
    assert set(keys.snapshot()) == {"issuer-v2"}

