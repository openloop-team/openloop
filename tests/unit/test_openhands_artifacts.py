"""Unit tests for encrypted OpenHands workspace artifacts."""

from __future__ import annotations

import base64
import io
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from openloop.tools.openhands_artifacts import (
    WorkspaceArtifactConflict,
    WorkspaceArtifactError,
    WorkspaceArtifactIdentity,
    WorkspaceArtifactManifest,
    WorkspaceArtifactStore,
    WorkspaceArtifactVerificationError,
)
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout


BASE = "a" * 40


def _keys(byte: int = 3):
    encoded = base64.urlsafe_b64encode(bytes([byte]) * 32).decode("ascii")
    return OpenHandsKeyDeriver.from_base64(encoded, master_key_id="key-v1")


def _identity(job_id="job-1"):
    return WorkspaceArtifactIdentity(job_id, "conversation-1", "segment-1", "paused")


def _manifest():
    return WorkspaceArtifactManifest(format="git-delta", base_commit=BASE)


def _store(tmp_path, *, key_byte=3):
    scratch = tmp_path / f"scratch-{key_byte}"
    return WorkspaceArtifactStore(
        OpenHandsStateLayout(tmp_path / "state"),
        _keys(key_byte),
        scratch_root=scratch,
    )


def _path(store, descriptor):
    return store.layout.root / descriptor.key


def test_encrypted_round_trip_and_private_files(tmp_path):
    store = _store(tmp_path)
    plaintext = b"diff --git a/secret b/secret\n+plaintext-marker\n"
    manifest = WorkspaceArtifactManifest(
        format="git-delta",
        base_commit=BASE,
        pr_title="Cold resume proof",
        pr_body="Recovered from the encrypted final envelope.",
    )
    descriptor = store.put_atomic(_identity(), io.BytesIO(plaintext), manifest)
    artifact_path = _path(store, descriptor)

    assert artifact_path.stat().st_mode & 0o777 == 0o600
    assert b"plaintext-marker" not in artifact_path.read_bytes()
    with store.open_verified(descriptor, _identity()) as verified:
        assert verified.manifest.format == "git-delta"
        assert verified.manifest.base_commit == BASE
        assert verified.manifest.pr_title == "Cold resume proof"
        assert verified.manifest.pr_body == (
            "Recovered from the encrypted final envelope."
        )
        assert verified.stream.read() == plaintext


def test_same_content_is_idempotent_and_different_content_conflicts(tmp_path):
    store = _store(tmp_path)
    first = store.put_atomic(_identity(), io.BytesIO(b"one"), _manifest())
    again = store.put_atomic(_identity(), io.BytesIO(b"one"), _manifest())

    assert again == first
    with pytest.raises(WorkspaceArtifactConflict):
        store.put_atomic(_identity(), io.BytesIO(b"two"), _manifest())
    with store.open_verified(first, _identity()) as verified:
        assert verified.stream.read() == b"one"


def test_provided_plaintext_hash_must_match(tmp_path):
    store = _store(tmp_path)
    manifest = WorkspaceArtifactManifest(
        format="git-delta", base_commit=BASE, plaintext_sha256="0" * 64
    )
    with pytest.raises(WorkspaceArtifactError, match="does not match"):
        store.put_atomic(_identity(), io.BytesIO(b"one"), manifest)


def test_ciphertext_tamper_is_rejected_before_plaintext_exposure(tmp_path):
    store = _store(tmp_path)
    descriptor = store.put_atomic(_identity(), io.BytesIO(b"payload"), _manifest())
    path = _path(store, descriptor)
    data = bytearray(path.read_bytes())
    data[-20] ^= 1
    path.write_bytes(data)

    with pytest.raises(WorkspaceArtifactVerificationError, match="checksum"):
        store.open_verified(descriptor, _identity())


def test_authenticated_tamper_fails_even_with_recomputed_descriptor_hash(tmp_path):
    store = _store(tmp_path)
    descriptor = store.put_atomic(_identity(), io.BytesIO(b"payload"), _manifest())
    path = _path(store, descriptor)
    data = bytearray(path.read_bytes())
    data[-20] ^= 1
    path.write_bytes(data)
    import hashlib

    forged = replace(
        descriptor,
        ciphertext_sha256=hashlib.sha256(data).hexdigest(),
        ciphertext_bytes=len(data),
    )
    with pytest.raises(WorkspaceArtifactVerificationError, match="authentication"):
        store.open_verified(forged, _identity())


def test_wrong_job_identity_and_wrong_master_key_fail_closed(tmp_path):
    store = _store(tmp_path)
    descriptor = store.put_atomic(_identity(), io.BytesIO(b"payload"), _manifest())

    with pytest.raises(WorkspaceArtifactVerificationError, match="identity"):
        store.open_verified(descriptor, _identity("job-2"))

    wrong = WorkspaceArtifactStore(
        store.layout, _keys(8), scratch_root=tmp_path / "wrong-scratch"
    )
    with pytest.raises(WorkspaceArtifactVerificationError, match="authentication"):
        wrong.open_verified(descriptor, _identity())


def test_scratch_plaintext_is_removed_after_close_and_verification_failure(tmp_path):
    store = _store(tmp_path)
    descriptor = store.put_atomic(_identity(), io.BytesIO(b"payload"), _manifest())
    verified = store.open_verified(descriptor, _identity())
    assert list(store.scratch_root.iterdir())
    verified.close()
    assert not list(store.scratch_root.iterdir())

    path = _path(store, descriptor)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    import hashlib

    forged = replace(descriptor, ciphertext_sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(WorkspaceArtifactVerificationError):
        store.open_verified(forged, _identity())
    assert not list(store.scratch_root.iterdir())


def test_delete_is_idempotent(tmp_path):
    store = _store(tmp_path)
    descriptor = store.put_atomic(_identity(), io.BytesIO(b"payload"), _manifest())

    assert store.delete(_identity()) is True
    assert store.delete(_identity()) is False
    assert not _path(store, descriptor).exists()


def test_list_orphans_returns_only_old_artifacts(tmp_path):
    store = _store(tmp_path)
    old = store.put_atomic(_identity(), io.BytesIO(b"old"), _manifest())
    recent_identity = WorkspaceArtifactIdentity(
        "job-1", "conversation-1", "segment-2", "paused"
    )
    store.put_atomic(recent_identity, io.BytesIO(b"new"), _manifest())
    old_time = datetime.now(timezone.utc) - timedelta(days=2)
    os.utime(_path(store, old), (old_time.timestamp(), old_time.timestamp()))

    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    assert store.list_orphans(cutoff) == [_identity()]


def test_artifact_identity_rejects_path_traversal():
    with pytest.raises(ValueError):
        WorkspaceArtifactIdentity("../job", "conversation", "segment", "paused")


def test_artifact_directory_symlink_is_rejected(tmp_path):
    store = _store(tmp_path)
    paths = store.layout.for_job("job-1")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, paths.artifacts / "conversation-1")

    with pytest.raises(WorkspaceArtifactError, match="symlink"):
        store.put_atomic(_identity(), io.BytesIO(b"payload"), _manifest())


def test_non_binary_plaintext_stream_is_rejected_and_cleaned(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(WorkspaceArtifactError, match="binary"):
        store.put_atomic(_identity(), io.StringIO("text"), _manifest())
    assert not list(store.scratch_root.iterdir())


def test_verified_stream_cleanup_is_explicit(tmp_path):
    store = _store(tmp_path)
    descriptor = store.put_atomic(_identity(), io.BytesIO(b"payload"), _manifest())
    with store.open_verified(descriptor, _identity()) as verified:
        scratch_path = next(store.scratch_root.iterdir())
        assert scratch_path.exists()
    assert not scratch_path.exists()
