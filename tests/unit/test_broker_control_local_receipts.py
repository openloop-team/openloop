import io
import os
from contextlib import contextmanager

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openloop.broker_control.local_receipts import (
    LocalCheckpointReceiptProblem,
    LocalCheckpointReceiptStore,
    checkpoint_artifact_identity,
    checkpoint_digest,
)
from openloop.broker_control.receipts import (
    CheckpointReceiptIssuer,
    CheckpointReceiptKey,
    CheckpointReceiptVerifier,
)
from openloop.broker_rpc.keys import VerificationKeySet
from openloop.tools.openhands_artifacts import (
    WorkspaceArtifactManifest,
    WorkspaceArtifactStore,
)
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout
from tests.support.broker_repository_contract import SequenceIds


def _receipt_fixture(tmp_path):
    ids = SequenceIds(700)
    key = CheckpointReceiptKey(
        "tenant-a", ids(), ids(), 3, "../barrier/with spaces/é"
    )
    private = Ed25519PrivateKey.generate()
    verifier = CheckpointReceiptVerifier(
        public_keys=VerificationKeySet({"receipt-v1": private.public_key()}),
        issuer="checkpoint-store",
    )
    artifacts = WorkspaceArtifactStore(
        OpenHandsStateLayout(tmp_path / "state"),
        OpenHandsKeyDeriver(bytes(range(32)), master_key_id="artifact-v1"),
        scratch_root=tmp_path / "scratch",
    )
    descriptor = artifacts.put_atomic(
        checkpoint_artifact_identity(key),
        io.BytesIO(b"checkpoint payload"),
        WorkspaceArtifactManifest(format="git-delta", base_commit="a" * 40),
    )
    store = LocalCheckpointReceiptStore(
        artifact_store=artifacts,
        issuer=CheckpointReceiptIssuer(
            private_key=private,
            key_id="receipt-v1",
            issuer="checkpoint-store",
        ),
        historical_verifier=verifier,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    return store, key, descriptor, verifier, artifacts, private


async def test_local_receipt_publish_lookup_and_claim_equal_replay(tmp_path):
    store, key, descriptor, verifier, artifacts, private = _receipt_fixture(
        tmp_path
    )

    first = await store.publish(key, descriptor)
    replay = await store.publish(key, descriptor)

    assert replay == first
    assert await store.lookup(key) == first
    receipt = verifier.verify(first)
    assert receipt.barrier_id == key.barrier_id
    assert receipt.generation == 3
    assert receipt.base_commit == "a" * 40
    assert receipt.byte_count == descriptor.ciphertext_bytes
    assert key.barrier_id not in checkpoint_digest(key)
    assert key.barrier_id not in descriptor.key

    rotated_private = Ed25519PrivateKey.generate()
    rotated_verifier = CheckpointReceiptVerifier(
        public_keys=VerificationKeySet(
            {
                "receipt-v1": private.public_key(),
                "receipt-v2": rotated_private.public_key(),
            }
        ),
        issuer="checkpoint-store",
    )
    rotated_store = LocalCheckpointReceiptStore(
        artifact_store=artifacts,
        issuer=CheckpointReceiptIssuer(
            private_key=rotated_private,
            key_id="receipt-v2",
            issuer="checkpoint-store",
        ),
        historical_verifier=rotated_verifier,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    assert await rotated_store.publish(key, descriptor) == first


async def test_local_receipt_publish_replaces_stale_pid_temporary(tmp_path):
    store, key, descriptor, _, artifacts, _ = _receipt_fixture(tmp_path)
    receipt_directory = (
        artifacts.layout.jobs_root
        / str(key.job_id)
        / "artifacts"
        / str(key.conversation_id)
        / "receipts"
    )
    receipt_directory.mkdir(mode=0o700)
    name = store._filename(key)
    same_pid = receipt_directory / f".{name}.{os.getpid()}.tmp"
    foreign_pid = receipt_directory / f".{name}.{os.getpid() + 1}.tmp"
    for temporary in (same_pid, foreign_pid):
        temporary.write_bytes(b"interrupted publication")
        temporary.chmod(0o400)

    signed = await store.publish(key, descriptor)

    assert await store.lookup(key) == signed
    assert not same_pid.exists()
    assert not foreign_pid.exists()


async def test_local_receipt_publish_rejects_missing_directory_explicitly(
    tmp_path, monkeypatch
):
    store, key, descriptor, verifier, _, _ = _receipt_fixture(tmp_path)
    signed = await store.publish(key, descriptor)
    expected = verifier.verify(signed)

    @contextmanager
    def missing_directory(_key, *, create):
        assert create is True
        yield None

    monkeypatch.setattr(store, "_receipt_directory", missing_directory)

    with pytest.raises(LocalCheckpointReceiptProblem):
        store._publish_sidecar(key, signed, expected)


async def test_local_receipt_publish_does_not_reread_verified_plaintext(
    tmp_path, monkeypatch
):
    store, key, descriptor, _, artifacts, _ = _receipt_fixture(tmp_path)
    original_open_verified = artifacts.open_verified

    class NoReadStream:
        def __init__(self, stream):
            self._stream = stream

        def read(self, *_args, **_kwargs):
            raise AssertionError("verified plaintext was read again")

        def __getattr__(self, name):
            return getattr(self._stream, name)

    def open_verified(*args, **kwargs):
        verified = original_open_verified(*args, **kwargs)
        verified.stream = NoReadStream(verified.stream)
        return verified

    monkeypatch.setattr(artifacts, "open_verified", open_verified)

    assert await store.publish(key, descriptor) is not None
