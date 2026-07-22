import fcntl
import io
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import openloop.broker_control.local_receipts as local_receipts_mod
from openloop.broker_control.local_receipts import (
    LocalCheckpointReceiptProblem,
    LocalCheckpointReceiptStore,
    ReadOnlyCheckpointReceiptLocator,
    _dedicated_receipt_relpath,
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


def _shared_receipt_store(tmp_path, artifacts, verifier, private):
    receipt_root = tmp_path / "shared-receipts"
    receipt_root.mkdir()
    os.chown(receipt_root, -1, os.getgid())
    receipt_root.chmod(0o2750)
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
        receipt_root=receipt_root,
        shared_gid=os.getgid(),
    )
    return store, receipt_root


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


async def test_shared_receipt_tree_dual_write_replay_and_read_only_lookup(tmp_path):
    store, key, descriptor, verifier, artifacts, private = _receipt_fixture(tmp_path)
    first = await store.publish(key, descriptor)
    private_sidecar = (
        artifacts.layout.jobs_root
        / str(key.job_id)
        / "artifacts"
        / str(key.conversation_id)
        / "receipts"
        / store._filename(key)
    )
    private_bytes = private_sidecar.read_bytes()

    shared_store, receipt_root = _shared_receipt_store(
        tmp_path, artifacts, verifier, private
    )
    replay = await shared_store.publish(key, descriptor)
    dedicated_sidecar = receipt_root / _dedicated_receipt_relpath(key)

    assert replay == first
    assert private_sidecar.read_bytes() == private_bytes
    assert dedicated_sidecar.read_bytes() == private_bytes
    assert stat.S_IMODE(dedicated_sidecar.stat().st_mode) == 0o440
    assert dedicated_sidecar.stat().st_gid == os.getgid()
    for directory in (
        receipt_root / key.tenant_id,
        receipt_root / key.tenant_id / str(key.job_id),
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o2750
        assert directory.stat().st_gid == os.getgid()

    locator = ReadOnlyCheckpointReceiptLocator(
        root=receipt_root,
        verifier=verifier,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    assert await locator.lookup(key) == first

    # Store lookup remains on the private artifact tree; the shared mirror is
    # only for broker-side recovery.
    dedicated_sidecar.unlink()
    assert await shared_store.lookup(key) == first

    await shared_store.publish(key, descriptor)
    replay_bytes = dedicated_sidecar.read_bytes()
    await shared_store.publish(key, descriptor)
    assert dedicated_sidecar.read_bytes() == replay_bytes == private_bytes


async def test_read_only_locator_missing_tampered_and_wrong_owner_fail_closed(
    tmp_path,
):
    _, key, descriptor, verifier, artifacts, private = _receipt_fixture(tmp_path)
    store, receipt_root = _shared_receipt_store(
        tmp_path, artifacts, verifier, private
    )
    await store.publish(key, descriptor)
    locator = ReadOnlyCheckpointReceiptLocator(
        root=receipt_root,
        verifier=verifier,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )

    assert await locator.lookup(replace(key, barrier_id="missing-barrier")) is None

    dedicated_sidecar = receipt_root / _dedicated_receipt_relpath(key)
    hardlink = dedicated_sidecar.parent / "receipt-hardlink"
    os.link(dedicated_sidecar, hardlink)
    with pytest.raises(LocalCheckpointReceiptProblem):
        await locator.lookup(key)
    hardlink.unlink()

    dedicated_sidecar.chmod(0o640)
    dedicated_sidecar.write_bytes(b"tampered.receipt.bytes")
    dedicated_sidecar.chmod(0o440)
    with pytest.raises(LocalCheckpointReceiptProblem):
        await locator.lookup(key)

    repaired = await store.publish(key, descriptor)
    assert await locator.lookup(key) == repaired

    wrong_owner = ReadOnlyCheckpointReceiptLocator(
        root=receipt_root,
        verifier=verifier,
        expected_uid=os.getuid() + 1,
        expected_gid=os.getgid(),
    )
    with pytest.raises(LocalCheckpointReceiptProblem):
        await wrong_owner.lookup(key)


async def test_read_only_locator_treats_missing_mount_as_unavailable(tmp_path):
    _, key, _, verifier, _, _ = _receipt_fixture(tmp_path)
    locator = ReadOnlyCheckpointReceiptLocator(
        root=tmp_path / "missing-receipt-mount",
        verifier=verifier,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )

    with pytest.raises(LocalCheckpointReceiptProblem):
        await locator.lookup(key)


async def test_shared_receipt_replay_does_not_replace_equal_sidecar(
    tmp_path, monkeypatch
):
    _, key, descriptor, verifier, artifacts, private = _receipt_fixture(tmp_path)
    store, _ = _shared_receipt_store(tmp_path, artifacts, verifier, private)
    signed = await store.publish(key, descriptor)

    def unexpected_replace(*_args, **_kwargs):
        raise AssertionError("equal replay must not replace the shared sidecar")

    monkeypatch.setattr(local_receipts_mod.os, "replace", unexpected_replace)

    assert await store.publish(key, descriptor) == signed


async def test_read_only_locator_accepts_verified_inode_unlinked_after_open(
    tmp_path, monkeypatch
):
    _, key, descriptor, verifier, artifacts, private = _receipt_fixture(tmp_path)
    store, receipt_root = _shared_receipt_store(
        tmp_path, artifacts, verifier, private
    )
    signed = await store.publish(key, descriptor)
    sidecar = receipt_root / _dedicated_receipt_relpath(key)
    replacement = sidecar.parent / ".replacement"
    replacement.write_bytes(sidecar.read_bytes())
    replacement.chmod(0o440)
    target_inode = sidecar.stat().st_ino
    original_fstat = local_receipts_mod.os.fstat
    replaced = False

    def racing_fstat(descriptor):
        nonlocal replaced
        info = original_fstat(descriptor)
        if not replaced and info.st_ino == target_inode:
            replaced = True
            os.replace(replacement, sidecar)
            return original_fstat(descriptor)
        return info

    monkeypatch.setattr(local_receipts_mod.os, "fstat", racing_fstat)
    locator = ReadOnlyCheckpointReceiptLocator(
        root=receipt_root,
        verifier=verifier,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )

    assert await locator.lookup(key) == signed
    assert replaced is True


async def test_shared_receipt_sweeps_only_unlocked_stale_temporaries(tmp_path):
    _, key, descriptor, verifier, artifacts, private = _receipt_fixture(tmp_path)
    store, receipt_root = _shared_receipt_store(
        tmp_path, artifacts, verifier, private
    )
    await store.publish(key, descriptor)
    directory = (receipt_root / _dedicated_receipt_relpath(key)).parent
    orphan = directory / ".orphan.checkpoint-receipt.jwt.tmp"
    orphan.write_bytes(b"interrupted publication")
    old = time.time() - local_receipts_mod._ORPHANED_TEMP_MIN_AGE_SECONDS - 1
    os.utime(orphan, (old, old))
    descriptor_fd = os.open(orphan, os.O_RDONLY)
    fcntl.flock(descriptor_fd, fcntl.LOCK_EX)
    try:
        await store.publish(key, descriptor)
        assert orphan.exists()
    finally:
        os.close(descriptor_fd)

    await store.publish(key, descriptor)
    assert not orphan.exists()


async def test_shared_receipt_fsyncs_new_children_and_their_parents(
    tmp_path, monkeypatch
):
    _, key, descriptor, verifier, artifacts, private = _receipt_fixture(tmp_path)
    store, receipt_root = _shared_receipt_store(
        tmp_path, artifacts, verifier, private
    )
    original_fsync = local_receipts_mod.os.fsync
    original_fstat = local_receipts_mod.os.fstat
    synced_inodes = []

    def recording_fsync(descriptor_fd):
        synced_inodes.append(original_fstat(descriptor_fd).st_ino)
        return original_fsync(descriptor_fd)

    monkeypatch.setattr(local_receipts_mod.os, "fsync", recording_fsync)

    await store.publish(key, descriptor)

    tenant = receipt_root / key.tenant_id
    job = tenant / str(key.job_id)
    assert {
        receipt_root.stat().st_ino,
        tenant.stat().st_ino,
        job.stat().st_ino,
    }.issubset(synced_inodes)
