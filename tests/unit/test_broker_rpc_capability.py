import base64
import os
from uuid import UUID

import pytest

from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker_rpc.capability import (
    CapabilityProblem,
    CapabilityRootRing,
    JobCapability,
    JobCapabilityAuthority,
)


OWNER = BrokerOwner("tenant-a", "workload-a")
JOB_ID = UUID("00000000-0000-4000-8000-000000000201")
ROOT_V1 = bytes(range(32))
ROOT_V2 = bytes(range(32, 64))


def _authority(*, current="cap-v1"):
    return JobCapabilityAuthority(
        CapabilityRootRing(
            {"cap-v1": ROOT_V1, "cap-v2": ROOT_V2},
            current_version=current,
        )
    )


def test_capability_has_fixed_domain_separated_vector_and_redacted_repr():
    authority = _authority()
    metadata = authority.issue_metadata(
        OWNER, JOB_ID, IsolationMode.DEDICATED
    )
    capability = authority.derive(OWNER, JOB_ID, metadata)
    assert metadata.key_version == "cap-v1"
    assert metadata.epoch == 1
    assert capability.value == "gLAya4ZYd0qbaNK4LJkJpFTmGYRADMUrN_kY3pDtxsw"
    assert metadata.capability_digest == (
        "c49f0c2712a35c075d49d30d88f3820924256e4bdd99cd96b053a6f7653b8a3d"
    )
    assert capability.value not in repr(capability)
    assert capability.value not in str(capability)
    assert metadata.capability_digest not in repr(metadata)


@pytest.mark.parametrize(
    ("owner", "job_id", "epoch", "key_version"),
    [
        (BrokerOwner("tenant-b", "workload-a"), JOB_ID, 1, "cap-v1"),
        (BrokerOwner("tenant-a", "workload-b"), JOB_ID, 1, "cap-v1"),
        (OWNER, UUID("00000000-0000-4000-8000-000000000202"), 1, "cap-v1"),
        (OWNER, JOB_ID, 2, "cap-v1"),
        (OWNER, JOB_ID, 1, "cap-v2"),
    ],
)
def test_capability_domains_are_independent(owner, job_id, epoch, key_version):
    authority = _authority()
    original_metadata = authority.issue_metadata(
        OWNER, JOB_ID, IsolationMode.SHARED
    )
    original = authority.derive(OWNER, JOB_ID, original_metadata)
    changed_metadata = type(original_metadata)(
        key_version=key_version,
        epoch=epoch,
        capability_digest=authority.digest_for(owner, job_id, key_version, epoch),
    )
    changed = authority.derive(owner, job_id, changed_metadata)
    assert changed != original


def test_old_job_capability_replays_after_current_root_rotates():
    first = _authority(current="cap-v1")
    metadata = first.issue_metadata(OWNER, JOB_ID, IsolationMode.SHARED)
    capability = first.derive(OWNER, JOB_ID, metadata)
    rotated = _authority(current="cap-v2")
    assert rotated.derive(OWNER, JOB_ID, metadata) == capability
    assert rotated.verify(OWNER, JOB_ID, metadata, capability)
    assert not rotated.verify(
        OWNER,
        JOB_ID,
        metadata,
        JobCapability("A" * 43),
    )


def test_capability_root_file_is_exact_base64url_and_private(tmp_path):
    encoded = base64.urlsafe_b64encode(ROOT_V1).rstrip(b"=") + b"\n"
    path = tmp_path / "cap-v1.key"
    path.write_bytes(encoded)
    path.chmod(0o400)
    ring = CapabilityRootRing.load(
        {"cap-v1": path},
        current_version="cap-v1",
        expected_uid=os.getuid(),
    )
    assert ring.current_version == "cap-v1"
    path.chmod(0o440)
    with pytest.raises(CapabilityProblem):
        CapabilityRootRing.load(
            {"cap-v1": path},
            current_version="cap-v1",
            expected_uid=os.getuid(),
        )


@pytest.mark.parametrize(
    "roots,current",
    [({}, "cap-v1"), ({"cap-v1": b"short"}, "cap-v1"), ({"cap-v1": ROOT_V1}, "missing")],
)
def test_capability_root_ring_rejects_invalid_configuration(roots, current):
    with pytest.raises(CapabilityProblem):
        CapabilityRootRing(roots, current_version=current)
