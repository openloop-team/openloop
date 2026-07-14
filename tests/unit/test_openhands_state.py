"""Unit tests for host-owned OpenHands state layout and key derivation."""

from __future__ import annotations

import base64
import os

import pytest

from openloop.tools.openhands_state import (
    OpenHandsKeyDeriver,
    OpenHandsStateError,
    OpenHandsStateLayout,
    default_openhands_state_root,
    validate_state_identifier,
)


def _encoded_key(byte: int = 7) -> str:
    return base64.urlsafe_b64encode(bytes([byte]) * 32).decode("ascii")


def test_default_state_root_is_under_system_temp():
    assert default_openhands_state_root().name == "openhands"
    assert default_openhands_state_root().parent.name == "openloop"


def test_layout_creates_disjoint_private_job_directories(tmp_path):
    layout = OpenHandsStateLayout(tmp_path / "state")
    paths = layout.for_job("job-1")

    assert paths.agent_server.parent == paths.root
    assert paths.conversations.parent == paths.agent_server
    assert paths.artifacts.parent == paths.root
    assert paths.artifacts not in paths.agent_server.parents
    for directory in (
        layout.root,
        layout.jobs_root,
        paths.root,
        paths.agent_server,
        paths.conversations,
        paths.artifacts,
    ):
        assert directory.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "../job", "job/child", "job\\child", "\x00bad", " space"],
)
def test_identifiers_reject_traversal_and_unsafe_components(value):
    with pytest.raises(OpenHandsStateError):
        validate_state_identifier(value, field="job_id")


def test_layout_refuses_prepositioned_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "state"
    (root / "jobs").mkdir(parents=True)
    os.symlink(outside, root / "jobs" / "job-1")

    layout = OpenHandsStateLayout(root)
    with pytest.raises(OpenHandsStateError, match="symlink"):
        layout.for_job("job-1")


def test_key_derivation_is_stable_separated_and_per_job():
    keys = OpenHandsKeyDeriver.from_base64(_encoded_key(), master_key_id="key-v1")

    assert keys.conversation_key("job-1") == keys.conversation_key("job-1")
    assert keys.conversation_key("job-1") != keys.artifact_key("job-1")
    assert keys.artifact_key("job-1") != keys.artifact_key("job-2")
    assert base64.urlsafe_b64decode(keys.conversation_secret("job-1")) == (
        keys.conversation_key("job-1")
    )


@pytest.mark.parametrize("encoded", ["", "not base64!", base64.b64encode(b"short").decode()])
def test_master_key_must_be_valid_base64_and_exactly_32_bytes(encoded):
    with pytest.raises(OpenHandsStateError):
        OpenHandsKeyDeriver.from_base64(encoded, master_key_id="key-v1")


def test_key_repr_redacts_master_secret():
    encoded = _encoded_key(9)
    keys = OpenHandsKeyDeriver.from_base64(encoded, master_key_id="key-v1")

    assert encoded not in repr(keys)
    assert "redacted" in repr(keys)
