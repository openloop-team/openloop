from __future__ import annotations

import inspect
import os
import stat
import types
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

import openloop.tools.openhands_relay_profile as profile_module
from openloop.tools.openhands_relay_profile import (
    CONTAINER_RELAY_CAPABILITY_FILE,
    DEFAULT_HAPROXY_RELAY_IMAGE,
    MAX_GENERATION,
    OpenHandsRelayProfileError,
    RelayMode,
    compile_openhands_relay,
    install_relay_artifacts,
)


JOB_ID = uuid.UUID("fc04973b-dc6b-4472-8903-e0981fbbd38e")
CONVERSATION_ID = uuid.UUID("9a1db585-06ba-47cd-952d-cd60c2d0d5d1")
CAPABILITY = "r" * 43
SESSION_KEY = "s" * 43


def _compile(**overrides):
    values = {
        "job_id": JOB_ID,
        "generation": 7,
        "conversation_id": CONVERSATION_ID,
        "relay_capability": CAPABILITY,
        "session_api_key": SESSION_KEY,
        "mode": RelayMode.RUNNING,
    }
    values.update(overrides)
    return compile_openhands_relay(**values)


def test_profile_derives_identity_paths_and_fixed_runtime_policy() -> None:
    compiled = _compile()

    assert str(compiled.endpoint.socket_path) == (
        f"/run/openloop/jobs/{JOB_ID}/7/agent.sock"
    )
    assert compiled.endpoint.conversation_id == CONVERSATION_ID
    assert compiled.endpoint.mode is RelayMode.RUNNING
    assert compiled.runtime.image == DEFAULT_HAPROXY_RELAY_IMAGE
    assert compiled.runtime.publish_ports is False
    assert compiled.runtime.read_only_root is True
    assert compiled.runtime.cap_drop == ("ALL",)
    assert compiled.runtime.no_new_privileges is True
    assert compiled.runtime.memory_bytes == 64 * 1024 * 1024
    assert compiled.runtime.pids_limit == 64
    assert compiled.runtime.tmpfs == ("/tmp:rw,nosuid,nodev,noexec,size=8m",)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"job_id": str(JOB_ID)}, "job id"),
        ({"conversation_id": str(CONVERSATION_ID)}, "conversation id"),
        ({"generation": 0}, "generation"),
        ({"generation": -1}, "generation"),
        ({"generation": MAX_GENERATION + 1}, "generation"),
        ({"relay_capability": "short"}, "relay capability"),
        ({"relay_capability": "r" * 42 + "="}, "relay capability"),
        ({"session_api_key": "contains whitespace" * 3}, "session API key"),
        ({"mode": "running"}, "relay mode"),
    ],
)
def test_profile_rejects_caller_selected_or_malformed_inputs(overrides, match) -> None:
    with pytest.raises(OpenHandsRelayProfileError, match=match):
        _compile(**overrides)


def test_running_config_uses_secret_file_and_contains_no_credentials() -> None:
    compiled = _compile()
    config = compiled.haproxy_config.decode("utf-8")
    conversation = f"/api/conversations/{CONVERSATION_ID}"

    assert (
        "acl relay_capability_ok "
        "req.hdr(X-OpenLoop-Relay-Capability) -m str "
        f"-f {CONTAINER_RELAY_CAPABILITY_FILE}"
    ) in config
    assert CAPABILITY not in config
    assert SESSION_KEY not in config
    assert compiled.capability_file.payload == f"{CAPABILITY}\n".encode()
    assert compiled.capability_file.mode == 0o400
    assert "bind :" not in config
    assert "path_beg /api/" not in config
    assert f"acl path_conversation path -m str {conversation}\n" in config
    assert f"acl path_events path -m str {conversation}/events/search\n" in config
    assert f"acl path_websocket path -m str /sockets/events/{CONVERSATION_ID}" in config
    assert "http-request allow if method_post path_conversations" in config
    assert "http-request allow if method_get path_archive" not in config

    capability_check = config.index("http-request deny deny_status 403 unless")
    capability_delete = config.index(
        "http-request del-header X-OpenLoop-Relay-Capability"
    )
    route_allow = config.index("http-request allow if method_get path_health")
    backend = config.index("default_backend openhands_agent")
    assert capability_check < capability_delete < route_allow < backend


def test_checkpoint_rotates_secret_but_retains_identity_derived_socket() -> None:
    running = _compile()
    checkpoint = _compile(
        mode=RelayMode.CHECKPOINT,
        relay_capability="c" * 43,
    )
    config = checkpoint.haproxy_config.decode("utf-8")

    assert checkpoint.endpoint.socket_path == running.endpoint.socket_path
    assert checkpoint.capability_file.payload == b"c" * 43 + b"\n"
    assert "http-request allow if method_get path_health" in config
    assert "http-request allow if method_get path_archive" in config
    assert "http-request allow if method_post path_conversations" not in config
    assert "http-request allow if { var(txn.valid_websocket)" not in config


def test_compiled_profile_is_deterministic_and_redacts_both_credentials() -> None:
    first = _compile()
    second = _compile()
    rendered = repr(first)

    assert first == second
    assert CAPABILITY not in rendered
    assert SESSION_KEY not in rendered
    assert rendered.count("<redacted>") >= 2


def _open_directory(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY)


def test_installer_writes_fixed_owner_readable_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "artifacts"
    target.mkdir(mode=0o700)
    descriptor = _open_directory(target)
    compiled = _compile()
    try:
        install_relay_artifacts(descriptor, compiled)
    finally:
        os.close(descriptor)

    assert sorted(path.name for path in target.iterdir()) == [
        "haproxy.cfg",
        "relay-capability",
    ]
    assert (target / "haproxy.cfg").read_bytes() == compiled.haproxy_config
    assert (target / "relay-capability").read_bytes() == (
        compiled.capability_file.payload
    )
    for path in target.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o400


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o600])
def test_installer_rejects_wrong_directory_mode(tmp_path: Path, mode: int) -> None:
    target = tmp_path / "artifacts"
    target.mkdir(mode=mode)
    target.chmod(mode)
    descriptor = _open_directory(target)
    try:
        with pytest.raises(OpenHandsRelayProfileError, match="mode 0700"):
            install_relay_artifacts(descriptor, _compile())
    finally:
        os.close(descriptor)
    assert list(target.iterdir()) == []


def test_installer_rejects_regular_file_descriptor(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_bytes(b"owned")
    descriptor = os.open(target, os.O_RDONLY)
    try:
        with pytest.raises(OpenHandsRelayProfileError, match="directory"):
            install_relay_artifacts(descriptor, _compile())
    finally:
        os.close(descriptor)
    assert target.read_bytes() == b"owned"


def test_installer_rejects_directory_not_owned_by_effective_uid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "artifacts"
    target.mkdir(mode=0o700)
    descriptor = _open_directory(target)
    actual = os.fstat(descriptor)
    monkeypatch.setattr(
        profile_module.os,
        "fstat",
        lambda _fd: types.SimpleNamespace(
            st_mode=actual.st_mode,
            st_uid=os.geteuid() + 1,
        ),
    )
    try:
        with pytest.raises(OpenHandsRelayProfileError, match="owned"):
            install_relay_artifacts(descriptor, _compile())
    finally:
        os.close(descriptor)
    assert list(target.iterdir()) == []


def test_installer_rejects_nonempty_directory_without_changes(tmp_path: Path) -> None:
    target = tmp_path / "artifacts"
    target.mkdir(mode=0o700)
    existing = target / "operator-owned"
    existing.write_bytes(b"preserve")
    descriptor = _open_directory(target)
    try:
        with pytest.raises(OpenHandsRelayProfileError, match="empty"):
            install_relay_artifacts(descriptor, _compile())
    finally:
        os.close(descriptor)
    assert existing.read_bytes() == b"preserve"


def test_installer_rejects_existing_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    target = tmp_path / "artifacts"
    target.mkdir(mode=0o700)
    (target / ".haproxy.cfg.tmp").symlink_to(outside)
    descriptor = _open_directory(target)
    try:
        with pytest.raises(OpenHandsRelayProfileError, match="empty"):
            install_relay_artifacts(descriptor, _compile())
    finally:
        os.close(descriptor)
    assert outside.read_bytes() == b"preserve"
    assert (target / ".haproxy.cfg.tmp").is_symlink()


def test_installer_handles_partial_writes(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "artifacts"
    target.mkdir(mode=0o700)
    descriptor = _open_directory(target)
    real_write = os.write

    def partial_write(fd: int, data: bytes) -> int:
        return real_write(fd, data[: max(1, len(data) // 3)])

    monkeypatch.setattr(os, "write", partial_write)
    try:
        install_relay_artifacts(descriptor, _compile())
    finally:
        os.close(descriptor)
    assert (target / "relay-capability").read_bytes() == b"r" * 43 + b"\n"


@pytest.mark.parametrize("failure", ["write", "chmod", "publish", "dir_sync"])
def test_installer_cleans_its_files_after_failure(
    tmp_path: Path,
    monkeypatch,
    failure: str,
) -> None:
    target = tmp_path / "artifacts"
    target.mkdir(mode=0o700)
    descriptor = _open_directory(target)

    if failure == "write":
        monkeypatch.setattr(os, "write", lambda _fd, _data: 0)
    elif failure == "chmod":

        def fail_chmod(_fd: int, _mode: int) -> None:
            raise OSError("chmod failed")

        monkeypatch.setattr(os, "fchmod", fail_chmod)
    elif failure == "publish":
        real_link = os.link
        calls = 0

        def fail_second_publish(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("publish failed")
            return real_link(*args, **kwargs)

        monkeypatch.setattr(os, "link", fail_second_publish)
    else:
        real_fsync = os.fsync

        def fail_directory_sync(fd: int) -> None:
            if fd == descriptor:
                raise OSError("directory sync failed")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fail_directory_sync)

    try:
        with pytest.raises(OpenHandsRelayProfileError, match="install"):
            install_relay_artifacts(descriptor, _compile())
    finally:
        os.close(descriptor)
    assert list(target.iterdir()) == []


def test_installer_never_replaces_target_created_during_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "artifacts"
    target.mkdir(mode=0o700)
    descriptor = _open_directory(target)
    real_link = os.link
    injected = False

    def inject_final_target(source, destination, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            (target / destination).write_bytes(b"preserve")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", inject_final_target)
    try:
        with pytest.raises(OpenHandsRelayProfileError, match="install"):
            install_relay_artifacts(descriptor, _compile())
    finally:
        os.close(descriptor)

    assert (target / "haproxy.cfg").read_bytes() == b"preserve"
    assert sorted(path.name for path in target.iterdir()) == ["haproxy.cfg"]


def test_installer_api_accepts_no_paths_filenames_or_ownership() -> None:
    assert tuple(inspect.signature(install_relay_artifacts).parameters) == (
        "directory_fd",
        "compiled",
    )


def test_installer_rejects_caller_constructed_bundle_without_writing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifacts"
    target.mkdir(mode=0o700)
    descriptor = _open_directory(target)
    forged = replace(_compile(), haproxy_config=b"global\n")
    try:
        with pytest.raises(OpenHandsRelayProfileError, match="compiled"):
            install_relay_artifacts(descriptor, forged)
    finally:
        os.close(descriptor)
    assert list(target.iterdir()) == []
