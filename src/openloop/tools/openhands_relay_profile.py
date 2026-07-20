"""Pure, fixed-policy compiler for one OpenHands generation relay."""

from __future__ import annotations

import os
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openloop.tools.openhands_relay_client import (
    OpenHandsRelayError,
    RelayClientEndpoint,
    RelayMode,
)


DEFAULT_HAPROXY_RELAY_IMAGE = (
    "haproxy@sha256:9f196dc9ec57310a1a430939221b33b36c14113497aca7ead9b13f4d5c2d37f5"
)
CONTAINER_RELAY_CONFIG_FILE = "/run/openloop/config/haproxy.cfg"
CONTAINER_RELAY_CAPABILITY_FILE = "/run/openloop/secrets/relay-capability"
MAX_GENERATION = 2**63 - 1
_CONFIG_BASENAME = "haproxy.cfg"
_CONFIG_TEMP_BASENAME = ".haproxy.cfg.tmp"
_CAPABILITY_BASENAME = "relay-capability"
_CAPABILITY_TEMP_BASENAME = ".relay-capability.tmp"


class OpenHandsRelayProfileError(OpenHandsRelayError):
    """Broker-owned relay identity or fixed profile output is invalid."""


@dataclass(frozen=True, slots=True)
class RelayRuntimePolicy:
    """Fixed launch metadata to be realized exactly by a future runtime driver."""

    image: str = DEFAULT_HAPROXY_RELAY_IMAGE
    publish_ports: bool = False
    read_only_root: bool = True
    cap_drop: tuple[str, ...] = ("ALL",)
    no_new_privileges: bool = True
    memory_bytes: int = 64 * 1024 * 1024
    pids_limit: int = 64
    tmpfs: tuple[str, ...] = ("/tmp:rw,nosuid,nodev,noexec,size=8m",)


@dataclass(frozen=True, slots=True, repr=False)
class RelaySecretFile:
    """One fixed-name, owner-readable secret file artifact."""

    payload: bytes = field(repr=False)
    basename: str = "relay-capability"
    mode: int = 0o400

    def __repr__(self) -> str:
        return (
            "RelaySecretFile(basename='relay-capability', "
            "payload=<redacted>, mode=0o400)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class CompiledOpenHandsRelay:
    """Deterministic artifacts and endpoint for one fixed relay generation."""

    job_id: uuid.UUID
    generation: int
    haproxy_config: bytes = field(repr=False)
    capability_file: RelaySecretFile
    endpoint: RelayClientEndpoint
    runtime: RelayRuntimePolicy = RelayRuntimePolicy()

    def __repr__(self) -> str:
        return (
            "CompiledOpenHandsRelay("
            f"job_id={str(self.job_id)!r}, generation={self.generation}, "
            "haproxy_config=<non-secret bytes>, "
            f"capability_file={self.capability_file!r}, "
            f"endpoint={self.endpoint!r}, runtime={self.runtime!r})"
        )


def _socket_path(job_id: uuid.UUID, generation: int) -> Path:
    return Path(f"/run/openloop/jobs/{job_id}/{generation}/agent.sock")


def _acl_path(endpoint: RelayClientEndpoint, suffix: str = "") -> str:
    return f"/api/conversations/{endpoint.conversation_id}{suffix}"


def _render_haproxy_config(endpoint: RelayClientEndpoint) -> str:
    conversation_path = _acl_path(endpoint)
    event_create_path = _acl_path(endpoint, "/events")
    events_search_path = _acl_path(endpoint, "/events/search")
    run_path = _acl_path(endpoint, "/run")
    confirmation_policy_path = _acl_path(endpoint, "/confirmation_policy")
    confirmation_response_path = _acl_path(
        endpoint, "/events/respond_to_confirmation"
    )
    websocket_path = f"/sockets/events/{endpoint.conversation_id}"

    common_acls = f"""\
  acl method_get method GET
  acl method_post method POST
  acl method_connect method CONNECT
  acl absolute_form url_reg -i ^[a-z][a-z0-9+.-]*://
  acl relay_capability_once req.hdr_cnt(X-OpenLoop-Relay-Capability) eq 1
  acl relay_capability_ok req.hdr(X-OpenLoop-Relay-Capability) -m str -f {CONTAINER_RELAY_CAPABILITY_FILE}
  acl path_health path -m str /health
  acl path_conversations path -m str /api/conversations
  acl path_conversation path -m str {conversation_path}
  acl path_event_create path -m str {event_create_path}
  acl path_events_search path -m str {events_search_path}
  acl path_run path -m str {run_path}
  acl path_confirmation_policy path -m str {confirmation_policy_path}
  acl path_confirmation_response path -m str {confirmation_response_path}
  acl path_archive path -m str /api/file/archive
  acl path_websocket path -m str {websocket_path}
  acl wants_websocket req.hdr(Upgrade) -m str -i websocket
  acl connection_upgrade req.hdr(Connection) -m sub -i upgrade
"""

    if endpoint.mode is RelayMode.RUNNING:
        route_rules = """\
  http-request allow if method_get path_health
  http-request allow if method_post path_conversations
  http-request allow if method_get path_conversation
  http-request allow if method_post path_event_create
  http-request allow if method_get path_events_search
  http-request allow if method_post path_run
  http-request allow if method_post path_confirmation_policy
  http-request allow if method_post path_confirmation_response
  http-request allow if method_get path_archive
  http-request allow if { var(txn.valid_websocket) -m bool }
"""
    else:
        route_rules = """\
  http-request allow if method_get path_health
  http-request allow if method_get path_archive
"""

    return f"""\
global
  log stdout format raw local0
  maxconn 16
  hard-stop-after 10s
  tune.bufsize 16384
  tune.maxrewrite 1024

defaults
  mode http
  log global
  option dontlognull
  timeout connect 5s
  timeout client 30s
  timeout server 30s
  timeout http-request 5s
  timeout http-keep-alive 5s
  timeout tunnel 60s

frontend openhands_generation
  bind {endpoint.socket_path}
  log-format "relay status=%ST bytes=%B termination=%ts"
{common_acls}\
  http-request deny deny_status 403 unless relay_capability_once relay_capability_ok
  http-request deny deny_status 400 if method_connect
  http-request deny deny_status 400 if absolute_form
  http-request set-var(txn.valid_websocket) bool(true) if method_get path_websocket wants_websocket connection_upgrade
  http-request del-header X-OpenLoop-Relay-Capability
  http-request set-header Host agent:8000
  http-request del-header Proxy-Connection
  http-request del-header Keep-Alive
  http-request del-header TE
  http-request del-header Trailer
  http-request del-header Connection
  http-request del-header Upgrade
  http-request set-header Connection upgrade if {{ var(txn.valid_websocket) -m bool }}
  http-request set-header Upgrade websocket if {{ var(txn.valid_websocket) -m bool }}
{route_rules}\
  http-request deny deny_status 403
  default_backend openhands_agent

backend openhands_agent
  mode http
  http-reuse safe
  server agent agent:8000 maxconn 16 init-addr libc,none
"""


def compile_openhands_relay(
    *,
    job_id: uuid.UUID,
    generation: int,
    conversation_id: uuid.UUID,
    relay_capability: str,
    session_api_key: str,
    mode: RelayMode,
) -> CompiledOpenHandsRelay:
    """Compile fixed relay artifacts from broker-owned generation identity."""
    if not isinstance(job_id, uuid.UUID):
        raise OpenHandsRelayProfileError("relay job id must be a UUID")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or not 1 <= generation <= MAX_GENERATION
    ):
        raise OpenHandsRelayProfileError("relay generation is out of range")
    if not isinstance(conversation_id, uuid.UUID):
        raise OpenHandsRelayProfileError("relay conversation id must be a UUID")
    if not isinstance(mode, RelayMode):
        raise OpenHandsRelayProfileError("invalid relay mode")

    try:
        endpoint = RelayClientEndpoint(
            socket_path=_socket_path(job_id, generation),
            conversation_id=conversation_id,
            relay_capability=relay_capability,
            session_api_key=session_api_key,
            mode=mode,
        )
    except Exception as exc:
        message = str(exc)
        if "capability" in message:
            message = "invalid relay capability"
        elif "session API key" in message:
            message = "invalid OpenHands session API key"
        raise OpenHandsRelayProfileError(message) from exc

    return CompiledOpenHandsRelay(
        job_id=job_id,
        generation=generation,
        haproxy_config=_render_haproxy_config(endpoint).encode("utf-8"),
        capability_file=RelaySecretFile(
            payload=f"{relay_capability}\n".encode("ascii")
        ),
        endpoint=endpoint,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("artifact write made no progress")
        remaining = remaining[written:]


def _validate_compiled(compiled: CompiledOpenHandsRelay) -> None:
    if not isinstance(compiled, CompiledOpenHandsRelay):
        raise OpenHandsRelayProfileError("invalid compiled relay bundle")
    try:
        expected = compile_openhands_relay(
            job_id=compiled.job_id,
            generation=compiled.generation,
            conversation_id=compiled.endpoint.conversation_id,
            relay_capability=compiled.endpoint.relay_capability,
            session_api_key=compiled.endpoint.session_api_key,
            mode=compiled.endpoint.mode,
        )
    except Exception as exc:
        raise OpenHandsRelayProfileError("invalid compiled relay bundle") from exc
    if compiled != expected:
        raise OpenHandsRelayProfileError("invalid compiled relay bundle")


def install_relay_artifacts(
    directory_fd: int,
    compiled: CompiledOpenHandsRelay,
) -> None:
    """Install fixed relay artifacts relative to an already-open directory."""
    _validate_compiled(compiled)
    if not isinstance(directory_fd, int) or isinstance(directory_fd, bool):
        raise OpenHandsRelayProfileError(
            "relay artifact descriptor must refer to a directory"
        )

    try:
        directory_stat = os.fstat(directory_fd)
    except OSError as exc:
        raise OpenHandsRelayProfileError(
            "relay artifact descriptor must refer to a directory"
        ) from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise OpenHandsRelayProfileError(
            "relay artifact descriptor must refer to a directory"
        )
    if directory_stat.st_uid != os.geteuid():
        raise OpenHandsRelayProfileError(
            "relay artifact directory must be owned by the installer"
        )
    if stat.S_IMODE(directory_stat.st_mode) != 0o700:
        raise OpenHandsRelayProfileError("relay artifact directory must have mode 0700")
    try:
        entries = os.listdir(directory_fd)
    except (OSError, TypeError) as exc:
        raise OpenHandsRelayProfileError(
            "relay artifact directory cannot be inspected by descriptor"
        ) from exc
    if entries:
        raise OpenHandsRelayProfileError("relay artifact directory must be empty")

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OpenHandsRelayProfileError(
            "relay artifact installation requires O_NOFOLLOW"
        )
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    created_names: list[str] = []
    open_descriptor: int | None = None

    artifacts = (
        (
            _CONFIG_TEMP_BASENAME,
            _CONFIG_BASENAME,
            compiled.haproxy_config,
        ),
        (
            _CAPABILITY_TEMP_BASENAME,
            _CAPABILITY_BASENAME,
            compiled.capability_file.payload,
        ),
    )
    try:
        for temporary_name, final_name, payload in artifacts:
            open_descriptor = os.open(
                temporary_name,
                open_flags,
                0o600,
                dir_fd=directory_fd,
            )
            created_names.append(temporary_name)
            try:
                _write_all(open_descriptor, payload)
                os.fchmod(open_descriptor, 0o400)
                os.fsync(open_descriptor)
            finally:
                os.close(open_descriptor)
                open_descriptor = None
            os.link(
                temporary_name,
                final_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            created_names.append(final_name)
            os.unlink(temporary_name, dir_fd=directory_fd)
            created_names.remove(temporary_name)
        os.fsync(directory_fd)
    except BaseException as exc:
        if open_descriptor is not None:
            try:
                os.close(open_descriptor)
            except OSError:
                pass
        for name in reversed(created_names):
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "relay artifact cleanup could not remove a fixed file: "
                        f"{cleanup_error.__class__.__name__}"
                    )
        if not isinstance(exc, Exception):
            raise
        raise OpenHandsRelayProfileError("failed to install relay artifacts") from exc


__all__ = [
    "CONTAINER_RELAY_CAPABILITY_FILE",
    "CONTAINER_RELAY_CONFIG_FILE",
    "DEFAULT_HAPROXY_RELAY_IMAGE",
    "MAX_GENERATION",
    "CompiledOpenHandsRelay",
    "OpenHandsRelayProfileError",
    "RelayMode",
    "RelayRuntimePolicy",
    "RelaySecretFile",
    "compile_openhands_relay",
    "install_relay_artifacts",
]
