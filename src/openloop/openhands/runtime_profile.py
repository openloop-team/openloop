"""Dependency-light constants and validation for one OpenHands generation."""

from __future__ import annotations

import platform as host_platform
import re


PINNED_OPENHANDS_VERSION = "1.36.0"
DEFAULT_OPENHANDS_SERVER_IMAGE = (
    "ghcr.io/openhands/agent-server@"
    "sha256:7a308731fed9ba6ace79e97fbab0b6e11fa48b8fe49510c5124163bc03ddf9d5"
)
CONVERSATION_LEASE_TTL_SECONDS = "45"

_DEFAULT_PLATFORM_IMAGES = {
    "linux/amd64": (
        "ghcr.io/openhands/agent-server@"
        "sha256:c21e0323cdc3691b54f9f6d980667a375a5df0e21e4c9c40ecb804f2455dd2ff"
    ),
    "linux/arm64": (
        "ghcr.io/openhands/agent-server@"
        "sha256:d619a0ccffd4ca657c5becab28eccc61b6eea4ea9f1aeda27f39829bbaca8161"
    ),
}
SUPPORTED_DOCKER_PLATFORMS = tuple(_DEFAULT_PLATFORM_IMAGES)
_DIGEST_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_NATIVE_DOCKER_PLATFORMS = {
    "aarch64": "linux/arm64",
    "amd64": "linux/amd64",
    "arm64": "linux/arm64",
    "x86_64": "linux/amd64",
}


class OpenHandsRuntimeProfileError(RuntimeError):
    """The pinned OpenHands runtime profile cannot be used safely."""


def require_immutable_server_image(image: str) -> str:
    if not isinstance(image, str) or not _DIGEST_IMAGE.fullmatch(image):
        raise OpenHandsRuntimeProfileError(
            "OpenHands agent-server image must be pinned by sha256 digest"
        )
    return image


def native_docker_platform(machine: str | None = None) -> str:
    """Select the pinned OCI index's native Linux manifest.

    The 1.36.0 image digest is a multi-platform OCI index. Forcing amd64 on an
    arm64 host runs ``tmux`` under QEMU, whose jemalloc warning is treated as a
    fatal error by pinned ``libtmux``. Selecting the matching immutable child
    manifest avoids emulation and is also the correct production default.
    """
    selected = (machine or host_platform.machine()).lower()
    try:
        return _NATIVE_DOCKER_PLATFORMS[selected]
    except KeyError as exc:
        raise OpenHandsRuntimeProfileError(
            f"unsupported OpenHands Docker host architecture: {selected!r}"
        ) from exc


def runtime_server_image(image: str, platform: str) -> str:
    """Resolve the pinned index to its immutable platform child manifest."""
    require_immutable_server_image(image)
    if image != DEFAULT_OPENHANDS_SERVER_IMAGE:
        return image
    try:
        return _DEFAULT_PLATFORM_IMAGES[platform]
    except KeyError as exc:
        raise OpenHandsRuntimeProfileError(
            f"the pinned OpenHands image has no supported {platform!r} manifest"
        ) from exc


__all__ = [
    "CONVERSATION_LEASE_TTL_SECONDS",
    "DEFAULT_OPENHANDS_SERVER_IMAGE",
    "PINNED_OPENHANDS_VERSION",
    "SUPPORTED_DOCKER_PLATFORMS",
    "OpenHandsRuntimeProfileError",
    "native_docker_platform",
    "require_immutable_server_image",
    "runtime_server_image",
]
