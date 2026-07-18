"""Tests for the dependency-light OpenHands runtime profile."""

import pytest

from openloop.openhands.runtime_profile import (
    CONVERSATION_LEASE_TTL_SECONDS,
    DEFAULT_OPENHANDS_SERVER_IMAGE,
    PINNED_OPENHANDS_VERSION,
    SUPPORTED_DOCKER_PLATFORMS,
    OpenHandsRuntimeProfileError,
    native_docker_platform,
    require_immutable_server_image,
    runtime_server_image,
)
from openloop.tools import openhands_docker as legacy_docker
from openloop.tools import openhands_relay


AMD64_IMAGE = (
    "ghcr.io/openhands/agent-server@"
    "sha256:c21e0323cdc3691b54f9f6d980667a375a5df0e21e4c9c40ecb804f2455dd2ff"
)
ARM64_IMAGE = (
    "ghcr.io/openhands/agent-server@"
    "sha256:d619a0ccffd4ca657c5becab28eccc61b6eea4ea9f1aeda27f39829bbaca8161"
)


def test_runtime_profile_preserves_fixed_policy():
    assert PINNED_OPENHANDS_VERSION == "1.36.0"
    assert CONVERSATION_LEASE_TTL_SECONDS == "45"
    assert SUPPORTED_DOCKER_PLATFORMS == ("linux/amd64", "linux/arm64")
    assert require_immutable_server_image(DEFAULT_OPENHANDS_SERVER_IMAGE) == (
        DEFAULT_OPENHANDS_SERVER_IMAGE
    )
    assert runtime_server_image(
        DEFAULT_OPENHANDS_SERVER_IMAGE, "linux/amd64"
    ) == AMD64_IMAGE
    assert runtime_server_image(
        DEFAULT_OPENHANDS_SERVER_IMAGE, "linux/arm64"
    ) == ARM64_IMAGE


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("x86_64", "linux/amd64"),
        ("amd64", "linux/amd64"),
        ("arm64", "linux/arm64"),
        ("aarch64", "linux/arm64"),
    ],
)
def test_runtime_profile_selects_native_platform(machine, expected):
    assert native_docker_platform(machine) == expected


def test_runtime_profile_rejects_unsupported_architecture():
    with pytest.raises(OpenHandsRuntimeProfileError, match="architecture"):
        native_docker_platform("riscv64")


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/openhands/agent-server:latest-python",
        "ghcr.io/openhands/agent-server@sha256:short",
        "",
    ],
)
def test_runtime_profile_rejects_mutable_or_malformed_images(image):
    with pytest.raises(OpenHandsRuntimeProfileError, match="digest"):
        require_immutable_server_image(image)


def test_runtime_profile_preserves_custom_immutable_image():
    image = "example.invalid/agent@sha256:" + "a" * 64
    assert runtime_server_image(image, "linux/amd64") == image


def test_legacy_adapter_reexports_the_runtime_profile():
    assert legacy_docker.HardenedDockerWorkspaceError is OpenHandsRuntimeProfileError
    assert legacy_docker.PINNED_OPENHANDS_VERSION == PINNED_OPENHANDS_VERSION
    assert (
        legacy_docker.DEFAULT_OPENHANDS_SERVER_IMAGE
        == DEFAULT_OPENHANDS_SERVER_IMAGE
    )
    assert (
        legacy_docker.CONVERSATION_LEASE_TTL_SECONDS
        == CONVERSATION_LEASE_TTL_SECONDS
    )
    assert legacy_docker.SUPPORTED_DOCKER_PLATFORMS is SUPPORTED_DOCKER_PLATFORMS
    assert legacy_docker.native_docker_platform is native_docker_platform
    assert legacy_docker.require_immutable_server_image is require_immutable_server_image
    assert legacy_docker.runtime_server_image is runtime_server_image


def test_relay_facade_uses_the_shared_version_pin():
    assert openhands_relay.PINNED_OPENHANDS_VERSION == PINNED_OPENHANDS_VERSION
