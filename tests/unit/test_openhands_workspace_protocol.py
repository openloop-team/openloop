"""Neutral contract between the coding worker and broker workspace adapter."""

from openloop.openhands.workspace_protocol import (
    ArchiveStreamResult,
    OpenHandsWorkspace,
)
from openloop.tools.openhands_broker_workspace import BrokerWorkspaceAdapter


def test_broker_adapter_satisfies_workspace_protocol() -> None:
    assert issubclass(BrokerWorkspaceAdapter, OpenHandsWorkspace)


def test_archive_stream_result_retains_value_semantics() -> None:
    result = ArchiveStreamResult(
        base_commit="a" * 40,
        base_ref="refs/heads/main",
        bytes_written=42,
    )
    assert result == ArchiveStreamResult("a" * 40, "refs/heads/main", 42)
