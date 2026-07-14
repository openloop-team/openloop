"""Host-owned durable state primitives for the OpenHands Docker backend.

The agent-server sees only a job's ``agent-server`` directory. Workspace
artifacts and their encryption keys stay on the host, outside every checkout.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CONVERSATION_DOMAIN = b"openhands-state\0"
_ARTIFACT_DOMAIN = b"workspace-artifact\0"


class OpenHandsStateError(ValueError):
    """OpenHands state configuration or identity is unsafe."""


def validate_state_identifier(value: str, *, field: str) -> str:
    """Return a path-safe durable identity or raise before filesystem access."""
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise OpenHandsStateError(f"invalid OpenHands {field}")
    if value in {".", ".."}:
        raise OpenHandsStateError(f"invalid OpenHands {field}")
    return value


def default_openhands_state_root() -> Path:
    """The approved local fallback; production may configure shared storage."""
    return Path(tempfile.gettempdir()) / "openloop" / "openhands"


@dataclass(frozen=True, slots=True)
class OpenHandsJobPaths:
    """Host paths for one job. Only ``agent_server`` may enter the container."""

    root: Path
    agent_server: Path
    conversations: Path
    artifacts: Path


class OpenHandsStateLayout:
    """Create and validate the per-job state tree with restrictive modes."""

    def __init__(self, root: str | Path | None = None) -> None:
        selected = Path(root) if root is not None else default_openhands_state_root()
        self.root = self._secure_directory(selected.expanduser(), boundary=None)
        self.jobs_root = self._secure_directory(self.root / "jobs", boundary=self.root)

    def for_job(self, job_id: str) -> OpenHandsJobPaths:
        job_id = validate_state_identifier(job_id, field="job_id")
        job_root = self._secure_directory(
            self.jobs_root / job_id, boundary=self.jobs_root
        )
        agent_server = self._secure_directory(
            job_root / "agent-server", boundary=job_root
        )
        conversations = self._secure_directory(
            agent_server / "conversations", boundary=agent_server
        )
        artifacts = self._secure_directory(job_root / "artifacts", boundary=job_root)
        return OpenHandsJobPaths(
            root=job_root,
            agent_server=agent_server,
            conversations=conversations,
            artifacts=artifacts,
        )

    @staticmethod
    def _secure_directory(path: Path, *, boundary: Path | None) -> Path:
        # Refuse a pre-positioned symlink rather than chmod/mkdir through it.
        if path.is_symlink():
            raise OpenHandsStateError("OpenHands state directory cannot be a symlink")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        if boundary is not None:
            try:
                resolved.relative_to(boundary.resolve(strict=True))
            except ValueError as exc:
                raise OpenHandsStateError(
                    "OpenHands state directory escapes configured root"
                ) from exc
        if not resolved.is_dir():
            raise OpenHandsStateError("OpenHands state path is not a directory")
        resolved.chmod(0o700)
        return resolved


class OpenHandsKeyDeriver:
    """Derive independent per-job keys from one versioned host-only secret."""

    def __init__(self, master_key: bytes, *, master_key_id: str) -> None:
        if len(master_key) != 32:
            raise OpenHandsStateError("OpenHands state master key must be 32 bytes")
        self._master_key = bytes(master_key)
        self.master_key_id = validate_state_identifier(
            master_key_id, field="master_key_id"
        )

    @classmethod
    def from_base64(
        cls, encoded: str, *, master_key_id: str
    ) -> "OpenHandsKeyDeriver":
        if not isinstance(encoded, str) or not encoded:
            raise OpenHandsStateError(
                "OpenHands state master key must be base64-encoded"
            )
        padded = encoded + "=" * (-len(encoded) % 4)
        try:
            decoded = base64.b64decode(
                padded.encode("ascii"), altchars=b"-_", validate=True
            )
        except (UnicodeEncodeError, binascii.Error) as exc:
            raise OpenHandsStateError(
                "OpenHands state master key must be valid base64"
            ) from exc
        return cls(decoded, master_key_id=master_key_id)

    def conversation_key(self, job_id: str) -> bytes:
        job = validate_state_identifier(job_id, field="job_id").encode("utf-8")
        return hmac.new(
            self._master_key, _CONVERSATION_DOMAIN + job, hashlib.sha256
        ).digest()

    def conversation_secret(self, job_id: str) -> str:
        """The stable, per-job base64url value forwarded as ``OH_SECRET_KEY``."""
        return base64.urlsafe_b64encode(self.conversation_key(job_id)).decode("ascii")

    def artifact_key(self, job_id: str) -> bytes:
        job = validate_state_identifier(job_id, field="job_id").encode("utf-8")
        return hmac.new(
            self._master_key, _ARTIFACT_DOMAIN + job, hashlib.sha256
        ).digest()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(master_key=<redacted>, "
            f"master_key_id={self.master_key_id!r})"
        )
