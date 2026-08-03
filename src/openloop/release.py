"""The immutable tuple a production deployment resolves to.

A deployment is four independently chosen versions, not one: the application
image, the deploy bundle that composes it, the runtime-configuration bundle it
reads, and the secret-manager environments that inject its credentials. This
module names those choices, addresses each one by content, and writes them into
a record an operator can re-select later.

Two properties are load-bearing:

- **The image is pinned by digest.** A tag is mutable, so a tag cannot identify
  what ran. ``openloop:local`` in particular names an image that only exists on
  whichever host happened to build it.
- **The record carries revisions, never values.** Bundles are recorded as file
  digests and secret-manager environments as project/config *names*, so a
  record is safe to keep, diff, and copy between hosts.

Records are host state (``/var/lib/openloop/releases``), deliberately outside
the bundles they describe: a record that lived in the deploy bundle would
change the revision it was recording.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = "openloop.release/v1"

DEPLOY_BUNDLE = "deploy"
RUNTIME_CONFIG_BUNDLE = "runtime-config"

# What Compose and systemd need to compose the stack. Everything here is
# consumed by a deploy action; none of it is read by the running application.
DEPLOY_BUNDLE_PATTERNS: tuple[str, ...] = (
    "docker-compose.deploy.yml",
    "docker-compose.broker.yml",
    "ops/docker-socket-adapter/haproxy.cfg",
    "ops/systemd/*.service",
    "ops/systemd/*.target",
)

# What the running application reads. Agent definitions belong here rather than
# in the image: an agent change is a configuration revision, not a new build.
RUNTIME_CONFIG_BUNDLE_PATTERNS: tuple[str, ...] = (
    "agents/*.yaml",
    "configs/{env}/*.env",
)

# Every Compose invocation is wrapped in all three, so a release that named
# only some of them would not describe the deployment that ran.
REQUIRED_DOPPLER_PROJECTS: tuple[str, ...] = (
    "openloop-broker",
    "openloop-deploy",
    "openloop-runtime",
)

DEFAULT_CONFIG_ENV = "prd"
DEFAULT_RECORD_ROOT = Path("/var/lib/openloop/releases")
DEFAULT_SELECTION_PATH = Path("/etc/openloop/release.env")

IMAGE_VARIABLE = "OPENLOOP_IMAGE"
RELEASE_ID_VARIABLE = "OPENLOOP_RELEASE_ID"

_DIGEST_REFERENCE = re.compile(
    r"^(?P<repository>[^@\s]+)@sha256:(?P<digest>[0-9a-f]{64})$"
)
_DOPPLER_PROJECT = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_DOPPLER_CONFIG = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_CONFIG_ENV = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
_REVISION = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[0-9a-f]{64}$")

# Local, untracked files that hold real credentials. A bundle pattern must
# never reach one: the record would then depend on a secret's content.
_EXCLUDED_BUNDLE_NAMES = frozenset(
    {".env", ".env.e2e", ".runtime.env", ".broker.env"}
)
_EXCLUDED_BUNDLE_PREFIXES = ("secrets/",)


class ReleaseError(ValueError):
    """A release input that must not be recorded or selected."""


def image_reference(value: str) -> str:
    """Return ``value`` if it pins an image by digest, else refuse.

    Resolve a pushed tag to its digest with::

        docker buildx imagetools inspect <repository>:<tag> --format \\
            '{{println .Manifest.Digest}}'
    """
    candidate = value.strip()
    if not _DIGEST_REFERENCE.match(candidate):
        raise ReleaseError(
            f"image {value!r} is not pinned: a release selects "
            "<repository>@sha256:<64 hex digits>, never a tag"
        )
    return candidate


def config_env_name(value: str) -> str:
    """Return ``value`` if it names a ``configs/<env>`` directory, else refuse.

    A single path segment: the name selects the runtime-configuration bundle,
    so anything else would digest files from outside the checkout.
    """
    if not _CONFIG_ENV.match(value):
        raise ReleaseError(
            f"{value!r} is not a configs/<env> directory name"
        )
    return value


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 16), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _expand(pattern: str, env: str | None) -> str:
    if "{env}" not in pattern:
        return pattern
    if env is None:
        raise ReleaseError(f"pattern {pattern!r} needs a configuration env")
    return pattern.replace("{env}", env)


def _guard_path(relative: str) -> None:
    name = relative.rsplit("/", 1)[-1]
    if name in _EXCLUDED_BUNDLE_NAMES or relative.startswith(
        _EXCLUDED_BUNDLE_PREFIXES
    ):
        raise ReleaseError(
            f"refusing to record {relative}: local credential files are "
            "never part of a bundle"
        )


def bundle_files(
    root: Path,
    patterns: Sequence[str],
    *,
    env: str | None = None,
) -> dict[str, str]:
    """Digest every file the patterns select, keyed by repository path.

    A pattern that selects nothing fails closed. A bundle silently losing a
    file would otherwise produce a revision that looks legitimate and omits
    the deleted content.
    """
    files: dict[str, str] = {}
    for pattern in patterns:
        expanded = _expand(pattern, env)
        matched = sorted(
            path for path in root.glob(expanded) if path.is_file()
        )
        if not matched:
            raise ReleaseError(
                f"{expanded} matched no file under {root} — the checkout is "
                "not a complete bundle source"
            )
        for path in matched:
            relative = path.relative_to(root).as_posix()
            _guard_path(relative)
            files[relative] = _file_digest(path)
    return dict(sorted(files.items()))


def bundle_revision(files: Mapping[str, str]) -> str:
    """Address a file set by content, independently of any other bundle."""
    manifest = "".join(
        f"{path}\0{digest}\n" for path, digest in sorted(files.items())
    )
    return f"sha256:{hashlib.sha256(manifest.encode()).hexdigest()}"


def _canonical(document: Any) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Bundle:
    """A content-addressed set of tracked files with one revision."""

    name: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.files:
            raise ReleaseError(f"{self.name} bundle is empty")
        for path, digest in self.files.items():
            _guard_path(path)
            if not _REVISION.match(digest):
                raise ReleaseError(
                    f"{self.name} bundle entry {path} has no sha256 digest"
                )

    @property
    def revision(self) -> str:
        return bundle_revision(self.files)

    @classmethod
    def from_root(
        cls,
        name: str,
        root: Path,
        patterns: Sequence[str],
        *,
        env: str | None = None,
    ) -> Bundle:
        return cls(name=name, files=bundle_files(root, patterns, env=env))

    def drift(self, observed: Bundle) -> list[str]:
        """Per-file differences from ``observed``; empty means identical."""
        differences = []
        for path in sorted(set(self.files) | set(observed.files)):
            recorded = self.files.get(path)
            current = observed.files.get(path)
            if recorded == current:
                continue
            if recorded is None:
                differences.append(f"{self.name}: {path} is not recorded")
            elif current is None:
                differences.append(f"{self.name}: {path} is missing")
            else:
                differences.append(f"{self.name}: {path} differs")
        return differences


def doppler_environments(pairs: Iterable[str]) -> dict[str, str]:
    """Parse ``project=config`` selections into validated coordinates.

    Names only. The patterns reject anything shaped like a Doppler service
    token, so a mistyped ``--doppler`` cannot write a credential into a record.
    """
    environments: dict[str, str] = {}
    for pair in pairs:
        project, separator, config = pair.partition("=")
        if not separator:
            raise ReleaseError(
                f"--doppler {pair!r} must be <project>=<config>"
            )
        if not _DOPPLER_PROJECT.match(project):
            raise ReleaseError(f"{project!r} is not a Doppler project name")
        if not _DOPPLER_CONFIG.match(config):
            raise ReleaseError(
                f"{config!r} is not a Doppler config name — a release "
                "records which config was used, never its token or values"
            )
        if project in environments and environments[project] != config:
            raise ReleaseError(
                f"project {project!r} selected twice with different configs"
            )
        environments[project] = config
    missing = [
        project
        for project in REQUIRED_DOPPLER_PROJECTS
        if project not in environments
    ]
    if missing:
        raise ReleaseError(
            "release is missing the Doppler environment for "
            + ", ".join(missing)
        )
    return dict(sorted(environments.items()))


@dataclass(frozen=True)
class ReleaseRecord:
    """One selectable deployment: image, both bundles, and its environments."""

    image: str
    config_env: str
    deploy: Bundle
    runtime_config: Bundle
    doppler: Mapping[str, str]
    recorded_at: str
    source_commit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", image_reference(self.image))
        config_env_name(self.config_env)
        if self.deploy.name != DEPLOY_BUNDLE:
            raise ReleaseError(f"{self.deploy.name!r} is not the deploy bundle")
        if self.runtime_config.name != RUNTIME_CONFIG_BUNDLE:
            raise ReleaseError(
                f"{self.runtime_config.name!r} is not the runtime-config bundle"
            )
        selections = [
            f"{project}={config}"
            for project, config in self.doppler.items()
        ]
        object.__setattr__(self, "doppler", doppler_environments(selections))
        if self.source_commit is not None and not _COMMIT.match(
            self.source_commit
        ):
            raise ReleaseError(
                f"{self.source_commit!r} is not a git commit id"
            )

    @property
    def identity(self) -> dict[str, Any]:
        """The tuple itself: what two identical deployments agree on.

        Timestamps and the source-commit hint stay out, so re-recording the
        same selection twice yields the same release id.
        """
        return {
            "schema": SCHEMA,
            "image": self.image,
            "config_env": self.config_env,
            "deploy_revision": self.deploy.revision,
            "runtime_config_revision": self.runtime_config.revision,
            "doppler": dict(self.doppler),
        }

    @property
    def release_id(self) -> str:
        return hashlib.sha256(_canonical(self.identity).encode()).hexdigest()

    @property
    def short_id(self) -> str:
        return self.release_id[:12]

    @classmethod
    def from_checkout(
        cls,
        root: Path,
        *,
        image: str,
        doppler: Mapping[str, str],
        config_env: str = DEFAULT_CONFIG_ENV,
        source_commit: str | None = None,
        recorded_at: str | None = None,
    ) -> ReleaseRecord:
        # Before globbing, so a bad env name reports itself rather than an
        # empty bundle pattern.
        config_env_name(config_env)
        return cls(
            image=image,
            config_env=config_env,
            deploy=Bundle.from_root(
                DEPLOY_BUNDLE, root, DEPLOY_BUNDLE_PATTERNS
            ),
            runtime_config=Bundle.from_root(
                RUNTIME_CONFIG_BUNDLE,
                root,
                RUNTIME_CONFIG_BUNDLE_PATTERNS,
                env=config_env,
            ),
            doppler=dict(doppler),
            recorded_at=recorded_at or _now(),
            source_commit=source_commit,
        )

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema": SCHEMA,
            "release_id": self.release_id,
            "recorded_at": self.recorded_at,
            "image": self.image,
            "config_env": self.config_env,
            "deploy": {
                "revision": self.deploy.revision,
                "files": dict(self.deploy.files),
            },
            "runtime_config": {
                "revision": self.runtime_config.revision,
                "files": dict(self.runtime_config.files),
            },
            "doppler": dict(self.doppler),
        }
        if self.source_commit is not None:
            document["source_commit"] = self.source_commit
        return document

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ReleaseRecord:
        """Load a record, refusing anything it cannot vouch for.

        Every stored digest is recomputed from the content beside it, so a
        record edited to claim a different revision — or a different release —
        is rejected rather than selected.
        """
        if not isinstance(document, Mapping):
            raise ReleaseError("a release record is a JSON object")
        known = {
            "schema",
            "release_id",
            "recorded_at",
            "image",
            "config_env",
            "deploy",
            "runtime_config",
            "doppler",
            "source_commit",
        }
        unknown = sorted(set(document) - known)
        if unknown:
            raise ReleaseError(f"unknown release fields: {', '.join(unknown)}")
        if document.get("schema") != SCHEMA:
            raise ReleaseError(
                f"unsupported release schema {document.get('schema')!r}"
            )
        missing = sorted(known - {"source_commit"} - set(document))
        if missing:
            raise ReleaseError(
                f"incomplete release: missing {', '.join(missing)}"
            )
        record = cls(
            image=str(document["image"]),
            config_env=str(document["config_env"]),
            deploy=_bundle_from_dict(DEPLOY_BUNDLE, document["deploy"]),
            runtime_config=_bundle_from_dict(
                RUNTIME_CONFIG_BUNDLE, document["runtime_config"]
            ),
            doppler=dict(document["doppler"]),
            recorded_at=str(document["recorded_at"]),
            source_commit=(
                str(document["source_commit"])
                if document.get("source_commit") is not None
                else None
            ),
        )
        stored = str(document["release_id"])
        if not _RELEASE_ID.match(stored) or stored != record.release_id:
            raise ReleaseError(
                f"release id {stored!r} does not match the recorded tuple"
            )
        return record

    @classmethod
    def from_json(cls, text: str) -> ReleaseRecord:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ReleaseError(f"release record is not valid JSON: {exc}")
        return cls.from_dict(document)

    def verify(self, root: Path) -> list[str]:
        """Differences between this release and a checkout; empty means match.

        Rollback is a selection, so the checkout has to hold the recorded
        bundle content before the release can be selected again.
        """
        observed_deploy = Bundle.from_root(
            DEPLOY_BUNDLE, root, DEPLOY_BUNDLE_PATTERNS
        )
        observed_runtime = Bundle.from_root(
            RUNTIME_CONFIG_BUNDLE,
            root,
            RUNTIME_CONFIG_BUNDLE_PATTERNS,
            env=self.config_env,
        )
        differences = self.deploy.drift(observed_deploy)
        differences.extend(self.runtime_config.drift(observed_runtime))
        return differences

    def selection(self) -> dict[str, str]:
        """The Compose interpolation values this release selects."""
        return {
            IMAGE_VARIABLE: self.image,
            RELEASE_ID_VARIABLE: self.release_id,
        }

    def render_selection(self) -> str:
        """The env file systemd loads before every Compose invocation."""
        lines = [
            "# Generated by `openloop release select` — do not edit.",
            f"# release      {self.release_id}",
            f"# recorded at  {self.recorded_at}",
            f"# config env   {self.config_env}",
            f"# deploy       {self.deploy.revision}",
            f"# runtime cfg  {self.runtime_config.revision}",
        ]
        if self.source_commit is not None:
            lines.append(f"# source       {self.source_commit}")
        lines.extend(
            f"# doppler      {project}={config}"
            for project, config in self.doppler.items()
        )
        lines.extend(
            f"{name}={value}" for name, value in self.selection().items()
        )
        return "\n".join(lines) + "\n"


def _bundle_from_dict(name: str, document: Any) -> Bundle:
    if not isinstance(document, Mapping):
        raise ReleaseError(f"{name} bundle is not an object")
    unknown = sorted(set(document) - {"revision", "files"})
    if unknown:
        raise ReleaseError(
            f"unknown {name} bundle fields: {', '.join(unknown)}"
        )
    files = document.get("files")
    if not isinstance(files, Mapping):
        raise ReleaseError(f"{name} bundle has no file digests")
    bundle = Bundle(
        name=name,
        files={str(path): str(digest) for path, digest in files.items()},
    )
    if document.get("revision") != bundle.revision:
        raise ReleaseError(
            f"{name} revision {document.get('revision')!r} does not match "
            "the file digests recorded with it"
        )
    return bundle


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
