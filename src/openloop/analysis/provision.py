"""The Phase 4 provisioner seam: per-source input materialization.

A :class:`Provisioner` turns one parsed ``inputs[]`` entry into input files.
The seam is called from exactly one place — the sealed analysis orchestrator,
after the approval resolves and the monthly ledger gate passes, before
materialization — because provisioning is real work (a credentialed network
fetch) and must never run for a deniable request. It is pre-model-call, so a
crash-resume that re-provisions is safe: attempt accounting governs the model
spend, not the fetch.

Every failure raises :class:`ProvisionError` with sanitized copy (no token
material, no raw URLs) — the orchestrator maps it onto a terminal failure
after attempt reconciliation, preserving settle-known-spend-first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from openloop.analysis.inputs import GithubInput, StagedInput, UploadInput
from openloop.analysis.store import InputFile, InputStore
from openloop.analysis.uploads import UploadStore

logger = logging.getLogger(__name__)


class ProvisionError(RuntimeError):
    """A terminal provisioning failure; the message is surface-safe."""


@dataclass(slots=True, frozen=True)
class RequestIdentity:
    """The gateway-stamped identity a provisioner may scope against.

    ``scope_key`` is the full thread-ownership tuple key stamped from the
    session context; ``None`` means the request came from a scopeless path
    (the direct tools API) and scope-bound sources must refuse.
    """

    job_id: str
    scope_key: str | None = None
    agent: str | None = None


@runtime_checkable
class Provisioner(Protocol):
    """Materializes one ``inputs[]`` entry of its ``source`` into files."""

    source: str

    async def provision(
        self, spec, identity: RequestIdentity, *, budget_bytes: int
    ) -> tuple[InputFile, ...]: ...


class StagedProvisioner:
    """The operator-staged path, re-expressed through the seam.

    ``input_ref`` is a capability token: the job-agnostic store lookup is the
    authorization, so no job binding survives here (a caller-passable job_id
    used to be a soft hole — the model-facing schema merely didn't advertise
    it).
    """

    source = "staged"

    def __init__(self, inputs: InputStore) -> None:
        self.inputs = inputs

    async def provision(
        self, spec: StagedInput, identity: RequestIdentity, *, budget_bytes: int
    ) -> tuple[InputFile, ...]:
        manifest = await self.inputs.get(spec.input_ref)
        if manifest is None:
            raise ProvisionError("no staged input matches this input_ref")
        total = sum(len(file.content) for file in manifest.files)
        if total > budget_bytes:
            raise ProvisionError(
                f"staged input is {total} bytes; only {budget_bytes} bytes of "
                "the merged input budget remain"
            )
        return manifest.files


@runtime_checkable
class UploadFetcher(Protocol):
    """Fetches one shared file's bytes from the surface, capped in flight.

    Implementations must enforce ``max_bytes`` DURING the download (an
    oversized file must not buy unbounded controller memory before failing)
    and raise :class:`ProvisionError` with sanitized copy on any failure.
    """

    async def fetch(self, upload_ref: str, *, max_bytes: int) -> bytes: ...


class UploadProvisioner:
    """Surface uploads, fetched lazily — bytes leave the surface only here,
    post-approval."""

    source = "upload"

    def __init__(
        self,
        uploads: UploadStore,
        fetcher: UploadFetcher,
        *,
        max_bytes: int,
    ) -> None:
        self.uploads = uploads
        self.fetcher = fetcher
        self.max_bytes = max_bytes

    async def provision(
        self, spec: UploadInput, identity: RequestIdentity, *, budget_bytes: int
    ) -> tuple[InputFile, ...]:
        record = await self.uploads.get(spec.upload_ref)
        # Scope is RE-CHECKED here even though invoke-time resolution already
        # verified it: the durable record must not become a scope bypass if
        # store state changed between invoke and approval (TOCTOU). Unknown
        # ref and wrong scope share one message — existence across threads
        # must not leak.
        if (
            record is None
            or identity.scope_key is None
            or record.scope_key != identity.scope_key
        ):
            raise ProvisionError(
                "this upload was not shared in the requesting conversation "
                "thread"
            )
        cap = min(budget_bytes, self.max_bytes)
        data = await self.fetcher.fetch(spec.upload_ref, max_bytes=cap)
        return (InputFile(_bare_name(record.name, spec.upload_ref), data),)


@runtime_checkable
class TarballClient(Protocol):
    """The one GitHub call this seam needs; the full client satisfies it."""

    async def get_tarball(
        self, repo: str, ref: str | None, *, max_bytes: int
    ) -> bytes: ...


class GithubProvisioner:
    """A repository archive, delivered as one ``<name>.tar`` input file.

    The same convention as the staging CLI's ``--archive``: the tarball rides
    the flat-filename input contract and the generated program extracts it
    in-sandbox (stdlib ``tarfile`` reads gzip transparently) into tmpfs.
    """

    source = "github"

    def __init__(self, client: TarballClient, *, max_bytes: int) -> None:
        self.client = client
        self.max_bytes = max_bytes

    async def provision(
        self, spec: GithubInput, identity: RequestIdentity, *, budget_bytes: int
    ) -> tuple[InputFile, ...]:
        cap = min(budget_bytes, self.max_bytes)
        data = await self.client.get_tarball(spec.repo, spec.ref, max_bytes=cap)
        name = spec.repo.rsplit("/", 1)[-1] or "repository"
        return (InputFile(f"{name}.tar", data),)


def _bare_name(name: str, fallback: str) -> str:
    """A surface-supplied filename reduced to one safe path component."""
    bare = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not bare or bare in (".", ".."):
        return fallback
    return bare


async def provision_inputs(
    provisioners: dict[str, Provisioner],
    inputs,
    identity: RequestIdentity,
    *,
    max_total_bytes: int,
) -> tuple[InputFile, ...]:
    """Materialize every parsed input entry under one merged byte budget.

    The cap is a decrementing budget checked BEFORE each provisioner is
    invoked and passed in as its remaining allowance — once spent, no further
    fetch starts. (A post-hoc check would let one approved request retain up
    to ``len(inputs)`` individually-capped payloads in controller memory
    before failing.) Filenames are tagged per entry so the flat-filename
    manifest contract holds across sources.
    """
    remaining = max_total_bytes
    files: list[InputFile] = []
    used_names: set[str] = set()
    for index, spec in enumerate(inputs, start=1):
        provisioner = provisioners.get(spec.source)
        if provisioner is None:
            # Cannot happen through the front door (invoke-time resolution
            # refuses unavailable sources); covers stale records and direct
            # callers.
            raise ProvisionError(
                f"no provisioner is configured for source {spec.source!r}"
            )
        if remaining <= 0:
            raise ProvisionError(
                f"combined inputs exceed the {max_total_bytes}-byte merged cap"
            )
        fetched = await provisioner.provision(
            spec, identity, budget_bytes=remaining
        )
        for file in fetched:
            remaining -= len(file.content)
            if remaining < 0:
                # Belt: a provisioner returned more than its allowance.
                raise ProvisionError(
                    f"combined inputs exceed the {max_total_bytes}-byte "
                    "merged cap"
                )
            files.append(
                InputFile(_tagged(spec.source, index, file.name, used_names), file.content)
            )
    if not files:
        raise ProvisionError("provisioning produced no input files")
    return tuple(files)


def _tagged(source: str, index: int, name: str, used: set[str]) -> str:
    """A per-entry source tag keeping merged filenames flat and unique."""
    tagged = f"{source}__{name}"
    if tagged in used:
        tagged = f"{source}{index}__{name}"
    if tagged in used:
        raise ProvisionError(f"duplicate provisioned filename {name!r}")
    used.add(tagged)
    return tagged
