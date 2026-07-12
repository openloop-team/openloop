"""Phase 4 provisioner seam: source scoping, tagging, and byte budgets."""

from datetime import datetime, timezone

import pytest

from openloop.analysis import (
    GithubInput,
    GithubProvisioner,
    InMemoryInputStore,
    InMemoryUploadStore,
    InputFile,
    InputManifest,
    ProvisionError,
    RequestIdentity,
    StagedInput,
    StagedProvisioner,
    UploadInput,
    UploadProvisioner,
    UploadRecord,
    provision_inputs,
)
from openloop.credentials import EnvCredentialResolver
from openloop.surfaces.slack_files import SlackUploadFetcher
from openloop.tools.github import HttpGitHubClient

pytestmark = pytest.mark.unit


class _UploadFetcher:
    def __init__(self, body: bytes = b"upload") -> None:
        self.body = body
        self.calls: list[tuple[str, int]] = []

    async def fetch(self, upload_ref: str, *, max_bytes: int) -> bytes:
        self.calls.append((upload_ref, max_bytes))
        if len(self.body) > max_bytes:
            raise ProvisionError("upload cap exceeded")
        return self.body


class _TarballClient:
    def __init__(self, body: bytes = b"tarball") -> None:
        self.body = body
        self.calls: list[tuple[str, str | None, int]] = []

    async def get_tarball(
        self, repo: str, ref: str | None, *, max_bytes: int
    ) -> bytes:
        self.calls.append((repo, ref, max_bytes))
        if len(self.body) > max_bytes:
            raise ProvisionError("archive cap exceeded")
        return self.body


class _StaticProvisioner:
    def __init__(self, source: str, files: tuple[InputFile, ...]) -> None:
        self.source = source
        self.files = files
        self.calls: list[int] = []

    async def provision(self, spec, identity, *, budget_bytes: int):
        self.calls.append(budget_bytes)
        return self.files


class _StreamingResponse:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class _StreamingClient:
    chunks = (b"123", b"456")
    calls: list[tuple[str, str, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def stream(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return _StreamingResponse(self.chunks)


def _upload(scope_key: str = "slack\x1fws\x1fa\x1fC1\x1fT1") -> UploadRecord:
    return UploadRecord(
        upload_ref="F1",
        scope_key=scope_key,
        name="reports/data.csv",
        size=6,
        user="U1",
        shared_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
    )


async def test_staged_ref_is_a_job_agnostic_capability_and_respects_budget():
    store = InMemoryInputStore()
    await store.stage(
        InputManifest(
            input_ref="staged:secret",
            files=(InputFile("data.csv", b"1234"),),
        )
    )
    provisioner = StagedProvisioner(store)

    files = await provisioner.provision(
        StagedInput(source="staged", input_ref="staged:secret"),
        RequestIdentity(job_id="an-unrelated-job"),
        budget_bytes=4,
    )

    assert files == (InputFile("data.csv", b"1234"),)
    with pytest.raises(ProvisionError, match="merged input budget"):
        await provisioner.provision(
            StagedInput(source="staged", input_ref="staged:secret"),
            RequestIdentity(job_id="another-job"),
            budget_bytes=3,
        )


async def test_upload_scope_is_rechecked_post_approval_before_fetch():
    uploads = InMemoryUploadStore()
    await uploads.record(_upload())
    fetcher = _UploadFetcher()
    provisioner = UploadProvisioner(uploads, fetcher, max_bytes=10)
    spec = UploadInput(source="upload", upload_ref="F1")

    for scope in (None, "slack\x1fws\x1fa\x1fC2\x1fT1"):
        with pytest.raises(ProvisionError, match="requesting conversation thread"):
            await provisioner.provision(
                spec,
                RequestIdentity(job_id="j", scope_key=scope),
                budget_bytes=10,
            )

    assert fetcher.calls == []


async def test_upload_uses_smallest_source_or_remaining_cap_and_safe_basename():
    scope = "slack\x1fws\x1fa\x1fC1\x1fT1"
    uploads = InMemoryUploadStore()
    await uploads.record(_upload(scope))
    fetcher = _UploadFetcher(b"12345")
    provisioner = UploadProvisioner(uploads, fetcher, max_bytes=7)

    files = await provisioner.provision(
        UploadInput(source="upload", upload_ref="F1"),
        RequestIdentity(job_id="j", scope_key=scope),
        budget_bytes=5,
    )

    assert fetcher.calls == [("F1", 5)]
    assert files == (InputFile("data.csv", b"12345"),)


async def test_github_provisioner_forwards_ref_and_remaining_cap():
    client = _TarballClient(b"abc")
    provisioner = GithubProvisioner(client, max_bytes=20)

    files = await provisioner.provision(
        GithubInput(source="github", repo="acme/ingestion", ref="release/v1"),
        RequestIdentity(job_id="j"),
        budget_bytes=9,
    )

    assert client.calls == [("acme/ingestion", "release/v1", 9)]
    assert files == (InputFile("ingestion.tar", b"abc"),)


async def test_merged_manifest_tags_collisions_and_decrements_before_next_fetch():
    first = _StaticProvisioner("upload", (InputFile("data.csv", b"123"),))
    second = _StaticProvisioner("github", (InputFile("data.csv", b"45"),))
    specs = [
        UploadInput(source="upload", upload_ref="F1"),
        GithubInput(source="github", repo="acme/data"),
    ]

    files = await provision_inputs(
        {"upload": first, "github": second},
        specs,
        RequestIdentity(job_id="j"),
        max_total_bytes=5,
    )

    assert first.calls == [5]
    assert second.calls == [2]
    assert [file.name for file in files] == [
        "upload__data.csv",
        "github__data.csv",
    ]


async def test_no_later_fetch_starts_once_the_merged_budget_is_spent():
    first = _StaticProvisioner("staged", (InputFile("one", b"123"),))
    second = _StaticProvisioner("github", (InputFile("two", b"x"),))

    with pytest.raises(ProvisionError, match="merged cap"):
        await provision_inputs(
            {"staged": first, "github": second},
            [
                StagedInput(source="staged", input_ref="s"),
                GithubInput(source="github", repo="acme/x"),
            ],
            RequestIdentity(job_id="j"),
            max_total_bytes=3,
        )

    assert first.calls == [3]
    assert second.calls == []


async def test_unknown_source_and_empty_provisioning_fail_cleanly():
    spec = StagedInput(source="staged", input_ref="s")
    with pytest.raises(ProvisionError, match="no provisioner"):
        await provision_inputs(
            {}, [spec], RequestIdentity(job_id="j"), max_total_bytes=10
        )

    empty = _StaticProvisioner("staged", ())
    with pytest.raises(ProvisionError, match="produced no input files"):
        await provision_inputs(
            {"staged": empty},
            [spec],
            RequestIdentity(job_id="j"),
            max_total_bytes=10,
        )


async def test_github_archive_stream_is_capped_before_an_oversized_chunk_is_kept(
    monkeypatch,
):
    import httpx

    _StreamingClient.calls.clear()
    monkeypatch.setattr(httpx, "AsyncClient", _StreamingClient)
    client = HttpGitHubClient(EnvCredentialResolver({"github": "token"}))

    with pytest.raises(ProvisionError, match="exceeds the 5-byte cap"):
        await client.get_tarball("acme/repo", "main", max_bytes=5)

    method, url, kwargs = _StreamingClient.calls[0]
    assert method == "GET" and url.endswith("/repos/acme/repo/tarball/main")
    assert kwargs["headers"]["Authorization"] == "Bearer token"


async def test_slack_upload_stream_is_capped_and_uses_the_private_file_api(
    monkeypatch,
):
    import httpx
    from slack_sdk.web import async_client

    class _SlackClient:
        def __init__(self, *, token: str) -> None:
            assert token == "xoxb-secret"

        async def files_info(self, *, file: str):
            assert file == "F1"
            return {"file": {"url_private_download": "https://files.invalid/F1"}}

    _StreamingClient.calls.clear()
    monkeypatch.setattr(httpx, "AsyncClient", _StreamingClient)
    monkeypatch.setattr(async_client, "AsyncWebClient", _SlackClient)

    with pytest.raises(ProvisionError, match="exceeds the 5-byte upload cap"):
        await SlackUploadFetcher("xoxb-secret").fetch("F1", max_bytes=5)

    _, url, kwargs = _StreamingClient.calls[0]
    assert url == "https://files.invalid/F1"
    assert kwargs["headers"]["Authorization"] == "Bearer xoxb-secret"
