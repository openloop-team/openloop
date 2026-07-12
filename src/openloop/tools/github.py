"""Native GitHub connector.

Exposes permissioned actions (`issues:read`, `issues:write`, `pulls:read`,
`pulls:write`) over a small :class:`GitHubClient` interface so the REST calls can
be faked in tests. Use a fine-grained, least-privilege token (see Security in the
README); `pulls:write` additionally needs `contents:write` to push the branch.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openloop.analysis.provision import ProvisionError
from openloop.credentials import CredentialResolver, CredentialScope
from openloop.tools.base import ActionSpec, ToolResult

GITHUB_ARGS_VERSION = 1

_REPO_DESC = "owner/repo, e.g. acme/ingestion"


def _stripped(value):
    return value.strip() if isinstance(value, str) else value


class _GithubArgs(BaseModel):
    """Typed args base (typed-tool-args §3): schemas are generated from these
    models and the gateway parses raw args through them."""

    model_config = ConfigDict(extra="forbid")


class IssueWriteArgs(_GithubArgs):
    repo: str = Field(min_length=1, description=_REPO_DESC)
    title: str = Field(min_length=1)
    body: str = ""

    _strip = field_validator("repo", "title", mode="before")(_stripped)


class IssueReadArgs(_GithubArgs):
    repo: str = Field(min_length=1, description=_REPO_DESC)
    # strict so a string like "5" is a type error, not silently coerced —
    # preserving the pre-typed-args reject-wrong-type contract.
    number: int = Field(strict=True, description="issue or PR number")

    _strip = field_validator("repo", mode="before")(_stripped)


class PullWriteArgs(_GithubArgs):
    repo: str = Field(min_length=1, description=_REPO_DESC)
    head: str = Field(description="branch with the changes")
    base: str | None = Field(default=None, description="branch to merge into")
    title: str
    body: str = ""
    draft: bool = True

    _strip = field_validator("repo", mode="before")(_stripped)


class PullReadArgs(_GithubArgs):
    repo: str = Field(min_length=1, description=_REPO_DESC)
    # strict so a string like "5" is a type error, not silently coerced —
    # preserving the pre-typed-args reject-wrong-type contract.
    number: int = Field(strict=True, description="issue or PR number")

    _strip = field_validator("repo", mode="before")(_stripped)

# ToolResult.data goes back to the model verbatim, so trim GitHub's verbose
# payloads (nested user/reactions/_links objects) to what the model needs.
_ISSUE_FIELDS = (
    "number", "title", "state", "html_url", "user", "labels", "assignees",
    "created_at", "updated_at", "comments", "body",
)
_PULL_FIELDS = _ISSUE_FIELDS + ("draft", "merged", "merged_at", "head", "base")
_BODY_MAX_CHARS = 2000


def _trim(data: dict, fields: tuple[str, ...]) -> dict:
    out = {k: data[k] for k in fields if k in data}
    if isinstance(out.get("user"), dict):
        out["user"] = out["user"].get("login")
    if isinstance(out.get("labels"), list):
        out["labels"] = [
            label.get("name") if isinstance(label, dict) else label
            for label in out["labels"]
        ]
    if isinstance(out.get("assignees"), list):
        out["assignees"] = [
            a.get("login") if isinstance(a, dict) else a for a in out["assignees"]
        ]
    for ref in ("head", "base"):
        if isinstance(out.get(ref), dict):
            out[ref] = out[ref].get("ref")
    body = out.get("body")
    if isinstance(body, str) and len(body) > _BODY_MAX_CHARS:
        out["body"] = body[:_BODY_MAX_CHARS] + "… [truncated]"
    return out


def _trim_issue(data: dict) -> dict:
    return _trim(data, _ISSUE_FIELDS)


def _trim_pull(data: dict) -> dict:
    return _trim(data, _PULL_FIELDS)


@runtime_checkable
class GitHubClient(Protocol):
    async def create_issue(self, repo: str, title: str, body: str) -> dict: ...

    async def get_issue(self, repo: str, number: int) -> dict: ...

    async def get_pull(self, repo: str, number: int) -> dict: ...

    async def create_pull(
        self,
        repo: str,
        head: str,
        base: str,
        title: str,
        body: str,
        draft: bool = True,
    ) -> dict: ...

    async def find_pull(self, repo: str, head: str) -> dict | None: ...

    async def get_tarball(
        self, repo: str, ref: str | None, *, max_bytes: int
    ) -> bytes: ...


class HttpGitHubClient:
    """Thin httpx-backed client against the GitHub REST API.

    Auth flows through the :class:`CredentialResolver` seam **at request time**
    — the client never stores a raw token — so the backend (env token, GitHub
    App installation tokens, secrets manager) is swappable without touching
    this class.
    """

    def __init__(
        self,
        credentials: CredentialResolver,
        base_url: str = "https://api.github.com",
        *,
        scope: CredentialScope | None = None,
    ) -> None:
        self._credentials = credentials
        self._scope = scope or CredentialScope(integration="github")
        self.base_url = base_url.rstrip("/")

    async def _headers(self) -> dict[str, str]:
        token = await self._credentials.resolve(self._scope)
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        import httpx

        headers = await self._headers()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(
                method, f"{self.base_url}{path}", headers=headers, **kwargs
            )
            resp.raise_for_status()
            return resp.json()

    async def create_issue(self, repo: str, title: str, body: str) -> dict:
        return await self._request(
            "POST", f"/repos/{repo}/issues", json={"title": title, "body": body}
        )

    async def get_issue(self, repo: str, number: int) -> dict:
        return await self._request("GET", f"/repos/{repo}/issues/{number}")

    async def get_pull(self, repo: str, number: int) -> dict:
        return await self._request("GET", f"/repos/{repo}/pulls/{number}")

    async def create_pull(
        self,
        repo: str,
        head: str,
        base: str,
        title: str,
        body: str,
        draft: bool = True,
    ) -> dict:
        return await self._request(
            "POST",
            f"/repos/{repo}/pulls",
            json={
                "head": head,
                "base": base,
                "title": title,
                "body": body,
                "draft": draft,
            },
        )

    async def find_pull(self, repo: str, head: str) -> dict | None:
        owner = repo.split("/", 1)[0]
        pulls = await self._request(
            "GET",
            f"/repos/{repo}/pulls",
            params={"head": f"{owner}:{head}", "state": "all"},
        )
        return pulls[0] if pulls else None

    async def get_tarball(
        self, repo: str, ref: str | None, *, max_bytes: int
    ) -> bytes:
        """Stream a repository archive, capped IN FLIGHT at ``max_bytes``.

        The cap fires during the download — a huge repo must not buy unbounded
        controller memory before failing. Failure copy is sanitized for
        approval-adjacent surfaces: the repo name is fine (it came from args),
        token material and raw URLs are not.
        """
        import httpx

        path = f"/repos/{repo}/tarball" + (f"/{ref}" if ref else "")
        headers = await self._headers()
        received = bytearray()
        try:
            # GitHub answers with a redirect to codeload; httpx must follow it.
            async with httpx.AsyncClient(
                timeout=60, follow_redirects=True
            ) as client:
                async with client.stream(
                    "GET", f"{self.base_url}{path}", headers=headers
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if len(received) + len(chunk) > max_bytes:
                            raise ProvisionError(
                                f"repository archive of {repo} exceeds the "
                                f"{max_bytes}-byte cap"
                            )
                        received.extend(chunk)
        except ProvisionError:
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise ProvisionError(
                f"GitHub returned {status} for the {repo} archive "
                "(unknown repo/ref, or the credential lacks access)"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProvisionError(
                f"fetching the {repo} archive failed: {type(exc).__name__}"
            ) from exc
        return bytes(received)


class GitHubConnector:
    """Maps permissioned actions onto a :class:`GitHubClient`."""

    name = "github"

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def supported_permissions(self) -> set[str]:
        return {"issues:read", "issues:write", "pulls:read", "pulls:write"}

    def describe(self, permission: str) -> ActionSpec:
        # Schemas are GENERATED from the typed args models the gateway parses
        # with, so declaration and enforcement cannot drift.
        if permission == "issues:write":
            return _spec(
                "Create a new GitHub issue in a repository.", IssueWriteArgs
            )
        if permission == "issues:read":
            return _spec("Read a GitHub issue by number.", IssueReadArgs)
        if permission == "pulls:write":
            return _spec(
                "Open a GitHub pull request from an existing pushed branch.",
                PullWriteArgs,
            )
        # pulls:read
        return _spec("Read a GitHub pull request by number.", PullReadArgs)

    async def execute(self, permission: str, args: dict) -> ToolResult:
        if permission == "issues:write":
            issue = await self.client.create_issue(
                args["repo"], args["title"], args.get("body", "")
            )
            return ToolResult(
                ok=True,
                summary=f"created issue #{issue.get('number')} in {args['repo']}",
                data=_trim_issue(issue),
            )
        if permission == "issues:read":
            issue = await self.client.get_issue(args["repo"], int(args["number"]))
            return ToolResult(
                ok=True,
                summary=f"read issue #{args['number']} in {args['repo']}",
                data=_trim_issue(issue),
            )
        if permission == "pulls:read":
            pull = await self.client.get_pull(args["repo"], int(args["number"]))
            return ToolResult(
                ok=True,
                summary=f"read PR #{args['number']} in {args['repo']}",
                data=_trim_pull(pull),
            )
        if permission == "pulls:write":
            pull = await self.client.create_pull(
                args["repo"],
                args["head"],
                args.get("base", "main"),
                args["title"],
                args.get("body", ""),
                bool(args.get("draft", True)),
            )
            return ToolResult(
                ok=True,
                summary=f"opened PR #{pull.get('number')} in {args['repo']}",
                data=_trim_pull(pull),
            )
        return ToolResult(ok=False, summary=f"unsupported permission {permission}")


def _spec(description: str, model: type[_GithubArgs]) -> ActionSpec:
    return ActionSpec(
        description,
        model.model_json_schema(),
        model=model,
        version=GITHUB_ARGS_VERSION,
    )
