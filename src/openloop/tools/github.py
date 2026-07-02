"""Native GitHub connector.

Exposes permissioned actions (`issues:read`, `issues:write`, `pulls:read`,
`pulls:write`) over a small :class:`GitHubClient` interface so the REST calls can
be faked in tests. Use a fine-grained, least-privilege token (see Security in the
README); `pulls:write` additionally needs `contents:write` to push the branch.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openloop.credentials import CredentialResolver, CredentialScope
from openloop.tools.base import ActionSpec, ToolResult

_REPO = {"type": "string", "description": "owner/repo, e.g. acme/ingestion"}
_NUMBER = {"type": "integer", "description": "issue or PR number"}


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


class GitHubConnector:
    """Maps permissioned actions onto a :class:`GitHubClient`."""

    name = "github"

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def supported_permissions(self) -> set[str]:
        return {"issues:read", "issues:write", "pulls:read", "pulls:write"}

    def describe(self, permission: str) -> ActionSpec:
        if permission == "issues:write":
            return ActionSpec(
                "Create a new GitHub issue in a repository.",
                {
                    "type": "object",
                    "properties": {
                        "repo": _REPO,
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["repo", "title"],
                },
            )
        if permission == "issues:read":
            return ActionSpec(
                "Read a GitHub issue by number.",
                {
                    "type": "object",
                    "properties": {"repo": _REPO, "number": _NUMBER},
                    "required": ["repo", "number"],
                },
            )
        if permission == "pulls:write":
            return ActionSpec(
                "Open a GitHub pull request from an existing pushed branch.",
                {
                    "type": "object",
                    "properties": {
                        "repo": _REPO,
                        "head": {
                            "type": "string",
                            "description": "branch with the changes",
                        },
                        "base": {
                            "type": "string",
                            "description": "branch to merge into",
                        },
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "draft": {"type": "boolean"},
                    },
                    "required": ["repo", "head", "title"],
                },
            )
        # pulls:read
        return ActionSpec(
            "Read a GitHub pull request by number.",
            {
                "type": "object",
                "properties": {"repo": _REPO, "number": _NUMBER},
                "required": ["repo", "number"],
            },
        )

    async def execute(self, permission: str, args: dict) -> ToolResult:
        if permission == "issues:write":
            issue = await self.client.create_issue(
                args["repo"], args["title"], args.get("body", "")
            )
            return ToolResult(
                ok=True,
                summary=f"created issue #{issue.get('number')} in {args['repo']}",
                data=issue,
            )
        if permission == "issues:read":
            issue = await self.client.get_issue(args["repo"], int(args["number"]))
            return ToolResult(
                ok=True,
                summary=f"read issue #{args['number']} in {args['repo']}",
                data=issue,
            )
        if permission == "pulls:read":
            pull = await self.client.get_pull(args["repo"], int(args["number"]))
            return ToolResult(
                ok=True,
                summary=f"read PR #{args['number']} in {args['repo']}",
                data=pull,
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
                data=pull,
            )
        return ToolResult(ok=False, summary=f"unsupported permission {permission}")
