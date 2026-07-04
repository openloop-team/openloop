"""Unit: the GitHub connector trims verbose API payloads for the model."""

from openloop.tools.github import GitHubConnector

# A cut-down but structurally faithful GitHub REST issue payload: nested
# user/label/assignee objects plus noise fields the model never needs.
VERBOSE_ISSUE = {
    "number": 7,
    "title": "Widget breaks on empty input",
    "state": "open",
    "html_url": "https://github.com/acme/x/issues/7",
    "user": {"login": "octocat", "id": 1, "avatar_url": "https://…", "type": "User"},
    "labels": [{"id": 11, "name": "bug", "color": "d73a4a"}],
    "assignees": [{"login": "hubot", "id": 2}],
    "created_at": "2026-07-01T10:00:00Z",
    "updated_at": "2026-07-02T10:00:00Z",
    "comments": 3,
    "body": "Steps to reproduce…",
    "node_id": "I_kwDOA",
    "reactions": {"+1": 4, "url": "https://…"},
    "timeline_url": "https://…",
}


class VerboseGitHub:
    async def get_issue(self, repo, number):
        return dict(VERBOSE_ISSUE)

    async def get_pull(self, repo, number):
        return {
            **VERBOSE_ISSUE,
            "draft": True,
            "merged": False,
            "merged_at": None,
            "head": {"ref": "fix/empty-input", "sha": "abc123", "repo": {}},
            "base": {"ref": "main", "sha": "def456", "repo": {}},
            "_links": {"self": {"href": "https://…"}},
        }

    async def create_issue(self, repo, title, body):
        return dict(VERBOSE_ISSUE)


async def test_issue_read_trims_to_model_relevant_fields():
    connector = GitHubConnector(VerboseGitHub())
    result = await connector.execute("issues:read", {"repo": "acme/x", "number": 7})

    assert result.ok
    assert result.data["number"] == 7
    assert result.data["title"] == "Widget breaks on empty input"
    assert result.data["user"] == "octocat"
    assert result.data["labels"] == ["bug"]
    assert result.data["assignees"] == ["hubot"]
    # Noise fields are dropped entirely.
    for noisy in ("node_id", "reactions", "timeline_url"):
        assert noisy not in result.data


async def test_pull_read_trims_refs_and_links():
    connector = GitHubConnector(VerboseGitHub())
    result = await connector.execute("pulls:read", {"repo": "acme/x", "number": 7})

    assert result.data["head"] == "fix/empty-input"
    assert result.data["base"] == "main"
    assert result.data["draft"] is True
    assert "_links" not in result.data


async def test_long_issue_body_is_truncated():
    class LongBodyGitHub(VerboseGitHub):
        async def get_issue(self, repo, number):
            return {**VERBOSE_ISSUE, "body": "y" * 10_000}

    connector = GitHubConnector(LongBodyGitHub())
    result = await connector.execute("issues:read", {"repo": "acme/x", "number": 7})
    assert result.data["body"].endswith("… [truncated]")
    assert len(result.data["body"]) < 2100
