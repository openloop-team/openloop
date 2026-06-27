"""Live end-to-end test for the coding worker — gated on credentials.

Exercises the REAL chain: GitCodingWorker clones the repo, applies a diff,
commits and pushes a branch, then the connector opens a REAL *draft* PR. The PR
is closed and the branch deleted afterward, so the test is safe to re-run / use
in CI.

To keep the run deterministic (no flaky model output) the model call is stubbed
with a fixed new-file diff; everything else — git, push, the GitHub PR API — is
real.

Runs only when enabled; skips cleanly otherwise:
  E2E_LIVE=1
  GITHUB_TOKEN (needs contents:write + pull-requests:write), E2E_GITHUB_REPO=owner/repo
"""

import os
import uuid

import pytest

from openloop.models.gateway import ModelResponse
from openloop.tools.coding_worker import CodingWorkerConnector, GitCodingWorker
from openloop.tools.github import HttpGitHubClient


def _missing() -> str | None:
    if os.environ.get("E2E_LIVE") != "1":
        return "set E2E_LIVE=1 to run the live end-to-end test"
    for var in ("GITHUB_TOKEN", "E2E_GITHUB_REPO"):
        if not os.environ.get(var):
            return f"{var} not set"
    return None


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.live,
    pytest.mark.skipif(_missing() is not None, reason=_missing() or ""),
]


class _StubCompleter:
    """Returns a fixed new-file diff so the run is deterministic."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    async def complete(self, model, messages, **kwargs):
        path = f"openloop-e2e-{self.marker}.md"
        diff = (
            f"diff --git a/{path} b/{path}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{path}\n"
            "@@ -0,0 +1 @@\n"
            f"+openloop coding-worker e2e {self.marker}\n"
        )
        text = (
            f"TITLE: [openloop e2e] coding worker {self.marker}\n"
            "BODY: Automated end-to-end check; safe to close.\n"
            f"DIFF:\n{diff}"
        )
        return ModelResponse(text=text, model="stub")


async def test_coding_worker_live_draft_pr():
    repo = os.environ["E2E_GITHUB_REPO"]
    token = os.environ["GITHUB_TOKEN"]
    base = os.environ.get("E2E_GITHUB_BASE", "main")
    marker = uuid.uuid4().hex[:8]

    client = HttpGitHubClient(token)
    worker = GitCodingWorker(token, model="stub", gateway=_StubCompleter(marker))
    connector = CodingWorkerConnector(worker, client)

    args = connector.prepare_args(
        "pr:write",
        {"repo": repo, "instruction": f"add a marker file {marker}", "base": base},
    )
    job_id = args["job_id"]
    branch = f"openloop/job-{job_id}"

    result = await connector.execute("pr:write", args)

    pr_number = (result.data or {}).get("pr_number")
    try:
        assert result.ok, f"worker failed: {result.summary}"
        assert result.data["job_id"] == job_id
        assert result.data["branch"] == branch
        assert pr_number, "no PR opened"

        pull = await client.get_pull(repo, pr_number)
        assert pull["draft"] is True
        assert pull["head"]["ref"] == branch
    finally:
        if pr_number:
            await client._request(
                "PATCH", f"/repos/{repo}/pulls/{pr_number}", json={"state": "closed"}
            )
        # Best-effort branch cleanup so runs don't accumulate refs.
        try:
            await client._request(
                "DELETE", f"/repos/{repo}/git/refs/heads/{branch}"
            )
        except Exception:
            pass
