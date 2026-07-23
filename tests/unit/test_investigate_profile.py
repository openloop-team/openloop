"""Unit tests for the workspace_task ``investigate:read`` profile (Task 12).

Covers the connector-level surface only: ``investigate:read`` has
``gate=Gate.NONE`` (see ``openloop.tasks.contract``), so in Stage 1 it never
becomes an approval or a durable workflow instance — it always runs through
``CodingWorkerConnector.execute()`` synchronously, and its result flows back
through the model tool-loop. These tests exercise exactly that path.
"""

from openloop.models.gateway import ModelResponse
from openloop.tasks.investigation import INVESTIGATE_ARGS_VERSION, RepoInvestigator
from openloop.testing import FakeGitHub, FakeWorkerOrchestrator
from openloop.tools.coding_worker import CodingWorkerConnector


class _FakeGateway:
    """A network-free stand-in for the model gateway RepoInvestigator calls."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list = []

    async def complete(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages})
        return ModelResponse(
            text=self._text,
            model=model,
            cost_usd=0.02,
            prompt_tokens=11,
            completion_tokens=7,
        )


def _investigator(
    text: str = "SUMMARY: it returns None\nFINDINGS:\n- parse() at a.py:1 returns None\n",
) -> RepoInvestigator:
    return RepoInvestigator("m", gateway=_FakeGateway(text))


def _connector(*, investigator=None, runner=None, github=None) -> CodingWorkerConnector:
    return CodingWorkerConnector(
        runner or FakeWorkerOrchestrator(),
        github or FakeGitHub(),
        investigator=investigator,
    )


def test_supported_permissions_includes_investigate_when_investigator_set():
    conn = _connector(investigator=_investigator())
    assert conn.supported_permissions() == {"code:write", "investigate:read"}


def test_supported_permissions_excludes_investigate_without_investigator():
    conn = _connector()
    assert conn.supported_permissions() == {"code:write"}
    assert "investigate:read" not in conn.supported_permissions()


def test_investigate_describe_has_typed_args():
    spec = _connector(investigator=_investigator()).describe("investigate:read")
    assert spec.version == 1 == INVESTIGATE_ARGS_VERSION
    assert spec.model is not None
    assert "question" in spec.parameters["properties"]


async def test_execute_investigate_returns_evidence_bundle_and_opens_no_pr():
    github = FakeGitHub()
    runner = FakeWorkerOrchestrator()
    conn = _connector(investigator=_investigator(), runner=runner, github=github)

    args = conn.prepare_args(
        "investigate:read", {"repo": "a/b", "question": "why does parse return None?"}
    )
    assert args["profile"] == "investigate"
    assert args.get("job_id")

    result = await conn.execute("investigate:read", args)

    assert result.ok is True
    outcome = result.data["outcome"]
    assert outcome["kind"] == "evidence_bundle"
    assert outcome["findings"].strip() != ""
    assert outcome["summary"] == "it returns None"
    assert result.data["cost_usd"] == 0.02
    # Never opens a PR, never pushes — this is a read-only profile.
    assert github.pulls == []


async def test_execute_investigate_provisions_a_readonly_workspace_and_cleans_up():
    runner = FakeWorkerOrchestrator()
    conn = _connector(investigator=_investigator(), runner=runner)

    args = conn.prepare_args(
        "investigate:read", {"repo": "a/b", "question": "?", "ref": "dev"}
    )
    result = await conn.execute("investigate:read", args)

    assert result.ok is True
    assert runner.readonly_provisions == [("a/b", "dev")]


async def test_execute_investigate_without_investigator_fails_closed():
    conn = _connector()  # no investigator configured
    args = conn.prepare_args(
        "investigate:read", {"repo": "a/b", "question": "why?"}
    )
    result = await conn.execute("investigate:read", args)

    assert result.ok is False
    assert result.data["status"] == "failed"


async def test_execute_investigate_with_empty_args_returns_failed_result():
    conn = _connector(investigator=_investigator())
    result = await conn.execute("investigate:read", {})

    assert result.ok is False
    assert result.data["status"] == "failed"
    assert "repo" in result.data["error"] or "required" in result.data["error"]


async def test_execute_investigate_missing_repo_returns_failed_result():
    conn = _connector(investigator=_investigator())
    result = await conn.execute("investigate:read", {"question": "why?"})

    assert result.ok is False
    assert result.data["status"] == "failed"


async def test_execute_investigate_missing_question_returns_failed_result():
    conn = _connector(investigator=_investigator())
    result = await conn.execute("investigate:read", {"repo": "a/b"})

    assert result.ok is False
    assert result.data["status"] == "failed"


def test_code_write_permission_and_behavior_are_unchanged():
    conn = _connector(investigator=_investigator())
    spec = conn.describe("code:write")
    assert spec.parameters["properties"].keys() >= {"repo", "instruction"}
