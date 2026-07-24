import pytest
from pydantic import ValidationError

from openloop.tasks.investigation import InvestigateArgs, RepoInvestigator
from openloop.tasks.outcomes import EvidenceBundle


class _FakeGateway:
    def __init__(self, text):
        self._text = text
        self.calls = []

    async def complete(self, model, messages):
        self.calls.append((model, messages))
        from openloop.models.gateway import ModelResponse
        return ModelResponse(
            text=self._text,
            model=model,
            cost_usd=0.01,
            prompt_tokens=10,
            completion_tokens=5,
        )


def test_investigate_args_require_question():
    with pytest.raises(ValidationError):
        InvestigateArgs(repo="a/b", question="")


def test_parse_findings_inline_after_marker():
    """Regression: findings on same line as FINDINGS: marker are captured."""
    from openloop.tasks.investigation import _parse_findings

    # Inline findings on same line as marker (previously dropped silently)
    summary, findings = _parse_findings("SUMMARY: x\nFINDINGS: - only bullet")
    assert summary == "x"
    assert findings == "- only bullet"


def test_parse_findings_multiline_after_marker():
    """Findings on lines following the FINDINGS: marker are captured."""
    from openloop.tasks.investigation import _parse_findings

    # Multi-line findings (existing behavior, regression guard)
    summary, findings = _parse_findings("SUMMARY: one line\nFINDINGS:\n- a\n- b")
    assert summary == "one line"
    assert findings == "- a\n- b"


def test_parse_findings_summary_on_following_line():
    """Summary on a line following the SUMMARY: marker is captured, not dropped."""
    from openloop.tasks.investigation import _parse_findings

    # Regression: previously the empty inline slot made summary fall back to default.
    summary, findings = _parse_findings("SUMMARY:\nfoo\nFINDINGS:\n- a")
    assert summary == "foo"
    assert findings == "- a"


def test_parse_findings_empty_summary_falls_back():
    """A genuinely empty summary falls back to the default, never grabs FINDINGS:."""
    from openloop.tasks.investigation import _parse_findings

    # Guards the FINDINGS: bound: without it, "first non-empty line" would be "FINDINGS:".
    summary, findings = _parse_findings("SUMMARY:\nFINDINGS:\n- a")
    assert summary == "Investigation complete."
    assert findings == "- a"


async def test_investigator_returns_evidence_bundle_from_model_findings(tmp_path):
    (tmp_path / "p.py").write_text("def parse():\n    return None\n")
    gw = _FakeGateway(
        "SUMMARY: parse() can return None\n"
        "FINDINGS:\n- parse() at p.py:1 returns None unconditionally\n"
    )
    inv = RepoInvestigator("anthropic/claude-sonnet-4-6", gateway=gw)
    bundle, resp = await inv.investigate(tmp_path, "why does parse return None?", "a/b")
    assert isinstance(bundle, EvidenceBundle)
    assert "parse()" in bundle.findings
    assert bundle.summary == "parse() can return None"
    assert resp.cost_usd == 0.01
