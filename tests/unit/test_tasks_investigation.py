from pathlib import Path

import pytest

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
    with pytest.raises(Exception):
        InvestigateArgs(repo="a/b", question="")


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
