from pathlib import Path

from openloop.agents import load_agent
from openloop.tools.gateway import ACTION_ALIASES, _canonical_action, _summarize
from openloop.tools.policy import is_allowed

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def test_legacy_coding_worker_action_aliases_to_workspace_task():
    assert ACTION_ALIASES["coding_worker.pr:write"] == "workspace_task.code:write"
    assert _canonical_action("coding_worker.pr:write") == "workspace_task.code:write"


def test_unknown_action_is_returned_unchanged():
    assert _canonical_action("github.issues:write") == "github.issues:write"


# --- Finding 1: allowlist check must compare in canonical space ------------


def test_legacy_yaml_agent_is_allowed_both_legacy_and_canonical_action():
    # tests/unit/data/agent.yaml declares the LEGACY action
    # "coding_worker.pr:write" (mirrors agents/dev-platform.yaml) and is not
    # migrated as part of this task. Both spellings must be allowed.
    agent = load_agent(AGENT_YAML)
    assert is_allowed(agent, "coding_worker.pr:write")
    assert is_allowed(agent, "workspace_task.code:write")


# --- Finding 2: the approval-card summary must key on the canonical action -


def test_summarize_produces_friendly_text_for_canonical_action():
    summary = _summarize(
        "workspace_task.code:write",
        {"repo": "acme/x", "instruction": "fix the thing"},
    )
    assert "draft PR" in summary
    assert "acme/x" in summary
    assert summary != str({"repo": "acme/x", "instruction": "fix the thing"})
