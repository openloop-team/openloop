from openloop.tools.gateway import ACTION_ALIASES, _canonical_action


def test_legacy_coding_worker_action_aliases_to_workspace_task():
    assert ACTION_ALIASES["coding_worker.pr:write"] == "workspace_task.code:write"
    assert _canonical_action("coding_worker.pr:write") == "workspace_task.code:write"


def test_unknown_action_is_returned_unchanged():
    assert _canonical_action("github.issues:write") == "github.issues:write"
