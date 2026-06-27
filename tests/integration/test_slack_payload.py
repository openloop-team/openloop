"""Validate the button handler against a realistic Slack block_actions payload.

The Bolt action handler is thin glue:

    approver = _approver_handle(body.get("user", {}))
    await resolve_from_action(runtime.tools, action["value"], approver, ...)

These tests feed it a real-shaped `block_actions` body so the payload
assumptions (where the user identity and button value live) are locked in, and
the approver-matching behavior — including its failure mode — is explicit.
"""

import copy

from openloop.agents import load_agent
from openloop.surfaces.approvals import APPROVE_ACTION, DENY_ACTION
from openloop.surfaces.slack import _approver_handle
from openloop.surfaces.approvals import resolve_from_action
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.testing import EXAMPLE_AGENT, FakeGitHub

# A realistic Slack interactive payload for an Approve button click.
BLOCK_ACTIONS = {
    "type": "block_actions",
    "user": {"id": "U07ABC123", "username": "priya", "name": "priya",
             "team_id": "T01"},
    "api_app_id": "A01",
    "container": {"type": "message", "message_ts": "1700000000.000100"},
    "trigger_id": "123.456.abc",
    "channel": {"id": "C01DEV", "name": "dev-platform"},
    "actions": [
        {
            "action_id": APPROVE_ACTION,
            "block_id": "approval:PLACEHOLDER",
            "type": "button",
            "text": {"type": "plain_text", "text": "Approve"},
            "value": "PLACEHOLDER",
            "action_ts": "1700000000.000200",
        }
    ],
}


async def _gateway_with_pending():
    agent = load_agent(EXAMPLE_AGENT)
    github = FakeGitHub()
    gw = ToolGateway(tools=[GitHubConnector(github)])
    inv = await gw.invoke(
        agent, "github.issues:write", {"repo": "acme/x", "title": "T"}
    )
    return gw, github, inv.approval.id


def _payload(approval_id: str, action_id: str = APPROVE_ACTION, **user):
    body = copy.deepcopy(BLOCK_ACTIONS)
    body["actions"][0]["action_id"] = action_id
    body["actions"][0]["value"] = approval_id
    if user:
        body["user"] = user
    return body


def test_approver_handle_from_block_actions_user():
    # block_actions carries username — maps to the configured "@priya" approver.
    assert _approver_handle(BLOCK_ACTIONS["user"]) == "@priya"


def test_approver_handle_falls_back_when_username_absent():
    assert _approver_handle({"id": "U1", "name": "bob"}) == "@bob"
    # Only an id available: this WON'T match handle-based approvers — the known
    # gap. Documented here so a future id->handle mapping has a failing anchor.
    assert _approver_handle({"id": "U1"}) == "@U1"


async def test_approve_payload_executes_via_handler_path():
    gw, github, approval_id = await _gateway_with_pending()
    body = _payload(approval_id)

    # Exactly what the Bolt handler does with the payload.
    approver = _approver_handle(body.get("user", {}))
    action = body["actions"][0]
    msg = await resolve_from_action(gw, action["value"], approver, approve=True)

    assert msg.startswith("✅ Approved by @priya")
    assert github.created  # real execution on approval


async def test_deny_payload_does_not_execute():
    gw, github, approval_id = await _gateway_with_pending()
    body = _payload(approval_id, action_id=DENY_ACTION)

    approver = _approver_handle(body["user"])
    msg = await resolve_from_action(
        gw, body["actions"][0]["value"], approver, approve=False
    )
    assert msg.startswith("🚫 Denied")
    assert github.created == []


async def test_id_only_user_is_rejected_as_non_approver():
    gw, github, approval_id = await _gateway_with_pending()
    body = _payload(approval_id, id="U07ABC123")  # no username/name

    approver = _approver_handle(body["user"])
    msg = await resolve_from_action(gw, body["actions"][0]["value"], approver,
                                    approve=True)
    assert msg.startswith("⛔")  # @U07ABC123 is not in the approver list
    assert github.created == []
