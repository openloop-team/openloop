"""Surface-agnostic approval UI helpers (Slack Block Kit + resolution).

Kept separate from the Bolt wiring so the rendering and resolution logic can be
unit-tested without constructing a Slack app or talking to Slack.
"""

from __future__ import annotations

from openloop.approvals.store import ApprovalRequest
from openloop.tools import ToolGateway

APPROVE_ACTION = "openloop_approve"
DENY_ACTION = "openloop_deny"
OPENHANDS_ACCEPT_ACTION = "openloop_openhands_accept"
OPENHANDS_REJECT_ACTION = "openloop_openhands_reject"


def approval_blocks(requests: list[ApprovalRequest]) -> list[dict]:
    """Block Kit for one or more pending write actions, each with buttons."""
    blocks: list[dict] = []
    for req in requests:
        approvers = ", ".join(req.approvers) or "any approver"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⏳ *Approval required:* {req.summary}\n_{approvers}_",
                },
            }
        )
        blocks.append(
            {
                "type": "actions",
                "block_id": f"approval:{req.id}",
                "elements": [
                    {
                        "type": "button",
                        "action_id": APPROVE_ACTION,
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "value": req.id,
                    },
                    {
                        "type": "button",
                        "action_id": DENY_ACTION,
                        "text": {"type": "plain_text", "text": "Deny"},
                        "style": "danger",
                        "value": req.id,
                    },
                ],
            }
        )
    return blocks


def openhands_decision_blocks(
    job_id: str, decision_id: str, summary: str
) -> list[dict]:
    """Explicit accept/reject controls for one parked OpenHands action."""
    value = f"{job_id}|{decision_id}"
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"⏸️ *Confirmation needed:* {summary}",
            },
        },
        {
            "type": "actions",
            "block_id": f"openhands:{decision_id}",
            "elements": [
                {
                    "type": "button",
                    "action_id": OPENHANDS_ACCEPT_ACTION,
                    "text": {"type": "plain_text", "text": "Accept"},
                    "style": "primary",
                    "value": value,
                },
                {
                    "type": "button",
                    "action_id": OPENHANDS_REJECT_ACTION,
                    "text": {"type": "plain_text", "text": "Reject"},
                    "style": "danger",
                    "value": value,
                },
            ],
        },
    ]


def resolution_message(inv, approver: str) -> str:
    """Status line for a resolved approval — shared by the button reply and the
    session continuation so they never drift."""
    if inv.status == "executed":
        detail = inv.result.summary if inv.result else (inv.message or "done")
        return f"✅ Approved by {approver} — {detail}"
    if inv.status == "started":
        detail = inv.result.summary if inv.result else (inv.message or "started")
        return f"✅ Approved by {approver} — {detail}"
    if inv.status == "denied":
        return f"🚫 Denied by {approver}."
    # forbidden (not an approver / unknown / already resolved)
    return f"⛔ {inv.message}"


async def resolve_from_action(
    gateway: ToolGateway, approval_id: str, approver: str, *, approve: bool
) -> str:
    """Resolve an approval from a button click; return a status message."""
    inv = await gateway.resolve(approval_id, approver, approve=approve)
    return resolution_message(inv, approver)
