"""Slack rendering policy — pure text transforms for the Slack renderer.

Kept separate from the delivery client (like ``approvals.py`` keeps Block Kit
out of the Bolt wiring) so the policy is unit-testable without a Slack app:
:func:`sanitize` neutralizes broadcast pings in model-authored text, and
:func:`oversize_to_artifact` converts prose past Slack's ``markdown_text`` cap
into a snippet-plus-summary deliverable.
"""

from __future__ import annotations

import re

from openloop.deliverable import Artifact

# Slack's cap on chat.postMessage(markdown_text=...). Prose beyond it becomes a
# hosted snippet rather than degrading to an unformatted wall of text.
MARKDOWN_TEXT_LIMIT = 12_000

# Ceiling for the last-resort inline plain-text post (snippet upload failed).
# Slack rejects chat.postMessage text past 40k; stay well under it.
FALLBACK_TEXT_LIMIT = 30_000

# How much of an oversize answer to keep inline as the summary message. Cut at
# a paragraph boundary where possible so the preview reads as prose, not as a
# mid-sentence truncation.
_PREVIEW_LIMIT = 1_000

# Broadcast pings a model must never be able to trigger: user-supplied text
# containing `<!channel>` etc. would otherwise be echoed by the model and
# mass-ping on post (prompt-injection vector). User mentions `<@U...>` are
# deliberately preserved — the task text keeps other users' mentions, and the
# model may legitimately reference them. Plain "@channel" text (without the
# `<!...>` encoding) does not ping when posted via the API.
_BROADCAST_RE = re.compile(r"<!(channel|here|everyone)(?:\|[^>]*)?>", re.IGNORECASE)
_SUBTEAM_RE = re.compile(r"<!subteam\^[A-Z0-9]+(?:\|([^>]*))?>")


def sanitize(text: str) -> str:
    """Neutralize broadcast/user-group pings in model-authored text."""
    text = _BROADCAST_RE.sub(lambda m: f"@{m.group(1).lower()}", text)
    return _SUBTEAM_RE.sub(lambda m: m.group(1) or "@group", text)


def oversize_to_artifact(text: str) -> Artifact:
    """Turn prose past :data:`MARKDOWN_TEXT_LIMIT` into a snippet deliverable.

    The leading paragraphs stay inline as the summary; the full answer becomes
    the hosted snippet, so nothing is lost to truncation.
    """
    preview = text[:_PREVIEW_LIMIT]
    cut = preview.rfind("\n\n")
    if cut > 0:
        preview = preview[:cut]
    return Artifact(
        content=text,
        title="Full response",
        filename="response.md",
        summary=preview + "\n\n_(full response attached)_",
        snippet_type="markdown",
    )
