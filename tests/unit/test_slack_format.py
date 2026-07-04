"""Unit tests for the Slack rendering policy (sanitizer + oversize split).

The sanitizer is a security control: model output can echo user-supplied text,
so `<!channel>`-style broadcasts must be neutralized on every rendered post
while legitimate user mentions survive. The oversize split guards the
`markdown_text` cap without losing content.
"""

import pytest

from openloop.surfaces.slack_format import (
    MARKDOWN_TEXT_LIMIT,
    oversize_to_artifact,
    sanitize,
)

pytestmark = pytest.mark.unit


def test_sanitize_neutralizes_broadcasts():
    assert sanitize("ping <!channel> now") == "ping @channel now"
    assert sanitize("<!here> and <!everyone>") == "@here and @everyone"
    # Labelled and case-variant forms are still broadcasts.
    assert sanitize("<!channel|@channel>") == "@channel"
    assert sanitize("<!HERE>") == "@here"


def test_sanitize_neutralizes_user_groups():
    assert sanitize("cc <!subteam^S012345|@platform-team>") == "cc @platform-team"
    assert sanitize("cc <!subteam^S012345>") == "cc @group"


def test_sanitize_preserves_user_mentions_and_plain_text():
    text = "ask <@U07ABC123> about a < b and `<!channel>` in prose"
    # The user mention survives; the fenced literal is still neutralized —
    # over-flagging inside code spans is the safe direction for a ping guard.
    out = sanitize(text)
    assert "<@U07ABC123>" in out
    assert "<!channel>" not in out


def test_sanitize_is_noop_on_clean_markdown():
    text = "**bold**, a [link](https://x), and `code`"
    assert sanitize(text) == text


def test_oversize_split_keeps_full_content_and_paragraph_preview():
    text = "First paragraph.\n\nSecond paragraph.\n\n" + "x" * MARKDOWN_TEXT_LIMIT
    artifact = oversize_to_artifact(text)

    assert artifact.content == text  # the snippet holds everything
    assert artifact.summary.startswith("First paragraph.")
    assert "attached" in artifact.summary
    # The preview cuts on a paragraph boundary, not mid-sentence.
    assert not artifact.summary.startswith(text[:999] + "x")
    assert len(artifact.summary) < 1_100
