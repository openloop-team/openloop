"""Unit tests for SlackSurfaceDelivery's idempotency-key dedup (Finding #1).

Closes the post-succeeded-but-id-lost window: each post is tagged with a
deterministic key in Slack `metadata`, and a recovery post looks the thread up by
that key so a crashed first attempt isn't duplicated. The happy path never scans;
a missing history scope degrades to posting fresh.
"""

import pytest

from openloop.approvals.store import ApprovalRequest
from openloop.deliverable import Artifact
from openloop.sessions import SlackSurfaceDelivery, SurfaceTarget
from openloop.surfaces.slack_format import MARKDOWN_TEXT_LIMIT

pytestmark = pytest.mark.unit


class FakeSlackClient:
    """Minimal AsyncWebClient stand-in: records posts, serves metadata back."""

    def __init__(
        self, *, lookup_error: bool = False, upload_error: bool = False
    ) -> None:
        self.posted: list[dict] = []
        self.statuses: list[dict] = []
        self.uploads: list[dict] = []
        self.lookups = 0
        self.lookup_error = lookup_error
        self.upload_error = upload_error
        self._seq = 0

    async def chat_postMessage(
        self, *, channel, thread_ts=None, text=None, markdown_text=None,
        blocks=None, metadata=None, unfurl_links=None, unfurl_media=None,
    ):
        self._seq += 1
        ts = f"{self._seq}.0001"
        self.posted.append(
            {
                "channel": channel,
                "thread_ts": thread_ts,
                "text": text,
                "markdown_text": markdown_text,
                "blocks": blocks,
                "metadata": metadata,
                "unfurl_links": unfurl_links,
                "unfurl_media": unfurl_media,
                "ts": ts,
            }
        )
        return {"ts": ts}

    async def files_upload_v2(
        self, *, content, filename, title=None, snippet_type=None,
        channel=None, thread_ts=None, **kwargs
    ):
        if self.upload_error:
            raise RuntimeError("missing files:write scope")
        self.uploads.append(
            {
                "content": content,
                "filename": filename,
                "title": title,
                "snippet_type": snippet_type,
                "channel": channel,
                "thread_ts": thread_ts,
            }
        )
        fid = f"F{len(self.uploads)}"
        return {"ok": True, "files": [{"id": fid, "permalink": f"https://files.slack/{fid}"}]}

    async def assistant_threads_setStatus(
        self, *, channel_id, thread_ts, status, loading_messages=None, **kwargs
    ):
        if self.lookup_error:
            raise RuntimeError("missing assistant scope")
        self.statuses.append(
            {
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "status": status,
                "loading_messages": loading_messages,
            }
        )
        return {"ok": True}

    async def conversations_replies(
        self, *, channel, ts, include_all_metadata=False, limit=200
    ):
        self.lookups += 1
        if self.lookup_error:
            raise RuntimeError("missing channels:history scope")
        msgs = [
            {"ts": p["ts"], "metadata": p["metadata"], "text": p["text"]}
            for p in self.posted
            if p["channel"] == channel and (p["thread_ts"] == ts or p["ts"] == ts)
        ]
        return {"messages": msgs}

    async def conversations_history(
        self, *, channel, include_all_metadata=False, limit=200
    ):
        self.lookups += 1
        if self.lookup_error:
            raise RuntimeError("missing channels:history scope")
        msgs = [
            {"ts": p["ts"], "metadata": p["metadata"]}
            for p in self.posted
            if p["channel"] == channel
        ]
        return {"messages": msgs}


class ApiCallOnlySlackClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def api_call(self, api_method, *, json=None, **kwargs):
        self.calls.append({"api_method": api_method, "json": json})
        return {"ok": True}


def _target(thread="100.1"):
    return SurfaceTarget(
        surface="slack", workspace="w", agent="a", channel="C1", thread=thread
    )


async def test_tagged_post_records_metadata_without_scanning():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    ts = await delivery.post_final(_target(), "answer", key="s1:final")

    # Happy path: the message is tagged for later recovery, but no lookup runs.
    assert client.lookups == 0
    assert ts == client.posted[0]["ts"]
    md = client.posted[0]["metadata"]
    assert md["event_type"] == "openloop_delivery"
    assert md["event_payload"]["key"] == "s1:final"


async def test_progress_status_uses_assistant_thread_indicator():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.set_progress_status(_target(), "is thinking...")

    assert client.statuses == [
        {
            "channel_id": "C1",
            "thread_ts": "100.1",
            "status": "is thinking...",
            "loading_messages": ["is thinking..."],
        }
    ]
    assert client.posted == []


async def test_progress_status_fallback_sets_loading_messages():
    client = ApiCallOnlySlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.set_progress_status(_target(), "is thinking...")

    assert client.calls == [
        {
            "api_method": "assistant.threads.setStatus",
            "json": {
                "channel_id": "C1",
                "thread_ts": "100.1",
                "status": "is thinking...",
                "loading_messages": ["is thinking..."],
            },
        }
    ]


async def test_progress_status_failure_is_non_blocking():
    client = FakeSlackClient(lookup_error=True)
    delivery = SlackSurfaceDelivery(client)

    await delivery.set_progress_status(_target(), "is thinking...")

    assert client.statuses == []


async def test_recover_returns_existing_message_instead_of_duplicating():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    first = await delivery.post_final(_target(), "answer", key="s1:final")
    # The crash-retry path finds the tagged message and returns its id.
    again = await delivery.post_final(
        _target(), "answer", key="s1:final", recover=True
    )

    assert again == first
    assert len(client.posted) == 1  # no duplicate
    assert client.lookups == 1


async def test_recover_at_channel_root_uses_history():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)
    target = _target(thread=None)

    first = await delivery.post_final(target, "answer", key="k")
    again = await delivery.post_final(target, "answer", key="k", recover=True)

    assert again == first
    assert len(client.posted) == 1


async def test_recover_posts_fresh_when_no_prior_message():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    ts = await delivery.post_final(_target(), "answer", key="s1:final", recover=True)

    assert len(client.posted) == 1
    assert ts == client.posted[0]["ts"]


async def test_lookup_failure_degrades_to_posting():
    # Missing history scope / transient error must not crash delivery — post fresh
    # and fall back to at-least-once rather than dropping the answer.
    client = FakeSlackClient(lookup_error=True)
    delivery = SlackSurfaceDelivery(client)

    ts = await delivery.post_error(_target(), "boom", key="s1:error", recover=True)

    assert len(client.posted) == 1
    assert ts == client.posted[0]["ts"]


# --- Markdown rendering: the agent writes standard Markdown; Slack's `text`
# renders it literally, so plain posts must go out as `markdown_text` and only
# Block Kit or oversize posts fall back to the classic fields.


def _approval_request() -> ApprovalRequest:
    return ApprovalRequest(
        agent="a", action="github.issues:write", tool="github",
        permission="write", args={}, approvers=["@maciag.artur"],
        summary="create issue",
    )


async def test_final_answer_posts_as_markdown_text():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.post_final(_target(), "**bold** and a [link](https://x)")

    post = client.posted[0]
    assert post["markdown_text"] == "**bold** and a [link](https://x)"
    assert post["text"] is None  # mutually exclusive with markdown_text
    assert post["blocks"] is None


async def test_error_posts_as_markdown_text():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.post_error(_target(), "⚠️ something broke")

    assert client.posted[0]["markdown_text"] == "⚠️ something broke"
    assert client.posted[0]["text"] is None


async def test_approval_card_keeps_text_and_blocks():
    # Block Kit is mutually exclusive with markdown_text: approval cards keep
    # `text` as the notification fallback and render via their mrkdwn blocks.
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.post_approval(_target(), "Approval required", [_approval_request()])

    post = client.posted[0]
    assert post["markdown_text"] is None
    assert post["text"] == "Approval required"
    assert post["blocks"]  # section + buttons


async def test_broadcast_pings_are_neutralized_user_mentions_kept():
    # Model-echoed `<!channel>` must never mass-ping (prompt-injection vector);
    # user mentions are legitimate references and pass through.
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.post_final(
        _target(), "cc <@U07ABC123> — done. <!channel> <!here|@here>"
    )

    assert client.posted[0]["markdown_text"] == (
        "cc <@U07ABC123> — done. @channel @here"
    )


# --- Artifact rendering: bulk content becomes a hosted snippet plus a keyed
# summary message; the message stays the idempotency anchor.


def _artifact() -> Artifact:
    return Artifact(
        content="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new",
        title="Proposed change",
        filename="change.diff",
        summary="Applied the rename in `x.py`.",
        snippet_type="diff",
    )


async def test_artifact_uploads_snippet_and_posts_keyed_summary():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.post_final(_target(), _artifact(), key="s1:final")

    assert client.uploads == [
        {
            "content": _artifact().content,
            "filename": "change.diff",
            "title": "Proposed change",
            "snippet_type": "diff",
            # Shared into the thread at upload time — permalink-unfurl sharing
            # alone would leave the file invisible to other members if the
            # unfurl doesn't fire.
            "channel": "C1",
            "thread_ts": "100.1",
        }
    ]
    post = client.posted[0]
    # The summary is prose only — the file is already shared as its own message,
    # so no permalink rides the caption (a link would unfurl into a duplicate).
    assert post["markdown_text"] == "Applied the rename in `x.py`."
    assert "files.slack" not in post["markdown_text"]
    assert post["text"] is None
    # Unfurling stays off so any link in the prose summary itself can't render a
    # second card either.
    assert post["unfurl_links"] is False
    assert post["unfurl_media"] is False
    # The summary message, not the file, carries the delivery key.
    assert post["metadata"]["event_payload"]["key"] == "s1:final"


async def test_oversize_final_becomes_hosted_snippet():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)
    huge = "Intro paragraph.\n\n" + "x" * (MARKDOWN_TEXT_LIMIT + 1)

    await delivery.post_final(_target(), huge)

    assert len(client.uploads) == 1
    assert client.uploads[0]["content"] == huge  # nothing lost to truncation
    assert client.uploads[0]["channel"] == "C1"
    assert client.uploads[0]["thread_ts"] == "100.1"
    post = client.posted[0]
    assert post["markdown_text"].startswith("Intro paragraph.")
    # Caption only — no permalink to unfurl into a duplicate of the shared file.
    assert "files.slack" not in post["markdown_text"]
    assert post["unfurl_links"] is False
    assert post["unfurl_media"] is False


async def test_upload_failure_degrades_to_inline_text():
    # Losing the snippet host must not lose the answer: content lands inline as
    # plain text (formatting sacrificed, delivery kept).
    client = FakeSlackClient(upload_error=True)
    delivery = SlackSurfaceDelivery(client)

    await delivery.post_final(_target(), _artifact(), key="s1:final")

    post = client.posted[0]
    assert post["markdown_text"] is None
    assert "Applied the rename" in post["text"]
    assert "+new" in post["text"]
    assert post["metadata"]["event_payload"]["key"] == "s1:final"


async def test_upload_failure_fallback_sanitizes_content():
    # The inline fallback renders as mrkdwn where `<!channel>` is a live
    # broadcast — artifact *content* (logs can carry attacker-influenced text)
    # must be neutralized on this path, not just the summary.
    client = FakeSlackClient(upload_error=True)
    delivery = SlackSurfaceDelivery(client)
    artifact = Artifact(
        content="ERROR at line 3: <!channel> deploy failed <!here>",
        title="Log", filename="run.log", summary="Run failed — log below.",
    )

    await delivery.post_final(_target(), artifact)

    body = client.posted[0]["text"]
    assert "<!channel>" not in body
    assert "<!here>" not in body
    assert "@channel deploy failed @here" in body


async def test_artifact_recovery_skips_reupload():
    # Crash-retry: the keyed summary message is found first, so neither a second
    # file nor a second message is created.
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    first = await delivery.post_final(_target(), _artifact(), key="s1:final")
    again = await delivery.post_final(
        _target(), _artifact(), key="s1:final", recover=True
    )

    assert again == first
    assert len(client.uploads) == 1
    assert len(client.posted) == 1


async def test_unkeyed_posts_never_dedupe():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    a = await delivery.post_final(_target(), "x")
    b = await delivery.post_final(_target(), "x")

    assert a != b
    assert len(client.posted) == 2
    assert client.posted[0]["metadata"] is None
