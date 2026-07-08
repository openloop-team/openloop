"""Unit tests for the Phase A ThreadRecord store (delivered-transcript lane).

Covers idempotent append (a redelivery never duplicates a fragment), oldest-first
ordered replay bounded by a limit, thread-scope isolation, and exclude-self.
"""

from openloop.sessions import (
    InMemoryThreadRecordStore,
    SurfaceTarget,
    TranscriptFragment,
)


def _scope(*, agent="dev-platform", channel="C1", thread="100.1"):
    # event_id is deliberately varied/None — it must NOT affect the thread scope.
    return SurfaceTarget(
        surface="slack", workspace="acme", agent=agent,
        channel=channel, thread=thread, event_id="ignored",
    )


def _frag(turn_id, request="q", answer="a"):
    return TranscriptFragment(turn_id=turn_id, request=request, answer=answer)


async def test_append_then_replay_oldest_first():
    store = InMemoryThreadRecordStore()
    scope = _scope()
    await store.append_delivered_fragment(scope, _frag("t1", "q1", "a1"))
    await store.append_delivered_fragment(scope, _frag("t2", "q2", "a2"))

    out = await store.replayable_transcript(scope)

    assert [(f.request, f.answer) for f in out] == [("q1", "a1"), ("q2", "a2")]


async def test_append_is_idempotent_on_turn_id():
    # A redelivery / reconcile of the same turn must not double-append.
    store = InMemoryThreadRecordStore()
    scope = _scope()
    await store.append_delivered_fragment(scope, _frag("t1", "q1", "a1"))
    await store.append_delivered_fragment(scope, _frag("t1", "q1", "a1-again"))

    out = await store.replayable_transcript(scope)

    assert len(out) == 1
    assert out[0].answer == "a1"  # first (delivered) write wins


async def test_scope_isolation():
    # event_id differs but scope is the same thread → shared transcript; a
    # different channel/agent is a different thread → isolated.
    store = InMemoryThreadRecordStore()
    scope = _scope()
    other_thread = _scope(thread="200.2")
    other_agent = _scope(agent="other")
    await store.append_delivered_fragment(scope, _frag("t1"))

    assert len(await store.replayable_transcript(scope)) == 1
    assert await store.replayable_transcript(other_thread) == []
    assert await store.replayable_transcript(other_agent) == []


async def test_limit_keeps_most_recent():
    store = InMemoryThreadRecordStore()
    scope = _scope()
    for i in range(5):
        await store.append_delivered_fragment(scope, _frag(f"t{i}", f"q{i}", f"a{i}"))

    out = await store.replayable_transcript(scope, limit=2)

    # Most recent two, still oldest-first.
    assert [f.request for f in out] == ["q3", "q4"]


async def test_exclude_turn_id_drops_the_in_flight_turn():
    store = InMemoryThreadRecordStore()
    scope = _scope()
    await store.append_delivered_fragment(scope, _frag("t1"))
    await store.append_delivered_fragment(scope, _frag("t2"))

    out = await store.replayable_transcript(scope, exclude_turn_id="t2")

    assert [f.turn_id for f in out] == ["t1"]


async def test_get_or_create_is_idempotent():
    store = InMemoryThreadRecordStore()
    scope = _scope()
    r1 = await store.get_or_create(scope)
    r2 = await store.get_or_create(scope)
    assert r1.scope.thread == r2.scope.thread == "100.1"
