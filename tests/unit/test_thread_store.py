"""Unit tests for the Phase A ThreadRecord store (delivered-transcript lane).

Covers idempotent append (a redelivery never duplicates a fragment), oldest-first
ordered replay bounded by a limit, thread-scope isolation, and exclude-self.
"""

from openloop.sessions import (
    InMemoryThreadRecordStore,
    SurfaceTarget,
    TranscriptFragment,
    thread_scope_key,
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


# --- inbox + active-turn claim (Phase C) ---


async def test_inbox_enqueue_dedup_and_ordered_drain():
    store = InMemoryThreadRecordStore()
    scope = _scope()
    assert await store.append_inbox(scope, "e1", {"text": "one"}) is True
    assert await store.append_inbox(scope, "e2", {"text": "two"}) is True
    # A re-delivered event that is still pending does not enqueue twice.
    assert await store.append_inbox(scope, "e1", {"text": "one"}) is False

    assert await store.try_begin_turn(scope) is True
    drained = []
    while (item := await store.next_inbox(scope)) is not None:
        drained.append(item.payload["text"])
    assert drained == ["one", "two"]  # oldest-first


async def test_try_begin_turn_is_exclusive():
    store = InMemoryThreadRecordStore()
    scope = _scope()
    await store.append_inbox(scope, "e1", {"text": "one"})

    assert await store.try_begin_turn(scope) is True   # first handler wins
    assert await store.try_begin_turn(scope) is False  # second is refused

    await store.end_turn(scope)
    # After release, if work remains it can be re-claimed (drain-race guard).
    await store.append_inbox(scope, "e2", {"text": "two"})
    assert await store.try_begin_turn(scope) is True


async def test_try_begin_turn_false_on_empty_inbox():
    # A free thread with nothing queued is not claimed — so the drain loop's
    # re-claim after release can't spin on an empty inbox.
    store = InMemoryThreadRecordStore()
    scope = _scope()
    assert await store.try_begin_turn(scope) is False


async def test_claim_is_per_thread_scope():
    store = InMemoryThreadRecordStore()
    a, b = _scope(thread="100.1"), _scope(thread="200.2")
    await store.append_inbox(a, "e1", {"text": "a"})
    await store.append_inbox(b, "e1", {"text": "b"})

    assert await store.try_begin_turn(a) is True
    # A different thread scope is independently claimable while `a` is held.
    assert await store.try_begin_turn(b) is True


async def test_reset_active_claims_unwedges_a_crashed_leader():
    # A drain leader that crashed left its claim held; a startup reset clears it so
    # the thread is claimable again (its queued work is still pending).
    store = InMemoryThreadRecordStore()
    scope = _scope()
    await store.append_inbox(scope, "e1", {"text": "one"})
    assert await store.try_begin_turn(scope) is True   # "leader" claims...
    # ...then "crashes" (never releases). A second claim is refused.
    assert await store.try_begin_turn(scope) is False

    assert await store.reset_active_claims() == 1
    assert await store.try_begin_turn(scope) is True   # claimable again


# --- warm-context handle (Phase B) ---


async def test_context_ref_set_get_and_clear():
    store = InMemoryThreadRecordStore()
    key = thread_scope_key(_scope())

    assert await store.get_context_ref(key) is None
    await store.set_context_ref(key, "handle-1")
    assert await store.get_context_ref(key) == "handle-1"
    # Clearing (a warm context evicted) drops the handle.
    await store.set_context_ref(key, None)
    assert await store.get_context_ref(key) is None


async def test_context_ref_is_per_scope_key():
    store = InMemoryThreadRecordStore()
    a = thread_scope_key(_scope(thread="100.1"))
    b = thread_scope_key(_scope(thread="200.2"))
    await store.set_context_ref(a, "handle-a")

    assert await store.get_context_ref(a) == "handle-a"
    assert await store.get_context_ref(b) is None


async def test_get_or_create_reflects_context_ref():
    store = InMemoryThreadRecordStore()
    scope = _scope()
    await store.set_context_ref(thread_scope_key(scope), "handle-x")

    record = await store.get_or_create(scope)
    assert record.context_ref == "handle-x"
