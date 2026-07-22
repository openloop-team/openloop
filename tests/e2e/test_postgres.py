"""Happy-path end-to-end test against a real Postgres (pgvector).

Validates what the unit tests can't: the actual SQL, asyncpg type handling,
pgvector distance search, and approval persistence. The model, embedder, and
GitHub client are faked (no external credentials), but every store is real.

Runs only when a Postgres is reachable — set OPENLOOP_TEST_DATABASE_URL, or it
falls back to the docker-compose default. Skips cleanly otherwise so the normal
suite stays green without Docker.
"""

from pathlib import Path
import asyncio
import contextlib
import os
import uuid
from dataclasses import replace

import pytest

from openloop.agents import load_agent
from openloop.approvals.postgres import PostgresApprovalStore
from openloop.memory.postgres import PostgresMemoryStore
from openloop.memory.store import MemoryRecord, scope_key_for
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import UsageRecord, budget_scope_key
from openloop.usage.postgres import PostgresUsageStore
from openloop.workflows import WorkflowEngine
from openloop.workflows.postgres import PostgresWorkflowStore
from openloop.testing import (
    FakeEmbedder,
    FakeGitHub,
    ScriptedGateway,
    tool_call_response,
)

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"

DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop_agents",
)

# 26-dim to match FakeEmbedder (the real default is 1536; dim is configurable).
EMBED_DIM = 26

pytestmark = [pytest.mark.e2e, pytest.mark.postgres]


async def _reachable() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(DSN, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def postgres_pool():
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")
    import asyncpg

    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=10)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def stores(postgres_pool):
    # Unique table-free isolation isn't possible (shared tables), so scope keys
    # are made unique per run instead.
    memory = PostgresMemoryStore(embedding_dim=EMBED_DIM)
    usage = PostgresUsageStore()
    approvals = PostgresApprovalStore()
    workflows = PostgresWorkflowStore()
    await memory.setup(postgres_pool)
    await usage.setup(postgres_pool)
    await approvals.setup(postgres_pool)
    await workflows.setup(postgres_pool)
    try:
        yield memory, usage, approvals, workflows
    finally:
        await memory.close()
        await usage.close()
        await approvals.close()
        await workflows.close()


async def test_usage_attribution_envelope_round_trip(stores):
    # Finding-4 envelope must survive a real Postgres INSERT + row mapping, both
    # non-null (broker-run spend) and null (legacy). The broker_generation value
    # is past INT32 max on purpose — it round-trips only because the column is
    # BIGINT (an INTEGER column would reject the insert), pinning that width.
    _memory, usage, _approvals, _workflows = stores
    run_id = uuid.uuid4().hex[:8]
    scope = f"ws:e2e:agent:{run_id}"
    big_generation = 9_000_000_000  # > 2**31 - 1

    assert await usage.record(UsageRecord(
        scope_key=scope, workspace="e2e", agent="dev-platform",
        model="claude-sonnet-5", cost_usd=0.1,
        idempotency_key=f"e2e-env-{run_id}",
        job_id=f"job{run_id}",
        broker_job_id="11111111-2222-3333-4444-555555555555",
        broker_generation=big_generation,
        approval_id=f"apr-{run_id}", approver="alice",
        session_id=f"sess-{run_id}"))
    assert await usage.record(UsageRecord(
        scope_key=scope, workspace="e2e", agent="dev-platform",
        model="gpt-4o-mini", cost_usd=0.001,
        idempotency_key=f"e2e-legacy-{run_id}"))

    rows = {r.idempotency_key: r for r in await usage.recent(limit=500)}
    env = rows[f"e2e-env-{run_id}"]
    assert env.job_id == f"job{run_id}"
    assert env.broker_job_id == "11111111-2222-3333-4444-555555555555"
    assert env.broker_generation == big_generation
    assert env.approval_id == f"apr-{run_id}"
    assert env.approver == "alice"
    assert env.session_id == f"sess-{run_id}"

    legacy = rows[f"e2e-legacy-{run_id}"]
    for value in (
        legacy.job_id, legacy.broker_job_id, legacy.broker_generation,
        legacy.approval_id, legacy.approver, legacy.session_id,
    ):
        assert value is None


async def test_happy_path_end_to_end(stores):
    memory, usage, approvals, workflows = stores
    agent = load_agent(AGENT_YAML)
    run_id = uuid.uuid4().hex[:8]
    channel = f"#e2e-{run_id}"  # unique scope so the run is isolated
    scope = scope_key_for(agent, channel)

    # Seed a prior decision into channel memory (real pgvector insert).
    embedder = FakeEmbedder()
    seed_vec = (await embedder.embed(["Use Redis Streams for ingestion v1."]))[0]
    await memory.remember(MemoryRecord(
        scope_key=scope, text="Use Redis Streams for ingestion v1.",
        embedding=seed_vec))

    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)], approvals=approvals)

    # The model recalls context, then asks to open a GitHub issue (write action).
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_issues_write",
                                  {"repo": "acme/ingestion",
                                   "title": "Track: Redis Streams for v1"})]),
    ])
    runtime = Runtime(
        agent,
        gateway=gateway,
        memory=memory,
        embedder=embedder,
        usage=usage,
        tools=tools,
        engine=WorkflowEngine(workflows),
    )

    # --- the turn: write action is held for approval ---
    result = await runtime.handle(Task(
        text="open an issue to track the ingestion decision",
        surface="slack", channel=channel, user="U_requester"))

    assert result.model == "approval-gate"
    assert len(result.approval_ids) == 1
    approval_id = result.approval_ids[0]

    # Recall worked against pgvector: the seeded memory reached the model.
    system_text = " ".join(
        m["content"] for m in gateway.calls[0]["messages"]
        if m["role"] == "system")
    assert "Redis Streams" in system_text

    # The approval is persisted as pending in Postgres.
    pending = await approvals.pending(agent="dev-platform")
    assert any(p.id == approval_id for p in pending)
    assert github.created == []  # nothing executed yet

    # --- a human approves; the action executes and persists ---
    inv = await tools.resolve(approval_id, "@maciag.artur", approve=True)
    assert inv.status == "executed"
    assert github.created  # the issue was created on approval

    stored = await approvals.get(approval_id)
    assert stored.status == "approved"
    assert stored.decided_by == "@maciag.artur"

    # Usage was recorded to the real audit trail, and the turn was remembered.
    spent_records = await usage.recent(limit=200)
    assert any(r.channel == channel for r in spent_records)
    assert await usage.monthly_total(budget_scope_key(agent)) >= 0.0

    recalled = await memory.recall(scope, seed_vec, limit=5)
    texts = [r.text for r in recalled]
    assert "Use Redis Streams for ingestion v1." in texts
    # The requester's message was remembered this turn.
    assert any("open an issue to track" in t for t in texts)


async def test_worker_checkpoint_resume_across_real_postgres(postgres_pool):
    """A worker job persisted to Postgres resumes on a fresh store instance."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.checkpoints.postgres import PostgresCheckpointStore
    from openloop.tools.coding_worker import (
        STEPS,
        CodingWorkerConnector,
        WorkerOutcome,
    )

    job_id = f"e2e-{uuid.uuid4().hex[:8]}"

    class _Runner:
        def __init__(self):
            self.runs = 0

        async def run_attempt(self, state, on_step=None):
            self.runs += 1
            for step in STEPS:
                state.completed_steps.append(step)
                if on_step is not None:
                    await on_step(state)
            state.title, state.body = "t", "b"
            return WorkerOutcome(branch=state.branch, title="t", body="b")

    class _FlakyGitHub(FakeGitHub):
        def __init__(self):
            super().__init__()
            self.fail_next = True

        async def create_pull(self, *a, **k):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("blip")
            return await super().create_pull(*a, **k)

    store = PostgresCheckpointStore()
    await store.setup(postgres_pool)
    try:
        args = {"repo": "acme/x", "instruction": "do x", "job_id": job_id}

        # First store/worker: pushes, but the PR open fails — persisted to PG.
        runner1, github1 = _Runner(), _FlakyGitHub()
        conn1 = CodingWorkerConnector(runner1, github1, checkpoints=store)
        first = await conn1.execute("pr:write", args)
        assert not first.ok
        cp = await store.get(job_id)
        assert cp.status == "open_pr_failed" and "push" in cp.completed_steps

        # A *fresh* store + connector (simulating a restart) resumes from PG:
        # the worker is not re-run and exactly one PR is opened.
        store2 = PostgresCheckpointStore()
        await store2.setup(postgres_pool)
        try:
            runner2, github2 = _Runner(), FakeGitHub()
            conn2 = CodingWorkerConnector(runner2, github2, checkpoints=store2)
            second = await conn2.execute("pr:write", args)
            assert second.ok
            assert runner2.runs == 0  # resumed past the push
            assert len(github2.pulls) == 1
            assert (await store2.get(job_id)).status == "opened"
        finally:
            await store2.close()
    finally:
        # Best-effort cleanup of this run's row.
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM worker_checkpoints WHERE job_id = $1", job_id
                )
        except Exception:
            pass
        await store.close()


async def test_workflow_resume_across_real_postgres(postgres_pool):
    """A workflow parked at a wait node resumes from Postgres on a fresh engine."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.workflows import Step, Workflow, WorkflowEngine
    from openloop.workflows.postgres import PostgresWorkflowStore

    instance_id = f"wf-{uuid.uuid4().hex[:8]}"

    def _wf():
        async def finish(ctx):
            ctx.instance.result = {"ok": True}
            ctx.state["ran"] = True

        return Workflow("t", [Step("gate", wait=True), Step("finish", finish)])

    store = PostgresWorkflowStore()
    await store.setup(postgres_pool)
    try:
        engine1 = WorkflowEngine(store, {"t": _wf()})
        parked = await engine1.start("t", instance_id, {"seed": 1})
        assert parked.status == "waiting" and parked.waiting_on == "gate"

        # Fresh store + engine (a restart) delivers the event and completes.
        store2 = PostgresWorkflowStore()
        await store2.setup(postgres_pool)
        try:
            engine2 = WorkflowEngine(store2, {"t": _wf()})
            done = await engine2.send_event(instance_id, "gate", {"by": "x"})
            assert done.status == "completed"
            assert done.result == {"ok": True}
            assert done.state["ran"] is True
            assert done.state["seed"] == 1  # original state survived the restart
        finally:
            await store2.close()
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM workflow_instances WHERE id = $1", instance_id
                )
        except Exception:
            pass
        await store.close()


async def test_surface_session_roundtrip_across_real_postgres(postgres_pool):
    """Persist a surface session and look it up by event + approval id (Phase D)."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.sessions.postgres import PostgresSurfaceSessionStore
    from openloop.sessions.store import SurfaceSession, SurfaceTarget

    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    event_id = f"ev-{uuid.uuid4().hex[:8]}"
    approval_id = f"appr-{uuid.uuid4().hex[:8]}"

    store = PostgresSurfaceSessionStore()
    await store.setup(postgres_pool)
    try:
        await store.upsert(SurfaceSession(
            id=session_id,
            target=SurfaceTarget(
                surface="slack", workspace="acme", agent="dev-platform",
                channel="C1", thread="100.1", event_id=event_id,
            ),
            status="waiting",
            workflow_instance_id=session_id,
            progress_message_id="ts-1",
            approval_ids=[approval_id],
            request_text="please do the thing",
            result_artifact_ref="artifact://job-1/report.md",
        ))

        # A fresh store (a restart) reads it back by all three keys.
        store2 = PostgresSurfaceSessionStore()
        await store2.setup(postgres_pool)
        try:
            by_id = await store2.get(session_id)
            assert by_id is not None and by_id.status == "waiting"
            assert by_id.target.thread == "100.1"
            assert by_id.approval_ids == [approval_id]
            assert by_id.request_text == "please do the thing"
            assert by_id.result_artifact_ref == "artifact://job-1/report.md"
            assert (await store2.get_by_event(event_id)).id == session_id
            # The `@>` containment lookup (button → session) resolves the owner.
            assert (await store2.get_by_approval(approval_id)).id == session_id
            assert await store2.get_by_approval("nope") is None
            # The (scope-aware) thread lookup finds the session for this bot...
            in_thread = SurfaceTarget(
                surface="slack", workspace="acme", agent="dev-platform",
                channel="C1", thread="100.1",
            )
            assert (await store2.get_by_thread(in_thread)).id == session_id
            assert await store2.get_by_thread(
                replace(in_thread, thread="other")
            ) is None
            # ...but not a different agent sharing the same channel/thread.
            assert await store2.get_by_thread(
                replace(in_thread, agent="other-agent")
            ) is None
        finally:
            await store2.close()
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM surface_sessions WHERE id = $1", session_id
                )
        except Exception:
            pass
        await store.close()


async def test_thread_history_across_real_postgres(postgres_pool):
    """Rebuild conversation history from prior thread sessions (oldest-first)."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.sessions.postgres import PostgresSurfaceSessionStore
    from openloop.sessions.store import SurfaceSession, SurfaceTarget

    thread = f"th-{uuid.uuid4().hex[:8]}"
    ids = [f"sess-{uuid.uuid4().hex[:8]}" for _ in range(3)]

    def _target(tid, *, agent="dev-platform", thr=thread):
        return SurfaceTarget(
            surface="slack", workspace="acme", agent=agent,
            channel="C1", thread=thr, event_id=f"ev-{tid}",
        )

    store = PostgresSurfaceSessionStore()
    await store.setup(postgres_pool)
    try:
        # Two delivered turns + one still-running, plus an undelivered completed
        # turn (answer never reached the user) and a same-thread session for a
        # *different* agent — both must be excluded from history.
        await store.upsert(SurfaceSession(
            id=ids[0], target=_target(ids[0]), status="completed",
            request_text="q1", result_summary="a1", final_message_id="ts-0",
        ))
        await store.upsert(SurfaceSession(
            id=ids[1], target=_target(ids[1]), status="completed",
            request_text="q2", result_summary="a2", final_message_id="ts-1",
        ))
        await store.upsert(SurfaceSession(
            id=ids[2], target=_target(ids[2]), status="running",
            request_text="q3",
        ))
        undelivered_id = f"sess-{uuid.uuid4().hex[:8]}"
        await store.upsert(SurfaceSession(
            id=undelivered_id, target=_target(undelivered_id), status="completed",
            request_text="never-seen", result_summary="undelivered",
            final_message_id=None,
        ))
        other_agent_id = f"sess-{uuid.uuid4().hex[:8]}"
        await store.upsert(SurfaceSession(
            id=other_agent_id, target=_target(other_agent_id, agent="other"),
            status="completed", request_text="nope", result_summary="leak",
            final_message_id="ts-x",
        ))

        store2 = PostgresSurfaceSessionStore()
        await store2.setup(postgres_pool)
        try:
            # Oldest-first, scoped to this agent's thread, excluding the in-flight
            # turn — exactly the two *delivered* exchanges in order (the running,
            # undelivered, and other-agent sessions are all filtered out).
            prior = await store2.thread_history(
                _target("x"), exclude_id=ids[2], limit=20
            )
            assert [s.id for s in prior] == [ids[0], ids[1]]
            assert [(s.request_text, s.result_summary) for s in prior] == [
                ("q1", "a1"), ("q2", "a2"),
            ]
            # A different thread sees nothing of this one.
            assert await store2.thread_history(_target("x", thr="elsewhere")) == []
        finally:
            await store2.close()
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM surface_sessions WHERE thread = $1", thread
                )
        except Exception:
            pass
        await store.close()


async def test_postgres_advisory_lock_mutual_exclusion():
    """Two PostgresLock instances (≈ two replicas) can't both hold one key."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.coordination import PostgresLock

    key = f"lock-{uuid.uuid4().hex[:8]}"
    a, b = PostgresLock(DSN), PostgresLock(DSN)
    await a.setup()
    await b.setup()
    try:
        token = await a.acquire(key, ttl_seconds=60)
        assert token is not None
        # A separate instance (its own session) is refused while a holds it.
        assert await b.acquire(key, ttl_seconds=60) is None
        # renew is a no-op that confirms ownership (the session is the lease).
        assert await a.renew(key, token, ttl_seconds=60) is True
        # Explicit release frees it for the other replica.
        assert await a.release(key, token) is True
        other = await b.acquire(key, ttl_seconds=60)
        assert other is not None
        await b.release(key, other)
    finally:
        await a.close()
        await b.close()


async def test_postgres_advisory_lock_frees_when_holder_goes_away():
    """A holder whose pool closes (a graceful stand-in for a crashed replica)
    releases its session, so its advisory lock frees and another replica acquires —
    no TTL to wait out. This is the property that makes advisory locks a good fit."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.coordination import PostgresLock

    key = f"lock-{uuid.uuid4().hex[:8]}"
    a, b = PostgresLock(DSN), PostgresLock(DSN)
    await a.setup()
    await b.setup()
    try:
        token = await a.acquire(key, ttl_seconds=60)
        assert token is not None
        assert await b.acquire(key, ttl_seconds=60) is None

        # The holder "crashes": closing its pool ends the session holding the lock.
        await a.close()

        acquired = None
        for _ in range(40):  # allow a brief moment for the session to drop
            acquired = await b.acquire(key, ttl_seconds=60)
            if acquired is not None:
                break
            await asyncio.sleep(0.05)
        assert acquired is not None
        await b.release(key, acquired)
    finally:
        await b.close()
        with contextlib.suppress(Exception):
            await a.close()


async def test_session_reconcile_across_real_postgres(postgres_pool):
    """A session that crashed before delivery is recovered + delivered on a fresh
    runner reading both the session and workflow state from Postgres (Phase D)."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.runtime import Runtime
    from openloop.sessions import SessionRunner
    from openloop.sessions.postgres import PostgresSurfaceSessionStore
    from openloop.sessions.store import SurfaceSession, SurfaceTarget
    from openloop.testing import FakeGateway, FakeSurfaceDelivery
    from openloop.workflows import WorkflowEngine, WorkflowInstance
    from openloop.workflows.postgres import PostgresWorkflowStore

    sid = f"sess-{uuid.uuid4().hex[:8]}"
    agent = load_agent(AGENT_YAML)
    workflow_name = f"agent_task:{agent.metadata.name}"

    sessions = PostgresSurfaceSessionStore()
    workflows = PostgresWorkflowStore()
    await sessions.setup(postgres_pool)
    await workflows.setup(postgres_pool)
    try:
        # The turn's workflow completed, but the session crashed before delivery.
        await workflows.create(WorkflowInstance(
            id=sid, workflow=workflow_name, status="completed",
            state={
                "final_text": "recovered across restart",
                "accounted": {"model": "m", "prompt_tokens": 1,
                              "completion_tokens": 1, "cost_usd": 0.0},
                "approval_ids": [],
            },
        ))
        await sessions.upsert(SurfaceSession(
            id=sid,
            target=SurfaceTarget(surface="slack", workspace="acme",
                                 agent=agent.metadata.name, channel="C1",
                                 thread="100.1", event_id=f"ev-{sid}"),
            status="running", workflow_instance_id=sid, progress_message_id="p0",
        ))

        # Fresh stores + runner (a restart) reconcile and deliver the answer.
        sessions2 = PostgresSurfaceSessionStore()
        workflows2 = PostgresWorkflowStore()
        await sessions2.setup(postgres_pool)
        await workflows2.setup(postgres_pool)
        try:
            engine = WorkflowEngine(workflows2)
            runtime = Runtime(agent, gateway=FakeGateway(), engine=engine)
            delivery = FakeSurfaceDelivery()
            runner = SessionRunner(runtime, sessions2, delivery)

            repaired = await runner.reconcile()

            assert sid in repaired
            assert delivery.finals[-1]["text"] == "recovered across restart"
            assert (await sessions2.get(sid)).status == "completed"
            assert (await sessions2.get(sid)).final_message_id is not None
        finally:
            await sessions2.close()
            await workflows2.close()
    finally:
        try:
            pool = sessions._require_pool()
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM surface_sessions WHERE id = $1", sid)
            pool = workflows._require_pool()
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM workflow_instances WHERE id = $1", sid)
        except Exception:
            pass
        await sessions.close()
        await workflows.close()


async def test_thread_record_transcript_across_real_postgres(postgres_pool):
    """Phase A: the delivered-transcript lane round-trips and appends idempotently."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.sessions.threads import (
        PostgresThreadRecordStore,
        TranscriptFragment,
    )
    from openloop.sessions.store import SurfaceTarget

    thread = f"thr-{uuid.uuid4().hex[:8]}"
    scope = SurfaceTarget(
        surface="slack", workspace="acme", agent="dev-platform",
        channel="C1", thread=thread, event_id="ignored",
    )

    store = PostgresThreadRecordStore()
    await store.setup(postgres_pool)
    try:
        await store.get_or_create(scope)
        await store.append_delivered_fragment(
            scope, TranscriptFragment(turn_id="t1", request="q1", answer="a1")
        )
        await store.append_delivered_fragment(
            scope, TranscriptFragment(turn_id="t2", request="q2", answer="a2")
        )
        # Redelivery of t1 must not duplicate it (idempotent UPSERT on turn_id).
        await store.append_delivered_fragment(
            scope, TranscriptFragment(turn_id="t1", request="q1", answer="a1-DUP")
        )

        # A fresh store (a restart) reads the transcript back, oldest-first.
        store2 = PostgresThreadRecordStore()
        await store2.setup(postgres_pool)
        try:
            out = await store2.replayable_transcript(scope)
            assert [(f.request, f.answer) for f in out] == [("q1", "a1"), ("q2", "a2")]
            # Limit keeps the most recent, still oldest-first; exclude drops a turn.
            assert [f.turn_id for f in await store2.replayable_transcript(scope, limit=1)] == ["t2"]
            assert [f.turn_id for f in await store2.replayable_transcript(
                scope, exclude_turn_id="t2")] == ["t1"]
        finally:
            await store2.close()
    finally:
        try:
            pool = store._require_pool()
            key = "\x1f".join(("slack", "acme", "dev-platform", "C1", thread))
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM surface_thread_transcript WHERE scope_key = $1", key)
                await conn.execute(
                    "DELETE FROM surface_threads WHERE scope_key = $1", key)
        except Exception:
            pass
        await store.close()


async def test_thread_context_ref_across_real_postgres(postgres_pool):
    """Phase B: the warm-context handle column round-trips and clears."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.sessions.threads import PostgresThreadRecordStore, thread_scope_key
    from openloop.sessions.store import SurfaceTarget

    thread = f"thr-{uuid.uuid4().hex[:8]}"
    scope = SurfaceTarget(
        surface="slack", workspace="acme", agent="dev-platform",
        channel="C1", thread=thread, event_id="ignored",
    )
    key = thread_scope_key(scope)

    store = PostgresThreadRecordStore()
    await store.setup(postgres_pool)
    try:
        # The row must exist first — set_context_ref is UPDATE-only (the caller,
        # the warm pool, holds only the scope_key, not the full target).
        await store.get_or_create(scope)
        await store.set_context_ref(key, "handle-1")

        # A fresh store (a restart) reads the persisted handle back, then clears it.
        store2 = PostgresThreadRecordStore()
        await store2.setup(postgres_pool)
        try:
            assert await store2.get_context_ref(key) == "handle-1"
            await store2.set_context_ref(key, None)
            assert await store2.get_context_ref(key) is None
        finally:
            await store2.close()
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM surface_threads WHERE scope_key = $1", key)
        except Exception:
            pass
        await store.close()


async def test_thread_inbox_and_claim_across_real_postgres(postgres_pool):
    """Phase C: inbox dedup, ordered drain, and the atomic active-turn claim."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.sessions.threads import PostgresThreadRecordStore
    from openloop.sessions.store import SurfaceTarget

    thread = f"thr-{uuid.uuid4().hex[:8]}"
    scope = SurfaceTarget(
        surface="slack", workspace="acme", agent="dev-platform",
        channel="C1", thread=thread, event_id="ignored",
    )
    key = "\x1f".join(("slack", "acme", "dev-platform", "C1", thread))

    store = PostgresThreadRecordStore()
    await store.setup(postgres_pool)
    try:
        assert await store.append_inbox(scope, "e1", {"text": "one"}) is True
        assert await store.append_inbox(scope, "e2", {"text": "two"}) is True
        assert await store.append_inbox(scope, "e1", {"text": "one"}) is False  # dedup

        # A fresh store (restart) claims and drains, oldest-first; the claim is
        # exclusive against a concurrent second claimant.
        store2 = PostgresThreadRecordStore()
        await store2.setup(postgres_pool)
        try:
            assert await store.try_begin_turn(scope) is True
            assert await store2.try_begin_turn(scope) is False  # exclusive CAS

            drained = []
            while (item := await store.next_inbox(scope)) is not None:
                drained.append(item.payload["text"])
            assert drained == ["one", "two"]

            await store.end_turn(scope)
            # Nothing queued now → not claimable (drain-loop re-claim can't spin).
            assert await store2.try_begin_turn(scope) is False

            # A crashed leader's stale claim is cleared at startup, unwedging the
            # thread (with work queued again, it becomes claimable).
            await store.append_inbox(scope, "e3", {"text": "three"})
            assert await store.try_begin_turn(scope) is True  # "leader" holds
            assert await store2.try_begin_turn(scope) is False
            assert await store2.reset_active_claims() >= 1
            assert await store2.try_begin_turn(scope) is True  # unwedged
            await store2.end_turn(scope)
        finally:
            await store2.close()
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM surface_thread_inbox WHERE scope_key = $1", key)
                await conn.execute(
                    "DELETE FROM surface_threads WHERE scope_key = $1", key)
        except Exception:
            pass
        await store.close()


async def test_workflow_drive_arbitration_sql_semantics(postgres_pool):
    """The claim/fence/release/evict predicates against real server-side SQL."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.workflows import WorkflowInstance
    from openloop.workflows.postgres import PostgresWorkflowStore

    instance_id = f"wf-{uuid.uuid4().hex[:8]}"
    store = PostgresWorkflowStore()
    await store.setup(postgres_pool)
    try:
        pool = store._require_pool()

        # create: inserts once, conflict loses without overwriting.
        assert await store.create(WorkflowInstance(
            id=instance_id, workflow="t", status="running",
            state={"progress": "real"},
        ))
        assert not await store.create(WorkflowInstance(
            id=instance_id, workflow="t", status="running",
            state={"progress": "clobber"},
        ))
        assert (await store.get(instance_id)).state == {"progress": "real"}

        # claim_drive: wins unleased, loses to a live lease, wins expired.
        first = await store.claim_drive(instance_id, lease_seconds=60)
        assert first is not None and first.drive_gen == 1
        assert await store.claim_drive(instance_id, lease_seconds=60) is None
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE workflow_instances "
                "SET leased_until = now() - interval '1 second' WHERE id = $1",
                instance_id,
            )
        second = await store.claim_drive(instance_id, lease_seconds=60)
        assert second is not None and second.drive_gen == 2

        # fenced_write: stale gen mutates nothing; live gen lands but leaves
        # the (renewed) lease untouched — lease monotonicity.
        assert await store.renew_lease(instance_id, 1, lease_seconds=99) is False
        assert await store.renew_lease(instance_id, 2, lease_seconds=120)
        renewed_until = (await store.get(instance_id)).leased_until
        stale = await store.get(instance_id)
        stale.state = {"progress": "stale"}
        assert await store.fenced_write(stale, 1) is False
        assert (await store.get(instance_id)).state == {"progress": "real"}
        second.state = {"progress": "step-1"}
        assert await store.fenced_write(second, 2)
        after = await store.get(instance_id)
        assert after.state == {"progress": "step-1"}
        assert after.leased_until == renewed_until
        assert after.drive_gen == 2

        # release: gen bump + lease cleared; the writer's own gen goes stale.
        second.status = "waiting"
        second.waiting_on = "gate"
        assert await store.fenced_write(second, 2, release=True)
        parked = await store.get(instance_id)
        assert parked.drive_gen == 3 and parked.leased_until is None
        assert await store.fenced_write(second, 2) is False

        # claim_event: consumes the exact wait, leaves the lease unset.
        woken = await store.claim_event(instance_id, "gate", {"by": "x"})
        assert woken.status == "running" and woken.leased_until is None
        assert await store.claim_event(instance_id, "gate", {}) is None

        # cancel_instance: wins once (evicting via gen bump), then never again.
        cancelled = await store.cancel_instance(instance_id, "denied")
        assert cancelled.status == "cancelled"
        assert cancelled.drive_gen == woken.drive_gen + 1
        assert await store.cancel_instance(instance_id, "again") is None
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM workflow_instances WHERE id = $1", instance_id
                )
        except Exception:
            pass
        await store.close()


@pytest.mark.serial
async def test_workflow_drive_gen_migration_is_idempotent(postgres_pool):
    """setup() adds drive_gen to a pre-existing populated table, once.

    HAZARD: this DROPs a column on the shared `workflow_instances` table and
    re-adds it via setup(). It is safe only while the e2e suite runs serially —
    a concurrent test touching this table during the drop window would break.
    See the @pytest.mark.serial marker.
    """
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.workflows.postgres import PostgresWorkflowStore

    instance_id = f"wf-{uuid.uuid4().hex[:8]}"
    store = PostgresWorkflowStore()
    await store.setup(postgres_pool)
    try:
        pool = store._require_pool()
        async with pool.acquire() as conn:
            # Rewind to the pre-migration schema with a live row in place.
            await conn.execute(
                "ALTER TABLE workflow_instances DROP COLUMN IF EXISTS drive_gen"
            )
            await conn.execute(
                "INSERT INTO workflow_instances (id, workflow, status) "
                "VALUES ($1, 't', 'waiting')",
                instance_id,
            )
        await store.setup(postgres_pool)  # migrate
        await store.setup(postgres_pool)  # and prove it re-runs cleanly
        migrated = await store.get(instance_id)
        assert migrated.status == "waiting"
        assert migrated.drive_gen == 0
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM workflow_instances WHERE id = $1", instance_id
                )
        except Exception:
            pass
        await store.close()


async def test_workflow_concurrent_drives_run_steps_once(postgres_pool):
    """Two engines over one real store: exactly one claims and runs the step."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.workflows import Step, Workflow, WorkflowEngine
    from openloop.workflows.postgres import PostgresWorkflowStore

    instance_id = f"wf-{uuid.uuid4().hex[:8]}"
    calls = 0

    def _wf():
        async def work(ctx):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)

        return Workflow("t", [Step("gate", wait=True), Step("work", work)])

    store = PostgresWorkflowStore()
    await store.setup(postgres_pool)
    store2 = PostgresWorkflowStore()
    await store2.setup(postgres_pool)
    try:
        first = WorkflowEngine(store, {"t": _wf()})
        second = WorkflowEngine(store2, {"t": _wf()})
        await first.start("t", instance_id, {})
        await first.send_event(instance_id, "gate", drive=False)

        first.drive_background(instance_id)
        second.drive_background(instance_id)
        await first.wait_background(instance_id)
        await second.wait_background(instance_id)

        assert calls == 1
        assert (await store.get(instance_id)).status == "completed"
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM workflow_instances WHERE id = $1", instance_id
                )
        except Exception:
            pass
        await store2.close()
        await store.close()


async def test_workflow_lease_takeover_evicts_stale_drive(postgres_pool):
    """Forced lease expiry: a rival claims, the stale drive is cancelled and
    cannot clobber — asserting the public contract, not internal exceptions."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.workflows import Step, Workflow, WorkflowEngine
    from openloop.workflows.postgres import PostgresWorkflowStore

    instance_id = f"wf-{uuid.uuid4().hex[:8]}"
    started = asyncio.Event()
    step_cancelled = asyncio.Event()

    def _wf():
        async def hang(ctx):
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                step_cancelled.set()
                raise

        return Workflow("t", [Step("gate", wait=True), Step("hang", hang)])

    store = PostgresWorkflowStore()
    await store.setup(postgres_pool)
    store2 = PostgresWorkflowStore()
    await store2.setup(postgres_pool)
    try:
        # Lease 3s → the stale drive's ticker checks every 1s; the takeover
        # happens well inside the first tick.
        stale_engine = WorkflowEngine(store, {"t": _wf()}, lease_seconds=3)
        await stale_engine.start("t", instance_id, {})
        await stale_engine.send_event(instance_id, "gate", drive=False)
        stale_engine.drive_background(instance_id)
        await asyncio.wait_for(started.wait(), timeout=5)

        pool = store._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE workflow_instances "
                "SET leased_until = now() - interval '1 second' WHERE id = $1",
                instance_id,
            )
        rival = await store2.claim_drive(instance_id, lease_seconds=60)
        assert rival is not None

        # The stale drive's next renewal loses the fence and cancels its step.
        await asyncio.wait_for(step_cancelled.wait(), timeout=5)
        yielded = await stale_engine.wait_background(instance_id)
        assert yielded.drive_gen == rival.drive_gen  # the rival's row, as-is
        current = await store.get(instance_id)
        assert current.status == "running"
        assert "hang" not in current.completed_steps  # nothing clobbered
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM workflow_instances WHERE id = $1", instance_id
                )
        except Exception:
            pass
        await store2.close()
        await store.close()


async def test_workflow_cancel_during_drive_wins_over_writer(postgres_pool):
    """cancel_instance evicts a live drive mid-step; the terminal state and
    at-most-once callbacks survive the driver's losing write."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.workflows import Step, Workflow, WorkflowEngine
    from openloop.workflows.postgres import PostgresWorkflowStore

    instance_id = f"wf-{uuid.uuid4().hex[:8]}"
    started = asyncio.Event()
    release = asyncio.Event()
    terminal: list[str] = []

    def _wf():
        async def work(ctx):
            started.set()
            await release.wait()

        return Workflow("t", [Step("gate", wait=True), Step("work", work)])

    store = PostgresWorkflowStore()
    await store.setup(postgres_pool)
    store2 = PostgresWorkflowStore()
    await store2.setup(postgres_pool)
    try:
        driver = WorkflowEngine(store, {"t": _wf()})
        canceller = WorkflowEngine(store2, {"t": _wf()})

        async def on_terminal(inst):
            terminal.append(inst.id)

        driver.add_terminal_callback(on_terminal)
        canceller.add_terminal_callback(on_terminal)

        await driver.start("t", instance_id, {})
        await driver.send_event(instance_id, "gate", drive=False)
        driver.drive_background(instance_id)
        await asyncio.wait_for(started.wait(), timeout=5)

        cancelled = await canceller.cancel(instance_id, "denied")
        assert cancelled.status == "cancelled"
        release.set()
        await driver.wait_background(instance_id)  # its write loses the fence

        final = await store.get(instance_id)
        assert final.status == "cancelled"
        assert "work" not in final.completed_steps
        assert terminal == [instance_id]  # once, from the winning cancel only
        await canceller.cancel(instance_id, "again")
        assert terminal == [instance_id]
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM workflow_instances WHERE id = $1", instance_id
                )
        except Exception:
            pass
        await store2.close()
        await store.close()


def _approval(rid: str, *, created_at=None, **overrides):
    from datetime import datetime, timezone

    from openloop.approvals import ApprovalRequest

    return ApprovalRequest(
        agent="dev-platform",
        action="github.issues:write",
        tool="github",
        permission="issues:write",
        args={"repo": "acme/x", "title": "T"},
        approvers=["@a", "@b"],
        summary="s",
        id=rid,
        created_at=created_at or datetime.now(timezone.utc),
        **overrides,
    )


async def test_approval_claim_decision_sql_guard_under_concurrency(postgres_pool):
    """The WHERE status='pending' guard makes claim_decision win once server-side."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    rid = f"appr-{uuid.uuid4().hex[:8]}"
    store = PostgresApprovalStore()
    await store.setup(postgres_pool)
    try:
        await store.create(_approval(rid))
        results = await asyncio.gather(
            store.claim_decision(rid, "@a", approve=True),
            store.claim_decision(rid, "@b", approve=True),
            store.claim_decision(rid, "@a", approve=False),
        )
        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        stored = await store.get(rid)
        assert stored.decided_by == winners[0].decided_by
        assert stored.status == winners[0].status
        # A late claim on the decided row loses.
        assert await store.claim_decision(rid, "@b", approve=True) is None
    finally:
        await _delete_approvals(store, [rid])
        await store.close()


async def test_approval_decided_unreconciled_keyset_and_mark(postgres_pool):
    """Keyset pagination + (created_at, id) ordering + mark_reconciled idempotency."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 7, 19, tzinfo=timezone.utc)
    prefix = f"appr-{uuid.uuid4().hex[:8]}"
    ids = [f"{prefix}-{i}" for i in range(5)]
    store = PostgresApprovalStore()
    await store.setup(postgres_pool)
    try:
        for i, rid in enumerate(ids):
            await store.create(_approval(rid, created_at=base + timedelta(minutes=i)))
        # Decide the first four; leave the fifth pending. Mark the 2nd (excluded).
        for rid in ids[:4]:
            await store.claim_decision(rid, "@a", approve=True)
        await store.mark_reconciled(ids[1])

        # Walk the whole decided-unreconciled set with the keyset cursor in
        # small pages, collecting our rows. The table is shared across e2e
        # tests, so assert on the RELATIVE order and exclusions of our own ids
        # rather than exact global page contents.
        collected: list[str] = []
        cursor = None
        while True:
            page = await store.decided_unreconciled(limit=2, after=cursor)
            if not page:
                break
            cursor = (page[-1].created_at, page[-1].id)
            collected.extend(r.id for r in page if r.id in set(ids))
        # ids[0], ids[2], ids[3] in (created_at, id) order; ids[1] marked-out,
        # ids[4] still pending — and the cursor paginated past every row.
        assert collected == [ids[0], ids[2], ids[3]]

        # mark_reconciled is idempotent.
        await store.mark_reconciled(ids[0])
        marked = await store.get(ids[0])
        await store.mark_reconciled(ids[0])
        assert (await store.get(ids[0])).effect_at == marked.effect_at
    finally:
        await _delete_approvals(store, ids)
        await store.close()


@pytest.mark.serial
async def test_approval_decide_once_migration_idempotent_on_populated_table(
    postgres_pool,
):
    """setup() adds the three decide-once columns to a populated pre-migration
    table, once; the pre-existing row reads back with None sentinels and is
    swept once as a legacy decided row.

    HAZARD: this DROPs columns on the shared `approvals` table and re-adds them
    via setup(). It is safe only while the e2e suite runs serially — a
    concurrent test touching this table during the drop window would break. See
    the @pytest.mark.serial marker."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    rid = f"appr-{uuid.uuid4().hex[:8]}"
    store = PostgresApprovalStore()
    await store.setup(postgres_pool)
    try:
        pool = store._require_pool()
        async with pool.acquire() as conn:
            # Rewind to the pre-migration schema with a live decided row.
            await conn.execute(
                "ALTER TABLE approvals DROP COLUMN IF EXISTS workflow_backed"
            )
            await conn.execute(
                "ALTER TABLE approvals DROP COLUMN IF EXISTS workflow_instance_id"
            )
            await conn.execute(
                "ALTER TABLE approvals DROP COLUMN IF EXISTS effect_at"
            )
            await conn.execute(
                "INSERT INTO approvals "
                "(id, agent, action, tool, permission, status, decided_by) "
                "VALUES ($1, 'a', 'github.issues:write', 'github', "
                "'issues:write', 'approved', '@a')",
                rid,
            )
        await store.setup(postgres_pool)  # migrate
        await store.setup(postgres_pool)  # and prove it re-runs cleanly

        migrated = await store.get(rid)
        assert migrated.status == "approved"
        assert migrated.workflow_backed is None  # legacy sentinel
        assert migrated.workflow_instance_id is None
        assert migrated.effect_at is None
        # It appears once in the unreconciled sweep as a legacy decided row.
        found = await store.decided_unreconciled(limit=200)
        assert any(r.id == rid for r in found)
    finally:
        await _delete_approvals(store, [rid])
        await store.close()


async def _delete_approvals(store, ids):
    try:
        pool = store._require_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM approvals WHERE id = ANY($1)", ids)
    except Exception:
        pass
