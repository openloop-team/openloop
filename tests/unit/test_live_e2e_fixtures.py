"""Guard the agent fixtures inside credential-gated live E2E modules.

The live E2E tests skip unless E2E_LIVE=1 and real credentials are present, so
a schema change that invalidates one of their hand-built agents does not fail
the normal suite — it fails the nightly, silently, for as long as nobody reads
it. (`AgentMetadata.id` became required and this fixture went unfixed for five
nightly runs.) These tests construct those fixtures unconditionally so the
drift surfaces in the regular suite instead.
"""

from tests.e2e.test_runtime_github_live import _build_agent


def test_runtime_github_live_agent_validates():
    agent = _build_agent("openai/gpt-4o")

    assert agent.metadata.name == "e2e"
    assert agent.model_for() == "openai/gpt-4o"


def test_runtime_github_live_agent_id_is_pinned():
    """Budgets and the usage ledger are keyed on the durable id, so the nightly
    runner must come back as the same principal every run — not a fresh one."""
    assert _build_agent("openai/gpt-4o").metadata.id == (
        _build_agent("openai/gpt-4o").metadata.id
    )
