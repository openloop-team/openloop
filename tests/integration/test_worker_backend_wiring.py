"""Integration: worker-backend selection wiring — fail-closed, never weaker.

Phase 4 policy mirrors the Phase 3 sandbox wiring: a requested backend that
can't run safely (missing extra, unusable docker, typo'd value, or — for the
agentic backend — no fail-closed spend cap) disables the coding worker loudly
instead of degrading to a different worker or a weaker boundary.
"""

from pathlib import Path
import base64
import pytest

from openloop.wiring import builders as appmod
from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.config import Settings
from openloop.tools.claude_worker import ClaudeCodeCodingWorker
from openloop.tools.coding_worker import BuiltinCodingWorker
from openloop.tools.openhands_worker import (
    OpenHandsCodingWorker,
    OpenHandsUnavailable,
)
from openloop.usage import InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _settings(**kwargs):
    return Settings(
        coding_worker_enabled=True, github_token="t", **kwargs
    )


def _state_master_key():
    return base64.urlsafe_b64encode(b"k" * 32).decode("ascii")


def _gateway(settings, agents=None, usage=None, broker_handle=None):
    return appmod.build_tool_gateway(
        settings,
        agents if agents is not None else {"dev-platform": load_agent(AGENT_YAML)},
        InMemoryApprovalStore(),
        InMemoryCheckpointStore(),
        WorkflowEngine(InMemoryWorkflowStore()),
        usage=usage if usage is not None else InMemoryUsageStore(),
        broker_handle=broker_handle,
    )


def test_default_backend_is_the_builtin_diff_worker_with_ledger_attached():
    gateway = _gateway(_settings())

    connector = gateway._tools["workspace_task"]
    orchestrator = connector.orchestrator
    assert isinstance(orchestrator.worker, BuiltinCodingWorker)
    # The ledger rides along on the default backend too: spend is recorded
    # and the invoking agent's per-task cap enforced (the example agent is
    # the attribution fallback).
    assert orchestrator._ledger is not None
    assert orchestrator._ledger.default_agent == "dev-platform"
    assert orchestrator._ledger.per_task_usd_for(None) == 0.50


def test_openhands_cold_resume_flag_defaults_on():
    assert Settings().coding_worker_openhands_cold_resume_enabled is True


@pytest.mark.parametrize("retired", ["git", "diff"])
def test_retired_backend_names_fail_closed(retired, caplog):
    # Both pre-release names of the builtin backend are dead values now: a
    # stale config must disable the worker loudly, never guess a mapping.
    with caplog.at_level("ERROR"):
        gateway = _gateway(_settings(coding_worker_backend=retired))

    assert "workspace_task" not in gateway._tools
    assert "unknown CODING_WORKER_BACKEND" in caplog.text
    assert "expected builtin|openhands|claude" in caplog.text


def test_unknown_backend_fails_closed(caplog):
    with caplog.at_level("ERROR"):
        gateway = _gateway(_settings(coding_worker_backend="opnhands"))
    assert "workspace_task" not in gateway._tools
    assert "unknown CODING_WORKER_BACKEND" in caplog.text
    assert "expected builtin|openhands|claude" in caplog.text
    assert "CODING WORKER DISABLED" in caplog.text


def test_openhands_docker_requires_external_broker(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_sandbox="docker",
                coding_worker_openhands_network="egress-proxy",
                coding_worker_openhands_state_dir=str(tmp_path / "openhands-state"),
                coding_worker_openhands_state_master_key=_state_master_key(),
                anthropic_api_key="sk-test",
            )
        )

    assert "workspace_task" not in gateway._tools
    assert "external broker" in caplog.text


def test_openhands_docker_rejects_coprocess_broker_handle(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_sandbox="docker",
                coding_worker_openhands_broker_enabled=True,
                broker_mode="coprocess",
                coding_worker_openhands_state_dir=str(tmp_path / "openhands-state"),
                coding_worker_openhands_state_master_key=_state_master_key(),
            ),
            broker_handle=object(),
        )

    assert "workspace_task" not in gateway._tools
    assert "BROKER_MODE=external" in caplog.text


def test_openhands_docker_without_external_broker_fails_before_state_setup(
    monkeypatch, caplog
):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_sandbox="docker",
            )
        )

    assert "workspace_task" not in gateway._tools
    assert "external broker" in caplog.text


def test_rejected_docker_topology_does_not_log_state_secret(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    invalid_secret = "not-valid-base64!!"
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_sandbox="docker",
                coding_worker_openhands_state_dir=str(tmp_path / "state"),
                coding_worker_openhands_state_master_key=invalid_secret,
            )
        )

    assert "workspace_task" not in gateway._tools
    assert "external broker" in caplog.text
    assert invalid_secret not in caplog.text


def test_openhands_cold_resume_refuses_host_mode(monkeypatch, caplog):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_sandbox="host",
                coding_worker_openhands_cold_resume_enabled=True,
            )
        )

    assert "workspace_task" not in gateway._tools
    assert "cold resume requires" in caplog.text


def test_openhands_state_secret_is_redacted_from_settings_repr():
    secret = _state_master_key()
    settings = _settings(coding_worker_openhands_state_master_key=secret)
    assert secret not in repr(settings)


def test_openhands_without_per_task_cap_fails_closed(monkeypatch, caplog):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    agent = load_agent(AGENT_YAML)
    agent.spec.budget.per_task_usd = None

    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_openhands_cold_resume_enabled=False,
            ),
            agents={"dev-platform": agent},
        )

    assert "workspace_task" not in gateway._tools
    assert "CODING WORKER DISABLED" in caplog.text
    assert "per_task_usd" in caplog.text


def test_openhands_requires_a_cap_on_every_worker_agent(monkeypatch, caplog):
    # Phase 5 attribution enforces the *invoking* agent's cap, so the gate
    # must hold for every agent exposing the tool — one capped owner is no
    # longer enough.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    capped = load_agent(AGENT_YAML)
    uncapped = load_agent(AGENT_YAML)
    uncapped.metadata.name = "docs-bot"
    uncapped.spec.budget.per_task_usd = None

    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_openhands_cold_resume_enabled=False,
            ),
            agents={"dev-platform": capped, "docs-bot": uncapped},
        )

    assert "workspace_task" not in gateway._tools
    assert "CODING WORKER DISABLED" in caplog.text
    assert "docs-bot" in caplog.text  # the gate names the offender


def test_openhands_ignores_uncapped_agent_without_worker_action(monkeypatch):
    # Tool name alone is not enough: only agents that can invoke
    # workspace_task.code:write need a cap and can become the fallback owner.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    capped = load_agent(AGENT_YAML)
    observer = load_agent(AGENT_YAML)
    observer.metadata.name = "docs-bot"
    observer.spec.budget.per_task_usd = None
    for tool in observer.spec.tools:
        if tool.name == "workspace_task":
            tool.permissions = []

    gateway = _gateway(
        _settings(
            coding_worker_backend="openhands",
            coding_worker_openhands_cold_resume_enabled=False,
        ),
        agents={"docs-bot": observer, "dev-platform": capped},
    )

    connector = gateway._tools["workspace_task"]
    assert connector.orchestrator._ledger.default_agent == "dev-platform"


def test_openhands_without_usage_store_fails_closed(monkeypatch, caplog):
    # A deploy that passes no usage store cannot build a ledger — the agentic
    # backend must not run uncapped and unrecorded.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = appmod.build_tool_gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_openhands_cold_resume_enabled=False,
            ),
            {"dev-platform": load_agent(AGENT_YAML)},
            InMemoryApprovalStore(),
            InMemoryCheckpointStore(),
            WorkflowEngine(InMemoryWorkflowStore()),
            usage=None,
        )
    assert "workspace_task" not in gateway._tools
    assert "CODING WORKER DISABLED" in caplog.text


def test_openhands_probe_failure_fails_closed(monkeypatch, caplog):
    def boom(self):
        raise OpenHandsUnavailable("openhands extra not installed")

    monkeypatch.setattr(OpenHandsCodingWorker, "probe", boom)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_openhands_cold_resume_enabled=False,
            )
        )
    assert "workspace_task" not in gateway._tools
    assert "openhands backend probe failed" in caplog.text
    assert "CODING WORKER DISABLED" in caplog.text


def test_openhands_relay_probe_failure_disables_only_coding_worker(
    monkeypatch, caplog
):
    def boom(self):
        raise OpenHandsUnavailable(
            "native OpenHands relay compatibility check failed: SDK seam changed"
        )

    monkeypatch.setattr(OpenHandsCodingWorker, "probe", boom)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands",
                coding_worker_openhands_cold_resume_enabled=False,
            )
        )

    assert "workspace_task" not in gateway._tools
    assert "github" in gateway._tools
    assert "openhands backend probe failed" in caplog.text
    assert "native OpenHands relay compatibility check failed" in caplog.text


def test_openhands_with_sandbox_typo_fails_closed(monkeypatch, caplog):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(
                coding_worker_backend="openhands", coding_worker_sandbox="dokcer"
            )
        )
    assert "workspace_task" not in gateway._tools
    assert "unknown CODING_WORKER_SANDBOX" in caplog.text


def test_claude_registers_in_host_mode(monkeypatch):
    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", lambda self: None)
    gateway = _gateway(_settings(coding_worker_backend="claude"))

    worker = gateway._tools["workspace_task"].orchestrator.worker
    assert isinstance(worker, ClaudeCodeCodingWorker)
    # --max-turns + the deadline are threaded from settings as the fail-closed
    # bound; the deadline default (600) is passed through, not disabled.
    assert worker.max_turns == 100
    assert worker.deadline_seconds == 600.0


def test_claude_docker_mode_fails_closed(monkeypatch, caplog):
    # Docker isolation for the claude backend is not implemented: requesting it
    # must disable the worker loudly, never silently run on the host.
    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", lambda self: None)
    with caplog.at_level("ERROR"):
        gateway = _gateway(
            _settings(coding_worker_backend="claude", coding_worker_sandbox="docker")
        )
    assert "workspace_task" not in gateway._tools
    assert "supports only CODING_WORKER_SANDBOX=host" in caplog.text
    assert "CODING WORKER DISABLED" in caplog.text


def test_claude_registers_without_a_per_task_dollar_cap(monkeypatch):
    # Unlike openhands, the claude backend's fail-closed bound is turns +
    # deadline (the subscription dollar signal is unreliable), so it does NOT
    # require a per-task dollar cap to register. The ledger still rides along.
    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", lambda self: None)
    uncapped = load_agent(AGENT_YAML)
    uncapped.spec.budget.per_task_usd = None

    gateway = _gateway(
        _settings(coding_worker_backend="claude"),
        agents={"dev-platform": uncapped},
    )

    assert "workspace_task" in gateway._tools
    assert gateway._tools["workspace_task"].orchestrator._ledger is not None


def test_claude_probe_failure_fails_closed(monkeypatch, caplog):
    def boom(self):
        from openloop.tools.claude_worker import ClaudeCodeUnavailable

        raise ClaudeCodeUnavailable("claude CLI not found")

    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", boom)
    with caplog.at_level("ERROR"):
        gateway = _gateway(_settings(coding_worker_backend="claude"))
    assert "workspace_task" not in gateway._tools
    assert "claude backend probe failed" in caplog.text
    assert "CODING WORKER DISABLED" in caplog.text


def test_workspace_task_tool_registered_for_code_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    gateway = _gateway(_settings(coding_worker_backend="builtin"))
    assert "workspace_task" in gateway._tools
    assert "code:write" in gateway._tools["workspace_task"].supported_permissions()


async def test_legacy_action_string_still_invokes_the_code_profile(monkeypatch, tmp_path):
    # A durable approval row written pre-migration (or an agent whose YAML
    # hasn't been re-pointed yet) names the legacy action string. The alias
    # (Task 4) must still route it to the migrated workspace_task connector
    # and park it for approval — never bounce off as "no registered tool".
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    gateway = _gateway(_settings(coding_worker_backend="builtin"))
    agent = load_agent(AGENT_YAML)
    inv = await gateway.invoke(agent, "coding_worker.pr:write",
                               args={"repo": "a/b", "instruction": "x"},
                               requested_by="tester")
    # Aliased to workspace_task.code:write → parks for approval (start gate), not "no tool".
    assert inv.status in {"pending_approval", "started", "approved"}
    assert inv.message is None or "no registered tool" not in inv.message


async def test_legacy_yaml_agent_write_action_is_still_approval_gated(monkeypatch, tmp_path):
    # Security invariant: the alias must not let a legacy-YAML agent's write
    # action slip past the approve-before-work gate. Load the LEGACY fixture
    # tests/unit/data/agent.yaml (not the migrated AGENT_YAML above) BECAUSE
    # its `require_for` names the pre-migration spelling
    # (`coding_worker.pr:write`), never the canonical `workspace_task.code:write`
    # — using AGENT_YAML here (as the previous version of this test did) never
    # exercises the alias-dependent gate at all, since its `require_for`
    # already names the canonical action outright.
    #
    # Invoke via the legacy action string, exactly as this agent's own
    # tool_specs would hand it to a model (available_actions() reports the
    # action in the YAML's as-declared spelling) and exactly as a durable
    # approval row written pre-migration would still name it. `requires_approval`
    # does a plain membership check against `require_for`; the real bug this
    # pins was `requires_approval(canonical_action)` returning False for an
    # agent like this, so gateway.py now also checks `requires_approval` against
    # the as-requested string. Confirm it parks as a genuine, durable approval
    # row rather than running straight through to "executed".
    #
    # NOTE: invoking this same agent with the CANONICAL string directly
    # (`workspace_task.code:write`, as a caller of the raw POST /tools/invoke
    # endpoint could) is NOT gated by the current gateway.py check — that
    # check only tries the canonical action and the exact as-requested string
    # against `require_for`, it never re-canonicalizes `require_for`'s own
    # legacy entries for comparison. That is a distinct, real, reachable gap
    # discovered while fixing this test's fixture (confirmed via manual
    # reproduction), not covered here because closing it requires a src/
    # change out of scope for this test-only task — see task-10-report.md.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    legacy_agent = load_agent(Path(__file__).parents[1] / "unit" / "data" / "agent.yaml")
    gateway = _gateway(
        _settings(coding_worker_backend="builtin"),
        agents={"dev-platform": legacy_agent},
    )

    inv = await gateway.invoke(
        legacy_agent,
        "coding_worker.pr:write",
        args={"repo": "a/b", "instruction": "x"},
        requested_by="tester",
    )

    assert inv.status == "pending_approval"
    assert inv.status != "executed"
    assert inv.approval is not None

    # The pending status alone isn't proof of a real gate — confirm a
    # durable approval row actually exists (and is still undecided) in
    # the store the gateway itself would consult on resolve().
    stored = await gateway.approvals.get(inv.approval.id)
    assert stored is not None
    assert stored.status == "pending"
    assert stored.action == "workspace_task.code:write"
