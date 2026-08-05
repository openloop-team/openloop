"""Integration: worker-backend selection wiring — fail-closed, never weaker.

Phase 4 policy mirrors the Phase 3 sandbox wiring: a requested backend that
can't run safely (missing extra, unusable docker, typo'd value, or — for the
agentic backend — no fail-closed spend cap) disables the coding worker loudly
instead of degrading to a different worker or a weaker boundary.
"""

import base64
from pathlib import Path

import pytest

from openloop.agents import load_agent
from openloop.approvals import InMemoryApprovalStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.models.gateway import ModelResponse
from openloop.tasks.investigation import RepoInvestigator
from openloop.tasks.outcomes import EvidenceBundle
from openloop.tools.claude_worker import ClaudeCodeCodingWorker
from openloop.tools.coding_worker import BuiltinCodingWorker, GitWorkspaceOrchestrator
from openloop.tools.openhands_worker import (
    OpenHandsCodingWorker,
    OpenHandsUnavailable,
)
from openloop.usage import InMemoryUsageStore
from openloop.wiring import builders as appmod
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from tests.support.settings import IsolatedSettings as Settings

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _settings(**kwargs):
    return Settings(
        coding_worker_enabled=True,
        github_token="t",
        **kwargs,
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
    settings = Settings()

    assert settings.coding_worker_openhands_cold_resume_enabled is True


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


def test_openhands_docker_rejects_coprocess_broker_handle(
    monkeypatch, tmp_path, caplog
):
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


def test_rejected_docker_topology_does_not_log_state_secret(
    monkeypatch, caplog, tmp_path
):
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


def test_exposes_coding_worker_detects_legacy_yaml_tool_name():
    # `_exposes_coding_worker` backs both the boot-time fail-closed cap gate
    # (`_uncapped_worker_agents`) and ledger owner attribution. It must be
    # alias-aware: an agent whose YAML still declares the pre-rename
    # coding_worker/pr:write spelling (never migrated) is exactly as real a
    # worker-exposing agent as one spelled workspace_task/code:write, and a
    # raw (non-canonicalizing) name+permission comparison would make it
    # invisible to the gate — the exact bug this pins against a regression.
    legacy_agent = load_agent(
        Path(__file__).parents[1] / "unit" / "data" / "agent.yaml"
    )
    assert appmod._exposes_coding_worker(legacy_agent) is True


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


def test_openhands_relay_probe_failure_disables_only_coding_worker(monkeypatch, caplog):
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
            _settings(coding_worker_backend="openhands", coding_worker_sandbox="dokcer")
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


def test_claude_receives_mounted_auth_without_repr_leak(monkeypatch):
    monkeypatch.setattr(ClaudeCodeCodingWorker, "probe", lambda self: None)
    gateway = _gateway(
        _settings(
            coding_worker_backend="claude",
            claude_code_oauth_token="mounted-claude-secret",
        )
    )

    worker = gateway._tools["workspace_task"].orchestrator.worker
    assert worker._claude_auth.get_secret_value() == "mounted-claude-secret"
    assert "mounted-claude-secret" not in repr(vars(worker))


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


async def test_legacy_action_string_still_invokes_the_code_profile(
    monkeypatch, tmp_path
):
    # A durable approval row written pre-migration (or an agent whose YAML
    # hasn't been re-pointed yet) names the legacy action string. The alias
    # (Task 4) must still route it to the migrated workspace_task connector
    # and park it for approval — never bounce off as "no registered tool".
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    gateway = _gateway(_settings(coding_worker_backend="builtin"))
    agent = load_agent(AGENT_YAML)
    inv = await gateway.invoke(
        agent,
        "coding_worker.pr:write",
        args={"repo": "a/b", "instruction": "x"},
        requested_by="tester",
    )
    # Aliased to workspace_task.code:write → parks for approval (start gate), not "no tool".
    assert inv.status in {"pending_approval", "started", "approved"}
    assert inv.message is None or "no registered tool" not in inv.message


async def test_legacy_yaml_agent_write_action_is_still_approval_gated(
    monkeypatch, tmp_path
):
    # Security invariant: the alias must not let a legacy-YAML agent's write
    # action slip past the approve-before-work gate. Load the LEGACY fixture
    # tests/unit/data/agent.yaml (not the migrated AGENT_YAML above) BECAUSE
    # its `require_for` names the pre-migration spelling
    # (`coding_worker.pr:write`), never the canonical `workspace_task.code:write`
    # — using AGENT_YAML here (as the previous version of this test did) never
    # exercises the alias-dependent gate at all, since its `require_for`
    # already names the canonical action outright.
    #
    # Invoke via BOTH the legacy action string (exactly as this agent's own
    # tool_specs would hand it to a model — available_actions() reports the
    # action in the YAML's as-declared spelling, and exactly as a durable
    # approval row written pre-migration would still name it) AND the
    # canonical action string (exactly as a caller of the raw POST
    # /tools/invoke endpoint could spell it). This pins that NEITHER
    # invocation spelling can bypass approval for a legacy-`require_for`
    # agent — the exact bug fixed here: gateway.py's approval condition used
    # to compare the canonical `action` (and the as-requested string)
    # directly against `require_for` without canonicalizing `require_for`'s
    # own entries, so the canonical spelling silently matched nothing and
    # ran straight through to "executed". The gateway now canonicalizes both
    # sides symmetrically, exactly like `policy.is_allowed` does for the
    # allowlist.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    legacy_agent = load_agent(
        Path(__file__).parents[1] / "unit" / "data" / "agent.yaml"
    )

    for requested_action in ("coding_worker.pr:write", "workspace_task.code:write"):
        gateway = _gateway(
            _settings(coding_worker_backend="builtin"),
            agents={"dev-platform": legacy_agent},
        )

        inv = await gateway.invoke(
            legacy_agent,
            requested_action,
            args={"repo": "a/b", "instruction": "x"},
            requested_by="tester",
        )

        assert inv.status == "pending_approval", requested_action
        assert inv.status != "executed", requested_action
        assert inv.approval is not None, requested_action

        # The pending status alone isn't proof of a real gate — confirm a
        # durable approval row actually exists (and is still undecided) in
        # the store the gateway itself would consult on resolve().
        stored = await gateway.approvals.get(inv.approval.id)
        assert stored is not None, requested_action
        assert stored.status == "pending", requested_action
        assert stored.action == "workspace_task.code:write", requested_action


def test_code_write_gate_is_a_floor_even_if_agent_omits_it(monkeypatch):
    # Gate-as-floor (Task 13): the profile's declared gate is a mandatory
    # minimum. code:write's floor is Gate.START (always gates);
    # investigate:read's floor is Gate.NONE (never forced to gate).
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    gateway = _gateway(_settings(coding_worker_backend="builtin"))

    tool = gateway._tools["workspace_task"]
    assert getattr(tool, "requires_approval_for", None) is not None
    assert tool.requires_approval_for("code:write") is True
    assert tool.requires_approval_for("investigate:read") is False


async def test_code_write_gate_floor_forces_approval_even_without_require_for(
    monkeypatch,
):
    # Stronger proof: an agent whose `require_for` does NOT list
    # workspace_task.code:write still parks for approval — the connector's
    # floor forces it; agent config can only ADD gating, never remove it.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)
    gateway = _gateway(_settings(coding_worker_backend="builtin"))

    agent = load_agent(AGENT_YAML)
    agent.spec.approvals.require_for = [
        entry
        for entry in agent.spec.approvals.require_for
        if entry != "workspace_task.code:write"
    ]
    assert "workspace_task.code:write" not in agent.spec.approvals.require_for

    inv = await gateway.invoke(
        agent,
        "workspace_task.code:write",
        args={"repo": "a/b", "instruction": "do x"},
        requested_by="tester",
    )

    assert inv.status == "pending_approval"
    assert inv.approval is not None


async def test_investigate_read_registered_and_ungated(monkeypatch, tmp_path):
    # Task 13: with the investigator wired in production builder wiring,
    # workspace_task.investigate:read is registered and runs without parking
    # for approval — it is not in require_for and its floor is Gate.NONE.
    #
    # The connector, orchestrator, and RepoInvestigator are all built for
    # real by build_tool_gateway; only the two would-be-network calls
    # (the model completion and the credentialed git clone) are faked at the
    # class level, per the task's stated escape hatch.
    monkeypatch.setattr(OpenHandsCodingWorker, "probe", lambda self: None)

    async def fake_investigate(self, workspace, question, repo):
        return (
            EvidenceBundle(summary="it returns None", findings="- a.py:1"),
            ModelResponse(text="", model=self.model, cost_usd=0.01),
        )

    async def fake_provision_readonly(self, repo, ref=None):
        workspace = tmp_path / "investigate-ws"
        workspace.mkdir(exist_ok=True)
        return workspace

    monkeypatch.setattr(RepoInvestigator, "investigate", fake_investigate)
    monkeypatch.setattr(
        GitWorkspaceOrchestrator, "provision_readonly", fake_provision_readonly
    )

    gateway = _gateway(_settings(coding_worker_backend="builtin"))

    tool = gateway._tools["workspace_task"]
    assert "investigate:read" in tool.supported_permissions()

    agent = load_agent(AGENT_YAML)
    for t in agent.spec.tools:
        if t.name == "workspace_task":
            t.permissions = [*t.permissions, "investigate:read"]
    assert "workspace_task.investigate:read" not in agent.spec.approvals.require_for

    inv = await gateway.invoke(
        agent,
        "workspace_task.investigate:read",
        args={"repo": "a/b", "question": "why does parse return None?"},
        requested_by="tester",
    )

    assert inv.status == "executed"
    assert inv.result is not None
    assert inv.result.ok is True
    assert inv.result.data["outcome"]["kind"] == "evidence_bundle"
