"""Wiring tests for GitHub auth selection (`build_github_credentials`)."""

import sys
import types

from pydantic_settings import PydanticBaseSettingsSource

from openloop.agents.schema import Agent
from openloop.app import build_github_credentials, build_tool_gateway
from openloop.approvals import InMemoryApprovalStore
from openloop.checkpoints import InMemoryCheckpointStore
from openloop.config import Settings
from openloop.credentials import EnvCredentialResolver, GitHubAppResolver
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine


class _IsolatedSettings(Settings):
    """Settings that only reads from constructor kwargs — no env vars, no .env."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[Settings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def _settings(**kwargs) -> Settings:
    return _IsolatedSettings(**kwargs)


def test_token_only_selects_env_resolver():
    resolver = build_github_credentials(_settings(github_token="tok"))
    assert isinstance(resolver, EnvCredentialResolver)


def test_no_auth_configured_returns_none():
    assert build_github_credentials(_settings()) is None


def _stub_jwt(monkeypatch, *, encode=None):
    """Install a fake ``jwt`` module (PyJWT is an optional extra)."""
    stub = types.ModuleType("jwt")
    stub.encode = encode or (lambda payload, key, algorithm: "signed")
    monkeypatch.setitem(sys.modules, "jwt", stub)


def _app_settings(key_path, **kwargs) -> Settings:
    return _settings(
        github_app_id="1234",
        github_app_private_key_path=str(key_path),
        github_app_installation_id="42",
        **kwargs,
    )


def test_app_config_wins_over_token(tmp_path, monkeypatch):
    _stub_jwt(monkeypatch)
    key = tmp_path / "app.pem"
    key.write_text("-----BEGIN RSA PRIVATE KEY-----\nfake\n")

    resolver = build_github_credentials(_app_settings(key, github_token="tok"))
    assert isinstance(resolver, GitHubAppResolver)
    assert resolver.app_id == "1234"
    assert resolver.installation_id == "42"
    assert resolver.repositories is None  # unrestricted unless configured


def test_app_repositories_setting_restricts_the_mint(tmp_path, monkeypatch):
    _stub_jwt(monkeypatch)
    key = tmp_path / "app.pem"
    key.write_text("fake")

    resolver = build_github_credentials(
        _app_settings(key, github_app_repositories="ingestion, web ,")
    )
    assert isinstance(resolver, GitHubAppResolver)
    assert resolver.repositories == ["ingestion", "web"]


def test_unusable_signing_falls_back_at_boot_not_first_call(
    tmp_path, monkeypatch, caplog
):
    """PyJWT importable but signing broken (no crypto backend / bad key) must
    hit the loud fallback at boot — the resolver must not be selected."""

    def broken_encode(payload, key, algorithm):
        raise ValueError("Algorithm 'RS256' could not be found")

    _stub_jwt(monkeypatch, encode=broken_encode)
    key = tmp_path / "app.pem"
    key.write_text("not-a-real-key")

    resolver = build_github_credentials(_app_settings(key, github_token="tok"))
    assert "GITHUB APP AUTH DISABLED" in caplog.text
    assert isinstance(resolver, EnvCredentialResolver)


def test_broken_app_config_falls_back_to_token_loudly(tmp_path, caplog):
    resolver = build_github_credentials(
        _settings(
            github_token="tok",
            github_app_id="1234",
            github_app_private_key_path=str(tmp_path / "missing.pem"),
            github_app_installation_id="42",
        )
    )
    # Explicit App config that can't start fails loudly, then degrades.
    assert "GITHUB APP AUTH DISABLED" in caplog.text
    assert isinstance(resolver, EnvCredentialResolver)


def test_broken_app_config_without_token_registers_nothing(tmp_path, caplog):
    resolver = build_github_credentials(
        _settings(
            github_app_id="1234",
            github_app_private_key_path=str(tmp_path / "missing.pem"),
            github_app_installation_id="42",
        )
    )
    assert "GITHUB APP AUTH DISABLED" in caplog.text
    assert resolver is None


def _mcp_agent(tool: dict) -> Agent:
    return Agent(
        metadata={"name": "t", "workspace": "acme"},
        spec={"model_policy": {"default": "openai/gpt-4o-mini"}, "tools": [tool]},
    )


def _build_gateway(settings: Settings, agent: Agent):
    return build_tool_gateway(
        settings,
        {"t": agent},
        InMemoryApprovalStore(),
        InMemoryCheckpointStore(),
        WorkflowEngine(InMemoryWorkflowStore()),
    )


def test_mcp_tool_with_github_credentials_gets_the_resolver():
    agent = _mcp_agent({
        "name": "github-mcp",
        "type": "mcp",
        "server": "https://api.githubcopilot.com/mcp/",
        "credentials": "github",
        "headers": {"X-MCP-Readonly": "true"},
        "permissions": ["list_issues"],
    })
    gateway = _build_gateway(_settings(github_token="tok"), agent)

    client = gateway.mcp_connectors[0].client
    assert isinstance(client._credentials, EnvCredentialResolver)
    assert client._scope.integration == "github"
    assert client._static_headers == {"X-MCP-Readonly": "true"}


def test_mcp_tool_credentials_without_github_auth_degrades_loudly(caplog):
    agent = _mcp_agent({
        "name": "github-mcp",
        "type": "mcp",
        "server": "https://api.githubcopilot.com/mcp/",
        "credentials": "github",
        "permissions": ["list_issues"],
    })
    gateway = _build_gateway(_settings(), agent)  # no GITHUB_TOKEN / app config

    client = gateway.mcp_connectors[0].client
    assert client._credentials is None
    assert "registering unauthenticated" in caplog.text


def test_mcp_tool_without_credentials_stays_unauthenticated():
    agent = _mcp_agent({
        "name": "ci-logs",
        "type": "mcp",
        "server": "http://localhost:8931",
        "permissions": ["get_run_logs"],
    })
    gateway = _build_gateway(_settings(github_token="tok"), agent)
    assert gateway.mcp_connectors[0].client._credentials is None
