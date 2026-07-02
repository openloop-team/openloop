"""Wiring tests for GitHub auth selection (`build_github_credentials`)."""

import sys
import types

from openloop.app import build_github_credentials
from openloop.config import Settings
from openloop.credentials import EnvCredentialResolver, GitHubAppResolver


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


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
