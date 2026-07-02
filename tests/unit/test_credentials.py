"""Unit tests for the credential resolver seam (hardening Phase 1)."""

import dataclasses
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from openloop.credentials import (
    CredentialError,
    CredentialResolver,
    CredentialScope,
    EnvCredentialResolver,
    GitHubAppResolver,
)
from openloop.tools.github import HttpGitHubClient


# --- CredentialScope -------------------------------------------------------


def test_scope_is_tenant_shaped_with_default_tenant():
    scope = CredentialScope(integration="github")
    assert scope.tenant == "default"
    assert scope.agent is None


def test_scope_is_frozen():
    scope = CredentialScope(integration="github")
    with pytest.raises(dataclasses.FrozenInstanceError):
        scope.tenant = "other"


# --- EnvCredentialResolver -------------------------------------------------


async def test_env_resolver_resolves_configured_integration():
    resolver = EnvCredentialResolver({"github": "tok-123"})
    assert isinstance(resolver, CredentialResolver)
    token = await resolver.resolve(CredentialScope(integration="github"))
    assert token == "tok-123"


async def test_env_resolver_raises_for_missing_or_empty_secret():
    resolver = EnvCredentialResolver({"github": None, "slack": ""})
    for integration in ("github", "slack", "never-configured"):
        with pytest.raises(CredentialError):
            await resolver.resolve(CredentialScope(integration=integration))


# --- GitHubAppResolver -----------------------------------------------------


class _Clock:
    """A controllable now() so expiry behavior is testable."""

    def __init__(self) -> None:
        self.now = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def _minting_transport(counter: list, clock: _Clock, ttl_minutes: int = 60):
    """A fake GitHub that mints a numbered token per request."""

    def handler(request: httpx.Request) -> httpx.Response:
        counter.append(request)
        assert request.method == "POST"
        assert request.url.path == "/app/installations/42/access_tokens"
        assert request.headers["Authorization"] == "Bearer signed-jwt"
        expires = clock.now + timedelta(minutes=ttl_minutes)
        return httpx.Response(
            201,
            json={
                "token": f"ghs_minted{len(counter)}",
                "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

    return httpx.MockTransport(handler)


def _app_resolver(counter: list, clock: _Clock, **kwargs) -> GitHubAppResolver:
    return GitHubAppResolver(
        app_id="1234",
        private_key="-----BEGIN RSA PRIVATE KEY-----\nfake\n",
        installation_id=42,
        jwt_encoder=lambda: "signed-jwt",
        transport=_minting_transport(counter, clock, **kwargs),
        now=clock,
    )


async def test_app_resolver_mints_an_installation_token():
    counter: list = []
    resolver = _app_resolver(counter, _Clock())
    token = await resolver.resolve(CredentialScope(integration="github"))
    assert token == "ghs_minted1"
    assert len(counter) == 1


async def test_app_resolver_caches_until_near_expiry():
    counter: list = []
    clock = _Clock()
    resolver = _app_resolver(counter, clock)
    scope = CredentialScope(integration="github")

    first = await resolver.resolve(scope)
    clock.now += timedelta(minutes=30)  # well inside the 60-minute TTL
    second = await resolver.resolve(scope)
    assert first == second == "ghs_minted1"
    assert len(counter) == 1  # cached — no second mint


async def test_app_resolver_refreshes_within_expiry_margin():
    counter: list = []
    clock = _Clock()
    resolver = _app_resolver(counter, clock)
    scope = CredentialScope(integration="github")

    await resolver.resolve(scope)
    # Cross into the refresh margin (60s before the 60-minute expiry).
    clock.now += timedelta(minutes=59, seconds=30)
    refreshed = await resolver.resolve(scope)
    assert refreshed == "ghs_minted2"
    assert len(counter) == 2


def _expires(clock: _Clock) -> str:
    return (clock.now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


async def test_mint_restricts_to_configured_repositories():
    clock = _Clock()

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"repositories": ["ingestion"]}
        return httpx.Response(
            201, json={"token": "ghs_scoped", "expires_at": _expires(clock)}
        )

    resolver = GitHubAppResolver(
        app_id="1234",
        private_key="fake",
        installation_id=42,
        repositories=["ingestion"],
        jwt_encoder=lambda: "signed-jwt",
        transport=httpx.MockTransport(handler),
        now=clock,
    )
    token = await resolver.resolve(CredentialScope(integration="github"))
    assert token == "ghs_scoped"


async def test_mint_without_repositories_sends_no_restriction():
    clock = _Clock()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.content == b""  # installation-scoped: no body at all
        return httpx.Response(
            201, json={"token": "ghs_wide", "expires_at": _expires(clock)}
        )

    resolver = GitHubAppResolver(
        app_id="1234",
        private_key="fake",
        installation_id=42,
        jwt_encoder=lambda: "signed-jwt",
        transport=httpx.MockTransport(handler),
        now=clock,
    )
    token = await resolver.resolve(CredentialScope(integration="github"))
    assert token == "ghs_wide"


async def test_app_resolver_rejects_other_integrations():
    resolver = _app_resolver([], _Clock())
    with pytest.raises(CredentialError):
        await resolver.resolve(CredentialScope(integration="slack"))


async def test_app_resolver_wraps_minting_failures():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    resolver = GitHubAppResolver(
        app_id="1234",
        private_key="fake",
        installation_id=42,
        jwt_encoder=lambda: "signed-jwt",
        transport=httpx.MockTransport(failing),
    )
    with pytest.raises(CredentialError, match="minting"):
        await resolver.resolve(CredentialScope(integration="github"))


# --- Phase 1 client contract ------------------------------------------------


async def test_github_client_resolves_at_call_time_not_construction():
    """Rotating the secret behind the resolver changes the next request's
    header — proof the client holds no snapshot."""
    secrets = {"github": "tok-old"}

    class LiveResolver:
        async def resolve(self, scope):
            return secrets[scope.integration]

    client = HttpGitHubClient(LiveResolver())
    assert (await client._headers())["Authorization"] == "Bearer tok-old"
    secrets["github"] = "tok-new"
    assert (await client._headers())["Authorization"] == "Bearer tok-new"


def test_github_client_stores_no_raw_token_attribute():
    client = HttpGitHubClient(EnvCredentialResolver({"github": "tok-secret"}))
    assert "tok-secret" not in repr(vars(client))
    assert not hasattr(client, "token")
