"""Credential resolution behind one tenant-shaped seam (hardening Phase 1).

Every secret read flows through ``CredentialResolver.resolve(CredentialScope)``.
Clients resolve **at call time** and never store a raw token as a long-lived
attribute — so swapping where secrets come from (env today; GitHub App minting;
a secrets manager or egress proxy later) is an implementation swap, not a
refactor. The scope carries ``tenant`` from day one (always ``"default"`` in a
single-tenant deploy) so multi-tenant later changes values, not signatures.

Resolvers are the one sanctioned place a credential may be *cached* — short-TTL,
in memory — which is how a minting backend avoids a network round-trip per call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


class CredentialError(RuntimeError):
    """A credential could not be resolved for the requested scope."""


@dataclass(frozen=True, slots=True)
class CredentialScope:
    """What a credential is for: ``(integration, tenant[, agent])``.

    ``tenant`` defaults to ``"default"`` — one tenant today, but the seam is
    tenant-shaped so later phases only change the value passed in.
    """

    integration: str
    tenant: str = "default"
    agent: str | None = None


@runtime_checkable
class CredentialResolver(Protocol):
    """Resolves a scope to a secret value, at call time."""

    async def resolve(self, scope: CredentialScope) -> str: ...


class EnvCredentialResolver:
    """Today's behavior behind the seam: static secrets keyed by integration.

    Built from ``Settings`` values (which pydantic reads from env / ``.env``).
    Missing or empty entries raise :class:`CredentialError` at resolve time —
    callers that register only when configured never hit that path.
    """

    def __init__(self, secrets: dict[str, str | None]) -> None:
        self._secrets = dict(secrets)

    async def resolve(self, scope: CredentialScope) -> str:
        secret = self._secrets.get(scope.integration)
        if not secret:
            raise CredentialError(
                f"no credential configured for integration "
                f"{scope.integration!r} (tenant {scope.tenant!r})"
            )
        return secret


# GitHub App JWTs are valid up to 10 minutes; stay under with clock-skew slack.
_JWT_SKEW_SECONDS = 60
_JWT_TTL_SECONDS = 540


class GitHubAppResolver:
    """Mints short-lived GitHub App installation tokens.

    This retires the long-lived PAT: each ``resolve`` returns an installation
    token (valid ~1 hour) minted by exchanging an RS256 JWT — signed with the
    App's private key — at ``POST /app/installations/{id}/access_tokens``. The
    token is cached and re-minted ``refresh_margin_seconds`` before expiry, so
    a leaked token has a capped lifetime and the App's own permissions bound
    what it can do.

    By default the minted token spans **every repository the installation can
    access**; pass ``repositories`` (bare repo names, no owner) to restrict
    each mint to just those repos — least privilege for multi-repo installs.

    Signing needs PyJWT with the crypto extra (``pip install
    pyopenloop[githubapp]``); the import is lazy so the dependency stays
    optional. ``jwt_encoder`` / ``transport`` / ``now`` exist for tests.
    """

    integration = "github"

    def __init__(
        self,
        app_id: str | int,
        private_key: str,
        installation_id: str | int,
        *,
        repositories: list[str] | None = None,
        base_url: str = "https://api.github.com",
        refresh_margin_seconds: float = 60.0,
        jwt_encoder: Callable[[], str] | None = None,
        transport=None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.app_id = str(app_id)
        self._private_key = private_key
        self.installation_id = str(installation_id)
        self.repositories = list(repositories) if repositories else None
        self.base_url = base_url.rstrip("/")
        self._refresh_margin = timedelta(seconds=refresh_margin_seconds)
        self._jwt_encoder = jwt_encoder
        self._transport = transport
        self._now = now or (lambda: datetime.now(timezone.utc))
        # Short-TTL cache — the one place a credential may live between calls.
        self._token: str | None = None
        self._expires_at: datetime | None = None
        self._lock = asyncio.Lock()

    async def resolve(self, scope: CredentialScope) -> str:
        if scope.integration != self.integration:
            raise CredentialError(
                f"GitHubAppResolver resolves {self.integration!r}, "
                f"not {scope.integration!r}"
            )
        async with self._lock:
            if (
                self._token is not None
                and self._expires_at is not None
                and self._now() < self._expires_at - self._refresh_margin
            ):
                return self._token
            self._token, self._expires_at = await self._mint()
            return self._token

    async def _mint(self) -> tuple[str, datetime]:
        import httpx

        body = (
            {"repositories": self.repositories} if self.repositories else None
        )
        try:
            async with httpx.AsyncClient(
                timeout=30, transport=self._transport
            ) as client:
                resp = await client.post(
                    f"{self.base_url}/app/installations/"
                    f"{self.installation_id}/access_tokens",
                    headers={
                        "Authorization": f"Bearer {self._app_jwt()}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except CredentialError:
            raise
        except Exception as exc:
            raise CredentialError(
                f"minting GitHub App installation token failed: {exc}"
            ) from exc
        expires_at = datetime.fromisoformat(
            data["expires_at"].replace("Z", "+00:00")
        )
        return data["token"], expires_at

    def _app_jwt(self) -> str:
        if self._jwt_encoder is not None:
            return self._jwt_encoder()
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover — exercised via builder
            raise CredentialError(
                "GitHub App auth needs PyJWT with crypto support — "
                "install the extra: pip install pyopenloop[githubapp]"
            ) from exc
        now = int(self._now().timestamp())
        return jwt.encode(
            {
                "iat": now - _JWT_SKEW_SECONDS,
                "exp": now + _JWT_TTL_SECONDS,
                "iss": self.app_id,
            },
            self._private_key,
            algorithm="RS256",
        )
