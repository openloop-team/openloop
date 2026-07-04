"""Unit: HttpMCPClient auth — bearer tokens via the credential seam."""

from openloop.credentials import CredentialScope, EnvCredentialResolver
from openloop.tools.mcp import HttpMCPClient


async def test_no_credentials_no_headers():
    client = HttpMCPClient("http://mcp.local")
    assert await client._headers() is None


async def test_bearer_token_resolved_per_request():
    resolver = EnvCredentialResolver({"github": "tok-1"})
    client = HttpMCPClient(
        "http://mcp.local",
        credentials=resolver,
        scope=CredentialScope(integration="github"),
    )
    assert await client._headers() == {"Authorization": "Bearer tok-1"}

    # The token is not cached on the client: a rotated secret is picked up
    # on the next request (GitHub App installation tokens expire hourly).
    resolver._secrets["github"] = "tok-2"
    assert await client._headers() == {"Authorization": "Bearer tok-2"}


async def test_static_headers_merge_with_auth():
    client = HttpMCPClient(
        "http://mcp.local",
        credentials=EnvCredentialResolver({"github": "tok"}),
        scope=CredentialScope(integration="github"),
        headers={"X-MCP-Readonly": "true"},
    )
    assert await client._headers() == {
        "X-MCP-Readonly": "true",
        "Authorization": "Bearer tok",
    }


async def test_static_headers_without_credentials():
    client = HttpMCPClient("http://mcp.local", headers={"X-MCP-Readonly": "true"})
    assert await client._headers() == {"X-MCP-Readonly": "true"}
