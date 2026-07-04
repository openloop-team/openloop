"""MCP connector — exposes a Model Context Protocol server's tools to an agent.

An MCP server publishes named tools (each with a JSON-schema input). Those tool
names *are* this connector's permissions: an action is ``<connector>.<mcp_tool>``
(e.g. ``ci-logs.get_run_logs``), and the agent's `tools` allowlist names the
exact MCP tools it may call — least privilege, consistent with native tools.

Discovery is async (it queries the server), so the connector caches its tool
list in :meth:`setup`; the sync :meth:`supported_permissions` / :meth:`describe`
then read that cache, matching the :class:`~openloop.tools.base.Tool` protocol.
The real client is behind :class:`MCPClient` so tests use a fake — no network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from openloop.credentials import CredentialResolver, CredentialScope
from openloop.tools.base import ActionSpec, ToolResult

logger = logging.getLogger(__name__)

_EMPTY_SCHEMA = {"type": "object", "properties": {}}


@dataclass(slots=True)
class MCPToolInfo:
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=lambda: dict(_EMPTY_SCHEMA))


@runtime_checkable
class MCPClient(Protocol):
    async def list_tools(self) -> list[MCPToolInfo]: ...

    async def call_tool(self, name: str, args: dict) -> str: ...


class HttpMCPClient:
    """Talks to an MCP server over streamable HTTP via the official SDK.

    The ``mcp`` SDK is an optional dependency (``pip install
    'openloop[mcp]'``) and imported lazily.

    Auth mirrors :class:`~openloop.tools.github.HttpGitHubClient`: the bearer
    token flows through the :class:`CredentialResolver` seam **per request** —
    the client never stores a raw token — so short-lived credentials (GitHub
    App installation tokens) stay fresh across a long-running process.
    ``headers`` carries static extras from config (e.g. GitHub's
    ``X-MCP-Readonly``).
    """

    def __init__(
        self,
        url: str,
        *,
        credentials: CredentialResolver | None = None,
        scope: CredentialScope | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self._credentials = credentials
        self._scope = scope
        self._static_headers = dict(headers or {})

    async def _headers(self) -> dict[str, str] | None:
        headers = dict(self._static_headers)
        if self._credentials is not None:
            scope = self._scope or CredentialScope(integration="mcp")
            token = await self._credentials.resolve(scope)
            headers["Authorization"] = f"Bearer {token}"
        return headers or None

    def _require_sdk(self):
        try:
            from mcp import ClientSession  # noqa: F401
            from mcp.client.streamable_http import streamablehttp_client  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise RuntimeError(
                "MCP support needs the 'mcp' extra: pip install "
                "'openloop[mcp]'"
            ) from exc
        return ClientSession, streamablehttp_client

    async def list_tools(self) -> list[MCPToolInfo]:  # pragma: no cover - needs server
        ClientSession, streamablehttp_client = self._require_sdk()
        headers = await self._headers()
        async with streamablehttp_client(self.url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    MCPToolInfo(
                        name=t.name,
                        description=t.description or "",
                        input_schema=dict(t.inputSchema or _EMPTY_SCHEMA),
                    )
                    for t in result.tools
                ]

    async def call_tool(self, name: str, args: dict) -> str:  # pragma: no cover - needs server
        ClientSession, streamablehttp_client = self._require_sdk()
        headers = await self._headers()
        async with streamablehttp_client(self.url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=args)
                parts = [
                    getattr(block, "text", "")
                    for block in getattr(result, "content", [])
                ]
                return "\n".join(p for p in parts if p)


class MCPConnector:
    """Maps an MCP server's tools onto the agent tool gateway."""

    def __init__(self, name: str, client: MCPClient) -> None:
        self.name = name
        self.client = client
        self._tools: dict[str, MCPToolInfo] = {}

    async def setup(self) -> None:
        """Discover and cache the server's tools."""
        self._tools = {t.name: t for t in await self.client.list_tools()}
        logger.info(
            "mcp connector %r discovered %d tool(s): %s",
            self.name,
            len(self._tools),
            ", ".join(self._tools) or "none",
        )

    def supported_permissions(self) -> set[str]:
        return set(self._tools)

    def describe(self, permission: str) -> ActionSpec:
        info = self._tools.get(permission)
        if info is None:
            return ActionSpec(f"MCP tool {permission}", dict(_EMPTY_SCHEMA))
        return ActionSpec(
            info.description or f"MCP tool {permission}",
            info.input_schema or dict(_EMPTY_SCHEMA),
        )

    async def execute(self, permission: str, args: dict) -> ToolResult:
        if permission not in self._tools:
            return ToolResult(ok=False, summary=f"unknown MCP tool {permission}")
        text = await self.client.call_tool(permission, args)
        return ToolResult(
            ok=True,
            summary=f"{self.name}.{permission} returned {len(text)} chars",
            data={"text": text},
        )
