"""Slack file bytes for the upload provisioner (Phase 4 lazy staging).

Upload staging is metadata-only at share time; this fetcher is the one place
bytes leave Slack, and it runs solely inside the sealed-analysis
orchestrator's provisioning step — post-approval, post-monthly-gate — with the
remaining byte allowance enforced DURING the download.

Requires the ``files:read`` bot scope: ``files.info`` resolves the private
download URL, then the bytes are streamed with the bot token as the bearer.
Failure copy is sanitized (no token, no URLs) because it lands in workflow
failure records and surface replies.
"""

from __future__ import annotations

import logging

from openloop.analysis.provision import ProvisionError

logger = logging.getLogger(__name__)


class SlackUploadFetcher:
    """Fetches one shared file's bytes from Slack, capped in flight."""

    def __init__(self, bot_token: str) -> None:
        self._token = bot_token

    async def fetch(self, upload_ref: str, *, max_bytes: int) -> bytes:
        import httpx
        from slack_sdk.errors import SlackApiError
        from slack_sdk.web.async_client import AsyncWebClient

        client = AsyncWebClient(token=self._token)
        try:
            info = await client.files_info(file=upload_ref)
        except SlackApiError as exc:
            error = exc.response.get("error", "unknown_error")
            # file_deleted / file_not_found: shared, approved, then removed
            # before provisioning — a clean terminal failure by design.
            raise ProvisionError(
                f"the shared file is no longer available on Slack ({error})"
            ) from exc
        url = (info.get("file") or {}).get("url_private_download") or (
            info.get("file") or {}
        ).get("url_private")
        if not url:
            raise ProvisionError(
                "Slack did not return a download location for the shared file"
            )
        received = bytearray()
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
                async with http.stream(
                    "GET", url, headers={"Authorization": f"Bearer {self._token}"}
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if len(received) + len(chunk) > max_bytes:
                            raise ProvisionError(
                                f"the shared file exceeds the {max_bytes}-byte "
                                "upload cap"
                            )
                        received.extend(chunk)
        except ProvisionError:
            raise
        except httpx.HTTPError as exc:
            raise ProvisionError(
                f"downloading the shared file failed: {type(exc).__name__}"
            ) from exc
        return bytes(received)
