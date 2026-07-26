"""Run the Slack surface over Socket Mode — no public URL needed.

Socket Mode opens an outbound WebSocket to Slack, so you can test a real
mention → reply → approval round-trip from a laptop without a tunnel. Needs an
app-level token (``SLACK_APP_TOKEN``, ``xapp-…``) plus the bot token.

    openloop slack socket
"""

from __future__ import annotations

import logging

from openloop.config import Settings

logger = logging.getLogger("openloop.slack_socket")


async def run_socket(settings: Settings) -> None:
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

    from openloop.wiring import compose

    if not settings.slack_app_token:
        raise SystemExit(
            "Socket Mode needs SLACK_APP_TOKEN (xapp-…). Set it in .env.runtime."
        )

    async with compose(settings) as ctx:
        if ctx.slack_app is None:
            raise SystemExit(
                "No Slack app built. Set SLACK_BOT_TOKEN and ensure an agent has "
                "a Slack surface."
            )
        handler = AsyncSocketModeHandler(ctx.slack_app, settings.slack_app_token)
        logger.info("starting Slack Socket Mode — mention the bot to test")
        await handler.start_async()


def main() -> None:
    """Process entrypoint: load configuration once, then compose with it."""
    import asyncio

    from openloop.config import Settings

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_socket(Settings()))


if __name__ == "__main__":
    main()
