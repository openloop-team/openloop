"""Run the Slack surface over Socket Mode — no public URL needed.

Socket Mode opens an outbound WebSocket to Slack, so you can test a real
mention → reply → approval round-trip from a laptop without a tunnel. Needs an
app-level token (``SLACK_APP_TOKEN``, ``xapp-…``) plus the bot token.

    openloop slack socket
"""

from __future__ import annotations

import logging

logger = logging.getLogger("openloop.slack_socket")


async def run_socket() -> None:
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

    # Imported here so the module loads without constructing the app.
    from openloop.app import app
    from openloop.config import get_settings

    settings = get_settings()
    if not settings.slack_app_token:
        raise SystemExit(
            "Socket Mode needs SLACK_APP_TOKEN (xapp-…). Set it in .env."
        )

    slack_app = getattr(app.state, "slack_app", None)
    if slack_app is None:
        raise SystemExit(
            "No Slack app built. Set SLACK_BOT_TOKEN and ensure an agent has a "
            "Slack surface."
        )

    # Run the FastAPI lifespan so stores/tools are set up, then start the socket.
    async with app.router.lifespan_context(app):
        handler = AsyncSocketModeHandler(slack_app, settings.slack_app_token)
        logger.info("starting Slack Socket Mode — mention the bot to test")
        await handler.start_async()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_socket())


if __name__ == "__main__":
    main()
