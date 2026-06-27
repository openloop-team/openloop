"""Tests for the Slack app build (HTTP vs Socket Mode) and the socket runner."""

import pytest
from slack_bolt.async_app import AsyncApp

from openloop.agents import load_agent
from openloop.runtime import Runtime
from openloop.surfaces.slack import build_slack_app
from openloop.surfaces.slack_socket import run_socket
from openloop.testing import EXAMPLE_AGENT


def _runtime():
    return Runtime(load_agent(EXAMPLE_AGENT))


def test_build_slack_app_socket_mode_without_signing_secret():
    app = build_slack_app(_runtime(), bot_token="xoxb-test")
    assert isinstance(app, AsyncApp)


def test_build_slack_app_http_mode_with_signing_secret():
    app = build_slack_app(
        _runtime(), bot_token="xoxb-test", signing_secret="shhh"
    )
    assert isinstance(app, AsyncApp)


async def test_run_socket_requires_app_token(monkeypatch):
    # No SLACK_APP_TOKEN configured → exits before opening a socket.
    from openloop.config import get_settings

    monkeypatch.setattr(get_settings(), "slack_app_token", None, raising=False)
    with pytest.raises(SystemExit):
        await run_socket()
