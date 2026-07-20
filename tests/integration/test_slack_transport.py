"""Tests for the Slack app build (HTTP vs Socket Mode) and the socket runner."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from slack_bolt.async_app import AsyncApp

from openloop.agents import load_agent
from openloop.runtime import Runtime
from openloop.sessions import InMemorySurfaceSessionStore
from openloop.sessions.postgres import PostgresSurfaceSessionStore
from openloop.surfaces.slack import build_slack_app
from openloop.surfaces.slack_socket import run_socket
from openloop.testing import in_memory_workflow_engine

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _runtime():
    return Runtime(load_agent(AGENT_YAML), engine=in_memory_workflow_engine())


def test_build_slack_app_socket_mode_without_signing_secret():
    app = build_slack_app(
        _runtime(), InMemorySurfaceSessionStore(), bot_token="xoxb-test"
    )
    assert isinstance(app, AsyncApp)


def test_build_slack_app_http_mode_with_signing_secret():
    app = build_slack_app(
        _runtime(), InMemorySurfaceSessionStore(),
        bot_token="xoxb-test", signing_secret="shhh",
    )
    assert isinstance(app, AsyncApp)


def test_session_store_fallback_settles_before_slack_runner_is_built(monkeypatch):
    # No runner exists before startup. The composition root settles the fallback
    # first, then builds the runner against that final instance.
    from openloop import app as appmod
    from openloop.config import get_settings

    monkeypatch.setattr(
        get_settings(), "slack_bot_token", "xoxb-test", raising=False
    )

    class FailingPgSessions(PostgresSurfaceSessionStore):
        async def setup(self, pool):
            raise RuntimeError("no postgres")

    app = appmod.create_app(
        compose_overrides={"sessions": FailingPgSessions()}
    )
    assert getattr(app.state, "ctx", None) is None

    with TestClient(app):
        ctx = app.state.ctx
        assert isinstance(ctx.sessions, InMemorySurfaceSessionStore)
        assert ctx.session_runner is not None
        assert ctx.session_runner.sessions is ctx.sessions


async def test_run_socket_requires_app_token(monkeypatch):
    # No SLACK_APP_TOKEN configured → exits before opening a socket.
    from openloop.config import get_settings

    monkeypatch.setattr(get_settings(), "slack_app_token", None, raising=False)
    with pytest.raises(SystemExit):
        await run_socket()
