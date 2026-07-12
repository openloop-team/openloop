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

AGENT_YAML = Path(__file__).parent / "data" / "agent.yaml"


def _runtime():
    return Runtime(load_agent(AGENT_YAML))


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


def test_session_store_postgres_fallback_repoints_slack_runner(monkeypatch):
    # If the Postgres session store fails to set up, the lifespan must repoint the
    # already-built Slack runner (which captured the store by reference) at the
    # in-memory fallback — otherwise background mentions hit an un-setup pool.
    from openloop import app as appmod
    from openloop.config import get_settings

    monkeypatch.setattr(
        get_settings(), "slack_bot_token", "xoxb-test", raising=False
    )

    class FailingPgSessions(PostgresSurfaceSessionStore):
        async def setup(self, pool):
            raise RuntimeError("no postgres")

    monkeypatch.setattr(
        appmod,
        "build_surface_session_store",
        lambda s: FailingPgSessions(),
    )

    app = appmod.create_app()
    runner = app.state.session_runner
    assert runner is not None
    # Before startup the runner holds the (un-setup) Postgres store.
    assert isinstance(runner.sessions, PostgresSurfaceSessionStore)

    with TestClient(app):  # runs the lifespan → setup fails → fallback
        assert isinstance(app.state.sessions, InMemorySurfaceSessionStore)
        # The runner now shares the same in-memory fallback instance.
        assert app.state.session_runner.sessions is app.state.sessions


async def test_run_socket_requires_app_token(monkeypatch):
    # No SLACK_APP_TOKEN configured → exits before opening a socket.
    from openloop.config import get_settings

    monkeypatch.setattr(get_settings(), "slack_app_token", None, raising=False)
    with pytest.raises(SystemExit):
        await run_socket()
