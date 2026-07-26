"""Config defaults that `.env.runtime.example` documents to operators.

The example file is copied verbatim into `.env.runtime`, so anywhere it and the
code disagree the operator silently gets a third behaviour. These pin the two
places that had drifted: the default database name (Compose provisions
`openloop`, so the app default must name the same database) and Groq, which the
example advertises alongside the other four providers.
"""

from openloop.wiring.builders import _provider_key
from tests.support.settings import IsolatedSettings as Settings


def test_default_database_url_names_the_database_compose_provisions():
    """Compose derives `.../${POSTGRES_DB:-openloop}`; the bare default must
    point at that same database rather than a second one."""
    assert Settings().database_url.endswith("/openloop")


def test_groq_is_reported_among_the_configured_providers():
    settings = Settings(groq_api_key="groq-secret")

    assert "groq" in settings.configured_providers


def test_groq_model_resolves_the_configured_key():
    """Without this the key reaches Settings but never the model call, so Groq
    works under Compose (env_file lands in os.environ for litellm) and fails
    under direct Python, where pydantic reads the file without exporting it."""
    settings = Settings(groq_api_key="groq-secret")

    assert _provider_key(settings, "groq/llama-3.3-70b-versatile") == "groq-secret"


def test_an_unconfigured_provider_still_resolves_to_no_key():
    settings = Settings()

    assert _provider_key(settings, "groq/llama-3.3-70b-versatile") is None
    assert settings.configured_providers == []
