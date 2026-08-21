"""Engine construction — the single door to PostgreSQL (ADR 0007).

`asyncpg.create_pool` opened `min_size` connections eagerly, so a database that
was down failed at construction and the composition root could fall back.
`create_async_engine` is lazy and never fails there, so this module probes once
before returning: the caller keeps the behaviour it was written against.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_DRIVER = "asyncpg"

# libpq accepts `postgres://` as an alias for `postgresql://` and DSNs in the
# wild carry both; SQLAlchemy reads the alias as its own backend name.
_BACKENDS = frozenset({"postgresql", "postgres"})

# Query parameters asyncpg (or the dialect) takes as connect() keyword
# arguments. asyncpg's own DSN parsing treats every *other* query parameter as
# a server setting — `?search_path=x` configures the session, it is not a
# connect argument — but the dialect copies the whole query into the connect
# kwargs, so an unlisted name would reach asyncpg as `TypeError`. Splitting the
# query here keeps a DSN meaning through SQLAlchemy what it means through the
# driver.
_CONNECT_QUERY_KEYS = frozenset(
    {
        "async_fallback",
        "command_timeout",
        "direct_tls",
        "gsslib",
        "krbsrvname",
        "max_cacheable_statement_size",
        "max_cached_statement_lifetime",
        "passfile",
        "prepared_statement_cache_size",
        "ssl",
        "statement_cache_size",
        "target_session_attrs",
        "timeout",
    }
)


def _split_server_settings(url: URL) -> tuple[URL, dict[str, str]]:
    """Lift the query's server settings out of *url*, as asyncpg's DSN does."""
    settings = {
        key: value
        for key, value in url.query.items()
        if key not in _CONNECT_QUERY_KEYS and isinstance(value, str)
    }
    if not settings:
        return url, {}
    remaining = {key: value for key, value in url.query.items() if key not in settings}
    return url.set(query=remaining), settings


def normalize_dsn(dsn: str) -> str:
    """Return *dsn* addressed to the asyncpg dialect.

    Configuration carries the bare ``postgresql://`` form so a DSN can be
    pasted into `psql`; SQLAlchemy needs the driver named in the scheme.
    """
    url = make_url(dsn)
    if url.get_backend_name() not in _BACKENDS:
        raise ValueError(f"not a PostgreSQL DSN: {url.render_as_string()}")
    return url.set(drivername=f"postgresql+{_DRIVER}").render_as_string(
        hide_password=False
    )


async def create_engine(
    dsn: str,
    *,
    min_size: int,
    max_size: int,
    password: str | None = None,
    server_settings: dict[str, str] | None = None,
) -> AsyncEngine:
    """Build an engine and prove the database answers before returning it.

    ``min_size`` has no SQLAlchemy equivalent — the pool fills on demand. It is
    honoured only as "open at least one connection now", which is what made the
    old eager pool a connectivity check.

    ``server_settings`` reaches asyncpg's connect call unchanged. The broker's
    integration suite uses it to give each test its own ``search_path``.
    """
    url, from_query = _split_server_settings(make_url(normalize_dsn(dsn)))

    connect_args: dict[str, Any] = {}
    if password is not None:
        connect_args["password"] = password
    settings = {**from_query, **(server_settings or {})}
    if settings:
        connect_args["server_settings"] = settings

    engine = create_async_engine(
        url,
        pool_size=max_size,
        max_overflow=0,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    if min_size > 0:
        try:
            async with engine.connect() as connection:
                # sql-text: a bare connectivity probe; there is no expression
                # form of "does this database answer".
                await connection.execute(text("SELECT 1"))
        except BaseException:
            await engine.dispose()
            raise
    return engine
