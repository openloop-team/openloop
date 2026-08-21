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
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_DRIVER = "asyncpg"

# libpq accepts `postgres://` as an alias for `postgresql://` and DSNs in the
# wild carry both; SQLAlchemy reads the alias as its own backend name.
_BACKENDS = frozenset({"postgresql", "postgres"})


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
    connect_args: dict[str, Any] = {}
    if password is not None:
        connect_args["password"] = password
    if server_settings is not None:
        connect_args["server_settings"] = server_settings

    engine = create_async_engine(
        normalize_dsn(dsn),
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
