"""Private canary-only TCP bridge to Docker Desktop's VM-local root socket."""

from __future__ import annotations

import asyncio


async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(64 * 1024):
            writer.write(chunk)
            await writer.drain()
    finally:
        writer.close()


async def _handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_unix_connection(
            "/var/run/docker.sock"
        )
    except Exception:
        writer.close()
        return
    await asyncio.gather(
        _copy(reader, upstream_writer),
        _copy(upstream_reader, writer),
        return_exceptions=True,
    )


async def _run() -> None:
    server = await asyncio.start_server(_handle, "0.0.0.0", 2375, backlog=16)
    print("PHASE5_DOCKER_PROXY_READY", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_run())
