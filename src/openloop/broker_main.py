"""Standalone lifecycle for the privileged external broker process."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import socket
import stat
from typing import Any

from pydantic import ValidationError

from openloop.broker_control import (
    BrokerLifecycleReconciler,
    ReadOnlyCheckpointReceiptLocator,
)
from openloop.broker_rpc.server import take_over_stale_socket
from openloop.config import Settings
from openloop.postgres import create_pool
from openloop.wiring.broker import build_broker_service


log = logging.getLogger("openloop.broker")


class _BrokerLockHeld(RuntimeError):
    """Another broker owns the exclusive startup-and-service lifecycle."""


def _acquire_broker_lock(path: Path) -> int:
    """Open and exclusively lock the broker lifetime lock file."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags |= nofollow
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
        ):
            raise RuntimeError("broker lock path rejected")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise _BrokerLockHeld from error
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def healthcheck(settings: Settings) -> int:
    """Return zero only when the broker control socket accepts a connection."""
    try:
        if not settings.broker_control_socket_dir:
            return 1
        path = Path(settings.broker_control_socket_dir) / "control.sock"
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(2.0)
            probe.connect(os.fspath(path))
        finally:
            probe.close()
    except (OSError, TypeError, ValueError):
        return 1
    return 0


def _log_recovery_report(prefix: str, report: Any) -> None:
    log.info(
        "%s: repaired=%d deferred=%d stale=%d failed_closed=%d error=%d",
        prefix,
        report.repaired,
        report.deferred,
        report.stale,
        report.failed_closed,
        report.error,
    )


async def _periodic_reconcile(
    reconciler: Any,
    *,
    interval_seconds: int,
    shutdown: asyncio.Event,
) -> None:
    """Run lenient post-bind recovery passes until shutdown is requested."""
    while not shutdown.is_set():
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval_seconds)
            return
        except TimeoutError:
            pass
        try:
            report = await reconciler.run_pass()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - periodic recovery remains available
            log.exception("broker periodic recovery pass failed")
            continue
        _log_recovery_report("broker periodic recovery pass", report)
        if report.error:
            log.error(
                "broker periodic recovery pass completed with %d item error(s)",
                report.error,
            )


async def run_broker(
    settings: Settings,
    *,
    _reconciler_factory: Callable[..., Any] | None = None,
    _shutdown_event: asyncio.Event | None = None,
) -> int:
    """Recover, bind, and serve the external broker until a stop signal."""
    if not isinstance(settings, Settings):
        raise TypeError("settings must be Settings")
    if settings.broker_mode != "external":
        log.error("openloop-broker requires BROKER_MODE=external")
        return 2

    signal_loop: asyncio.AbstractEventLoop | None = None
    installed_signals: list[signal.Signals] = []
    try:
        if not settings.broker_control_socket_dir:
            raise ValueError("broker_control_socket_dir is required")
        socket_dir = Path(settings.broker_control_socket_dir)
        socket_path = socket_dir / "control.sock"
        lock_path = socket_dir / "broker.lock"

        async with AsyncExitStack() as stack:
            log.info("broker startup: acquiring exclusive lifecycle lock")
            lock_descriptor = _acquire_broker_lock(lock_path)
            stack.callback(os.close, lock_descriptor)
            log.info("broker startup: exclusive lifecycle lock acquired")

            pool = None
            if settings.broker_dev_in_memory:
                log.warning("broker startup: DEVELOPMENT in-memory state enabled")
            else:
                log.info("broker startup: opening Postgres pool")
                pool = await create_pool(
                    settings.database_url,
                    min_size=settings.postgres_pool_min_size,
                    max_size=settings.postgres_pool_max_size,
                )
                stack.push_async_callback(pool.close)

            log.info("broker startup: building unbound service graph")
            service = await build_broker_service(settings, stack, pool=pool)

            receipt_root = settings.broker_checkpoint_receipt_root
            expected_uid = settings.broker_expected_app_uid
            expected_gid = settings.broker_shared_data_gid
            if receipt_root is None:
                raise ValueError(
                    "broker_checkpoint_receipt_root is required in external mode"
                )
            if expected_uid is None:
                raise ValueError(
                    "broker_expected_app_uid is required in external mode"
                )
            if expected_gid is None:
                raise ValueError(
                    "broker_shared_data_gid is required in external mode"
                )
            log.info(
                "broker startup: building receipt locator and lifecycle reconciler"
            )
            locator = ReadOnlyCheckpointReceiptLocator(
                root=Path(receipt_root),
                verifier=service.receipt_verifier,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
            factory = _reconciler_factory or BrokerLifecycleReconciler
            reconciler = factory(
                ledger=service.ledger,
                coordinator=service.coordinator,
                receipt_locator=locator,
                receipt_verifier=service.receipt_verifier,
            )

            log.info("broker startup recovery pass: starting")
            report = await reconciler.run_pass()
            _log_recovery_report("broker startup recovery pass", report)
            if report.error:
                log.error(
                    "broker startup recovery pass refused bind: %d item error(s)",
                    report.error,
                )
                return 1

            log.info("broker startup: checking for a stale control socket")
            take_over_stale_socket(socket_path, expected_uid=os.getuid())
            await service.bind()
            stack.push_async_callback(service.server.stop)

            shutdown = _shutdown_event or asyncio.Event()
            signal_loop = asyncio.get_running_loop()
            for signum in (signal.SIGTERM, signal.SIGINT):
                try:
                    signal_loop.add_signal_handler(signum, shutdown.set)
                except (NotImplementedError, RuntimeError):
                    log.warning(
                        "broker could not install the %s signal handler", signum.name
                    )
                else:
                    installed_signals.append(signum)
            log.info("broker serving on %s", socket_path)
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(
                    _periodic_reconcile(
                        reconciler,
                        interval_seconds=(
                            settings.broker_reconcile_interval_seconds
                        ),
                        shutdown=shutdown,
                    )
                )
                await shutdown.wait()
            log.info("broker shutdown requested; draining")
        log.info("broker shutdown complete")
        return 0
    except _BrokerLockHeld:
        log.error("another broker holds the lock")
        return 3
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - top-level process boundary is fail-fast
        log.exception("broker startup failed")
        return 1
    finally:
        if signal_loop is not None:
            for signum in installed_signals:
                signal_loop.remove_signal_handler(signum)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openloop-broker",
        description="Run the privileged OpenLoop container broker.",
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="probe the configured broker control socket and exit",
    )
    return parser


def _log_settings_validation(error: ValidationError) -> None:
    """Log structural diagnostics without rendering validator-controlled text."""
    for detail in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = json.dumps(
            list(detail["loc"]),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        error_type = json.dumps(
            detail["type"],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        log.error(
            "broker settings validation failed: loc=%s type=%s",
            location,
            error_type,
        )


def main(argv: list[str] | None = None) -> int:
    """Console entrypoint for the broker service and active health probe."""
    args = _parser().parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as error:
        logging.basicConfig(level=logging.INFO)
        _log_settings_validation(error)
        return 1
    except Exception as error:  # noqa: BLE001 - top-level config boundary
        logging.basicConfig(level=logging.INFO)
        log.error(
            "broker settings load failed: error_type=%s",
            type(error).__name__,
        )
        return 1
    logging.basicConfig(level=settings.log_level.upper())
    if args.healthcheck:
        return healthcheck(settings)
    try:
        return asyncio.run(run_broker(settings))
    except KeyboardInterrupt:
        log.warning("broker startup interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
