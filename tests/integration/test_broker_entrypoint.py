"""The external broker process owns recovery, binding, and shutdown."""

from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

import openloop.broker_main as broker_entrypoint
from openloop.broker_control import RecoveryPassReport
from openloop.broker_main import healthcheck, run_broker
from openloop.broker_rpc.client import BrokerRpcClientProblem
from openloop.broker_rpc.server import SocketPathProblem, take_over_stale_socket
from openloop.config import Settings
from openloop.wiring.broker import _derive_receipt_key, build_broker_client
from tests.support.processes import cleanup_processes


_IDENTITY_SEED = bytes(range(1, 33))
_RECEIPT_ROOT = bytes([3]) * 32
_POSTGRES_DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop",
)


def _root(seed: int) -> str:
    return base64.b64encode(bytes([seed]) * 32).decode()


@pytest.fixture
def broker_dir(short_socket_root):
    return short_socket_root


def _settings(root: Path, **overrides) -> Settings:
    identity_public = (
        Ed25519PrivateKey.from_private_bytes(_IDENTITY_SEED)
        .public_key()
        .public_bytes_raw()
    )
    receipt_public = (
        _derive_receipt_key(
            _RECEIPT_ROOT, "broker-receipt", "receipt-key-v1"
        )
        .public_key()
        .public_bytes_raw()
    )
    control = root / "control"
    state = root / "state"
    runtime = root / "runtime"
    ingress = root / "ingress"
    receipts = root / "receipts"
    control.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    runtime.mkdir(exist_ok=True)
    ingress.mkdir(exist_ok=True)
    receipts.mkdir(exist_ok=True)
    for path in (control, state, runtime, ingress, receipts):
        os.chown(path, -1, os.getgid())
    control.chmod(0o700)
    state.chmod(0o700)
    runtime.chmod(0o750)
    ingress.chmod(0o2750)
    receipts.chmod(0o2750)
    values = dict(
        openai_api_key="",
        anthropic_api_key="",
        gemini_api_key="",
        openrouter_api_key="",
        slack_bot_token=None,
        slack_signing_secret=None,
        slack_app_token=None,
        github_token=None,
        github_app_id=None,
        github_app_private_key_path=None,
        github_app_installation_id=None,
        broker_mode="external",
        broker_dev_in_memory=True,
        broker_control_socket_dir=str(control),
        broker_state_root=str(state),
        broker_runtime_root=str(runtime),
        broker_ingress_root=str(ingress),
        broker_checkpoint_receipt_root=str(receipts),
        broker_shared_data_gid=os.getgid(),
        broker_expected_app_uid=os.getuid(),
        broker_capability_roots={"cap-key-v1": _root(1)},
        broker_runtime_roots={"runtime-key-v1": _root(2)},
        broker_receipt_roots={
            "receipt-key-v1": base64.b64encode(_RECEIPT_ROOT).decode()
        },
        broker_identity_private_key=SecretStr(
            base64.b64encode(_IDENTITY_SEED).decode()
        ),
        broker_identity_public_keys={
            "identity-v1": base64.b64encode(identity_public).decode()
        },
        broker_receipt_public_keys={
            "receipt-key-v1": base64.b64encode(receipt_public).decode()
        },
        broker_reconcile_interval_seconds=60,
    )
    values.update(overrides)
    return Settings(**values)


def _empty_report(*, failed_closed: int = 0, error: int = 0):
    return RecoveryPassReport(
        items=(),
        repaired=0,
        deferred=0,
        stale=0,
        failed_closed=failed_closed,
        error=error,
    )


class _FixedReconciler:
    def __init__(self, report: RecoveryPassReport) -> None:
        self._report = report

    async def run_pass(self) -> RecoveryPassReport:
        return self._report


class _RaisingReconciler:
    async def run_pass(self) -> RecoveryPassReport:
        raise RuntimeError("injected startup recovery failure")


def test_stale_socket_takeover_matrix(broker_dir):
    path = broker_dir / "takeover.sock"
    take_over_stale_socket(path, expected_uid=os.getuid())

    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(os.fspath(path))
    dead.close()
    with pytest.raises(SocketPathProblem):
        take_over_stale_socket(path, expected_uid=os.getuid() + 1)
    assert path.exists()
    take_over_stale_socket(path, expected_uid=os.getuid())
    assert not path.exists()

    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(os.fspath(path))
    live.listen(1)
    try:
        with pytest.raises(SocketPathProblem):
            take_over_stale_socket(path, expected_uid=os.getuid())
        assert path.exists()
    finally:
        live.close()
        path.unlink()

    path.write_text("not a socket")
    with pytest.raises(SocketPathProblem):
        take_over_stale_socket(path, expected_uid=os.getuid())


async def test_startup_error_report_prevents_bind(broker_dir):
    settings = _settings(broker_dir)
    code = await run_broker(
        settings,
        _reconciler_factory=lambda **_kwargs: _FixedReconciler(
            _empty_report(error=1)
        ),
    )
    assert code == 1
    assert not (Path(settings.broker_control_socket_dir) / "control.sock").exists()


async def test_startup_recovery_exception_prevents_bind(broker_dir):
    settings = _settings(broker_dir)
    code = await run_broker(
        settings,
        _reconciler_factory=lambda **_kwargs: _RaisingReconciler(),
    )
    assert code == 1
    assert not (Path(settings.broker_control_socket_dir) / "control.sock").exists()


async def test_separate_process_refuses_coprocess_mode_before_lock(broker_dir):
    settings = _settings(broker_dir, broker_mode="coprocess")
    assert await run_broker(settings) == 2
    assert not (Path(settings.broker_control_socket_dir) / "broker.lock").exists()


async def test_failed_closed_report_allows_bind_and_clean_shutdown(broker_dir):
    settings = _settings(broker_dir)
    shutdown = asyncio.Event()
    shutdown.set()
    code = await run_broker(
        settings,
        _reconciler_factory=lambda **_kwargs: _FixedReconciler(
            _empty_report(failed_closed=2)
        ),
        _shutdown_event=shutdown,
    )
    assert code == 0
    assert not (Path(settings.broker_control_socket_dir) / "control.sock").exists()


async def test_signal_handlers_remain_installed_through_service_teardown(
    broker_dir, monkeypatch
):
    settings = _settings(broker_dir)
    shutdown = asyncio.Event()
    shutdown.set()
    events: list[str] = []

    class FakeServer:
        async def stop(self):
            events.append("server-stop")

    class FakeService:
        server = FakeServer()
        ledger = object()
        coordinator = object()
        receipt_verifier = object()

        async def bind(self):
            return None

    async def tracked_build(*_args, **_kwargs):
        return FakeService()

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        loop,
        "add_signal_handler",
        lambda signum, callback: events.append(f"add-{signum.name}"),
    )
    monkeypatch.setattr(
        loop,
        "remove_signal_handler",
        lambda signum: events.append(f"remove-{signum.name}") or True,
    )
    monkeypatch.setattr(
        broker_entrypoint,
        "build_broker_service",
        tracked_build,
    )
    monkeypatch.setattr(
        broker_entrypoint,
        "ReadOnlyCheckpointReceiptLocator",
        lambda **_kwargs: object(),
    )

    code = await broker_entrypoint.run_broker(
        settings,
        _reconciler_factory=lambda **_kwargs: _FixedReconciler(
            _empty_report()
        ),
        _shutdown_event=shutdown,
    )

    assert code == 0
    assert events == [
        "add-SIGTERM",
        "add-SIGINT",
        "server-stop",
        "remove-SIGTERM",
        "remove-SIGINT",
    ]


def _subprocess_env(settings: Settings) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BROKER_MODE": "external",
            "BROKER_DEV_IN_MEMORY": "1",
            "BROKER_CONTROL_SOCKET_DIR": settings.broker_control_socket_dir,
            "BROKER_STATE_ROOT": settings.broker_state_root,
            "BROKER_RUNTIME_ROOT": settings.broker_runtime_root,
            "BROKER_INGRESS_ROOT": settings.broker_ingress_root,
            "BROKER_CHECKPOINT_RECEIPT_ROOT": (
                settings.broker_checkpoint_receipt_root
            ),
            "BROKER_SHARED_DATA_GID": str(settings.broker_shared_data_gid),
            "BROKER_EXPECTED_APP_UID": str(settings.broker_expected_app_uid),
            "BROKER_CAPABILITY_ROOTS": json.dumps(
                {
                    key: value.get_secret_value()
                    for key, value in settings.broker_capability_roots.items()
                }
            ),
            "BROKER_RUNTIME_ROOTS": json.dumps(
                {
                    key: value.get_secret_value()
                    for key, value in settings.broker_runtime_roots.items()
                }
            ),
            "BROKER_IDENTITY_PUBLIC_KEYS": json.dumps(
                settings.broker_identity_public_keys
            ),
            "BROKER_RECEIPT_PUBLIC_KEYS": json.dumps(
                settings.broker_receipt_public_keys
            ),
            # The broker process owns only public app trust material. Blank any
            # ambient app/provider credentials inherited from the developer
            # shell so this test exercises the intended external topology.
            "BROKER_IDENTITY_PRIVATE_KEY": "",
            "BROKER_RECEIPT_ROOTS": "{}",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
            "SLACK_BOT_TOKEN": "",
            "SLACK_SIGNING_SECRET": "",
            "SLACK_APP_TOKEN": "",
            "GITHUB_TOKEN": "",
            "GITHUB_APP_ID": "",
            "GITHUB_APP_PRIVATE_KEY_PATH": "",
            "GITHUB_APP_INSTALLATION_ID": "",
            "BROKER_RECONCILE_INTERVAL_SECONDS": "60",
            "LOG_LEVEL": "info",
        }
    )
    return env


def _wait_until_healthy(settings: Settings, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _out, error = process.communicate()
            pytest.fail(
                f"broker exited {process.returncode} before healthy:\n{error}"
            )
        if healthcheck(settings) == 0:
            return
        time.sleep(0.05)
    pytest.fail("broker did not become healthy")


async def _postgres_reachable() -> bool:
    try:
        import asyncpg

        connection = await asyncpg.connect(_POSTGRES_DSN, timeout=3)
        await connection.close()
        return True
    except Exception:
        return False


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["search_path"] = schema
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )


async def test_two_process_client_seam_lock_and_signal_shutdown(broker_dir):
    settings = _settings(broker_dir)
    env = _subprocess_env(settings)
    first = None
    second = None
    try:
        first = subprocess.Popen(
            [sys.executable, "-m", "openloop.broker_main"],
            cwd=Path(__file__).parents[2],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await asyncio.to_thread(_wait_until_healthy, settings, first)

        healthy = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "openloop.broker_main", "--healthcheck"],
            cwd=Path(__file__).parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert healthy.returncode == 0, healthy.stderr

        async with AsyncExitStack() as stack:
            client = await build_broker_client(settings, stack)
            created = await client.client.create_job("entrypoint-process-seam-0001")
            inspected = await client.client.inspect_job(
                created.ticket.job_id, created.capability
            )
            assert inspected.snapshot.job_id == created.ticket.job_id

        bad_settings = _settings(
            broker_dir,
            broker_identity_private_key=SecretStr(
                base64.b64encode(bytes([9]) * 32).decode()
            ),
        )
        async with AsyncExitStack() as stack:
            bad_client = await build_broker_client(bad_settings, stack)
            with pytest.raises(BrokerRpcClientProblem):
                await bad_client.client.create_job("entrypoint-bad-identity-0001")

        second = subprocess.Popen(
            [sys.executable, "-m", "openloop.broker_main"],
            cwd=Path(__file__).parents[2],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        second_out, second_error = await asyncio.to_thread(
            second.communicate, timeout=5
        )
        assert second.returncode == 3, second_out + second_error
        assert "another broker holds the lock" in second_error
        assert "startup recovery pass" not in second_error

        first.send_signal(signal.SIGINT)
        first_out, first_error = await asyncio.to_thread(
            first.communicate, timeout=5
        )
        assert first.returncode == 0, first_out + first_error

        unhealthy = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "openloop.broker_main", "--healthcheck"],
            cwd=Path(__file__).parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert unhealthy.returncode == 1
    finally:
        cleanup_processes(second, first)


async def test_serving_sigterm_drains_and_exits_zero(broker_dir):
    settings = _settings(broker_dir)
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "openloop.broker_main"],
            cwd=Path(__file__).parents[2],
            env=_subprocess_env(settings),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await asyncio.to_thread(_wait_until_healthy, settings, process)

        process.send_signal(signal.SIGTERM)
        output, error = await asyncio.to_thread(process.communicate, timeout=5)

        assert process.returncode == 0, output + error
    finally:
        cleanup_processes(process)


def test_entrypoint_refuses_cross_boundary_root_reuse(broker_dir):
    settings = _settings(broker_dir)
    env = _subprocess_env(settings)
    env["BROKER_CAPABILITY_ROOTS"] = json.dumps(
        {"cap-key-v1": base64.b64encode(_RECEIPT_ROOT).decode()}
    )
    refused = subprocess.run(
        [sys.executable, "-m", "openloop.broker_main"],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert refused.returncode == 1
    assert "reuse across the process boundary" in refused.stderr
    assert healthcheck(settings) == 1


@pytest.mark.postgres
async def test_postgres_entrypoint_owns_broker_migrations(broker_dir):
    if not await _postgres_reachable():
        pytest.skip(f"no PostgreSQL reachable at {_POSTGRES_DSN}")
    import asyncpg

    schema = f"broker_entrypoint_{uuid4().hex}"
    admin = await asyncpg.connect(_POSTGRES_DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    await admin.close()
    schema_dsn = _dsn_with_search_path(_POSTGRES_DSN, schema)
    settings = _settings(
        broker_dir,
        broker_dev_in_memory=False,
        database_url=schema_dsn,
    )
    env = _subprocess_env(settings)
    env["BROKER_DEV_IN_MEMORY"] = "0"
    env["DATABASE_URL"] = schema_dsn
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "openloop.broker_main"],
            cwd=Path(__file__).parents[2],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await asyncio.to_thread(_wait_until_healthy, settings, process)
        process.send_signal(signal.SIGTERM)
        output, error = await asyncio.to_thread(process.communicate, timeout=5)
        assert process.returncode == 0, output + error

        connection = await asyncpg.connect(schema_dsn)
        try:
            assert await connection.fetchval(
                "SELECT to_regclass('broker_jobs') IS NOT NULL"
            )
            assert await connection.fetchval(
                "SELECT to_regclass('broker_rpc_audit') IS NOT NULL"
            )
        finally:
            await connection.close()
    finally:
        try:
            cleanup_processes(process)
        finally:
            admin = await asyncpg.connect(_POSTGRES_DSN)
            try:
                await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            finally:
                await admin.close()
