import os
from pathlib import Path
import shutil
import tempfile

import pytest

from openloop.broker_rpc.audit import PeerCredentials
from openloop.broker_rpc.peer import StaticPeerCredentialProvider
from openloop.broker_rpc.server import (
    BrokerRpcServer,
    SocketPathProblem,
    UnixSocketPolicy,
)
from tests.support.broker_rpc import broker_rpc_test_fixture


@pytest.fixture
def socket_path():
    directory = Path(tempfile.mkdtemp(prefix="olrpc-", dir="/private/tmp"))
    try:
        yield directory / "broker.sock"
    finally:
        shutil.rmtree(directory)


def _server(path):
    fixture = broker_rpc_test_fixture()
    return BrokerRpcServer(
        application=fixture.application,
        socket_policy=UnixSocketPolicy(path, mode=0o600),
        peer_provider=StaticPeerCredentialProvider(
            PeerCredentials(os.getpid(), os.getuid(), os.getgid())
        ),
    )


async def test_server_refuses_to_replace_any_existing_path(socket_path):
    path = socket_path
    path.write_text("operator data")
    server = _server(path)
    with pytest.raises(SocketPathProblem):
        await server.start()
    assert path.read_text() == "operator data"


async def test_shutdown_only_unlinks_the_socket_inode_it_created(socket_path):
    path = socket_path
    server = _server(path)
    await server.start()
    path.unlink()
    path.write_text("replacement")
    await server.stop()
    assert path.read_text() == "replacement"


async def test_normal_shutdown_removes_owned_socket(socket_path):
    path = socket_path
    server = _server(path)
    await server.start()
    assert path.is_socket()
    await server.stop()
    assert not path.exists()
