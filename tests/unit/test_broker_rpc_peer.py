import struct

import pytest

from openloop.broker_rpc.audit import PeerCredentials
from openloop.broker_rpc.peer import (
    LinuxPeerCredentialProvider,
    PeerCredentialProblem,
    StaticPeerCredentialProvider,
    decode_linux_peer_credentials,
)


def test_linux_peer_credentials_decode_pid_uid_gid():
    assert decode_linux_peer_credentials(struct.pack("3i", 42, 1000, 1001)) == (
        PeerCredentials(42, 1000, 1001)
    )


def test_linux_peer_credentials_allow_hidden_cross_namespace_pid():
    assert decode_linux_peer_credentials(struct.pack("3i", 0, 1000, 1001)) == (
        PeerCredentials(0, 1000, 1001)
    )


@pytest.mark.parametrize("value", [b"", b"x" * 8, b"x" * 16])
def test_linux_peer_credentials_reject_wrong_kernel_shape(value):
    with pytest.raises(PeerCredentialProblem):
        decode_linux_peer_credentials(value)


def test_production_provider_rejects_unsupported_platform():
    with pytest.raises(PeerCredentialProblem):
        LinuxPeerCredentialProvider(platform="darwin")


def test_static_provider_is_deterministic():
    peer = PeerCredentials(99, 1000, 1000)
    assert StaticPeerCredentialProvider(peer).get(object()) == peer
