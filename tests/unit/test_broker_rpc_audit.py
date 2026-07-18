from datetime import UTC, datetime
from uuid import UUID

import pytest

from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker_rpc.audit import (
    AuditDecision,
    AuditReason,
    InMemoryRpcAuditSink,
    PeerCredentials,
    RpcAuditRecord,
)
from openloop.broker_rpc.identity import WorkloadIntent, WorkloadPrincipal


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def _principal():
    return WorkloadPrincipal(
        owner=BrokerOwner("tenant-a", "workload-a"),
        worker_instance_id=UUID("00000000-0000-4000-8000-000000000501"),
        assignment_id=UUID("00000000-0000-4000-8000-000000000502"),
        isolation_mode=IsolationMode.DEDICATED,
        required_isolation=IsolationMode.SHARED,
        intents=frozenset({WorkloadIntent.CREATE_JOB}),
        key_id="issuer-v1",
        jwt_id=UUID("00000000-0000-4000-8000-000000000503"),
        issued_at=1,
        not_before=1,
        expires_at=301,
    )


def _record():
    return RpcAuditRecord(
        request_id=UUID("00000000-0000-4000-8000-000000000504"),
        method=WorkloadIntent.CREATE_JOB,
        decision=AuditDecision.ALLOWED,
        reason=AuditReason.ALLOWED,
        peer=PeerCredentials(1234, 1000, 1000),
        principal=_principal(),
        job_id=UUID("00000000-0000-4000-8000-000000000505"),
        operation_id=UUID("00000000-0000-4000-8000-000000000506"),
    )


async def test_in_memory_audit_is_immutable_bounded_and_uses_injected_time():
    sink = InMemoryRpcAuditSink(clock=lambda: NOW)
    stored = await sink.append(_record())
    assert stored.sequence == 1
    assert stored.created_at == NOW
    assert await sink.records_for_test() == (stored,)
    rendered = repr(stored)
    assert "capability" not in rendered
    assert "token" not in rendered


@pytest.mark.parametrize(
    "values",
    [(-1, 1, 1), (1, -1, 1), (1, 1, -1)],
)
def test_peer_credentials_reject_invalid_kernel_values(values):
    with pytest.raises(ValueError):
        PeerCredentials(*values)


def test_peer_credentials_accept_pid_hidden_by_container_namespace():
    assert PeerCredentials(0, 1000, 1000).pid == 0
