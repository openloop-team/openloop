from datetime import UTC, datetime
from uuid import UUID

import pytest

from openloop.broker.models import (
    BrokerOwner,
    CommandKind,
    IsolationMode,
    JobSnapshot,
    JobState,
    OperationTicket,
)
from openloop.broker_rpc.capability import JobCapability
from openloop.broker_rpc.errors import RpcErrorCode, RpcFailure
from openloop.broker_rpc.identity import WorkloadIdentityToken, WorkloadIntent
from openloop.broker_rpc.models import (
    CreateJobPayload,
    CreateJobResult,
    InspectJobPayload,
    InspectJobResult,
    RpcRequest,
    RpcResponse,
)


REQUEST_ID = UUID("00000000-0000-4000-8000-000000000301")
JOB_ID = UUID("00000000-0000-4000-8000-000000000302")
TOKEN = WorkloadIdentityToken("header.payload.signature")
CAPABILITY = JobCapability("A" * 43)


def _snapshot():
    now = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
    return JobSnapshot(
        job_id=JOB_ID,
        conversation_id=UUID("00000000-0000-4000-8000-000000000303"),
        owner=BrokerOwner("tenant-a", "workload-a"),
        profile="default",
        runtime_driver="docker",
        durable_state_driver="postgres",
        state=JobState.CREATED,
        revision=1,
        generation=0,
        current_generation=None,
        pending_operation_id=None,
        durable_key_version=None,
        terminal_outcome=None,
        created_at=now,
        updated_at=now,
        generation_record=None,
    )


def test_request_method_payload_and_capability_contract_is_exact():
    create = RpcRequest(
        version=1,
        request_id=REQUEST_ID,
        method=WorkloadIntent.CREATE_JOB,
        identity_token=TOKEN,
        job_capability=None,
        payload=CreateJobPayload("rpc-create-key-01"),
    )
    inspect = RpcRequest(
        version=1,
        request_id=REQUEST_ID,
        method=WorkloadIntent.INSPECT_JOB,
        identity_token=TOKEN,
        job_capability=CAPABILITY,
        payload=InspectJobPayload(JOB_ID),
    )
    assert create.job_capability is None
    assert inspect.job_capability == CAPABILITY
    with pytest.raises(ValueError):
        RpcRequest(1, REQUEST_ID, WorkloadIntent.CREATE_JOB, TOKEN, CAPABILITY, create.payload)
    with pytest.raises(ValueError):
        RpcRequest(1, REQUEST_ID, WorkloadIntent.INSPECT_JOB, TOKEN, None, inspect.payload)
    with pytest.raises(TypeError):
        RpcRequest(1, REQUEST_ID, WorkloadIntent.CREATE_JOB, TOKEN, None, inspect.payload)


def test_response_contains_exactly_one_typed_result_or_failure():
    ticket = OperationTicket(
        operation_id=UUID("00000000-0000-4000-8000-000000000304"),
        command=CommandKind.CREATE_JOB,
        job_id=JOB_ID,
        conversation_id=UUID("00000000-0000-4000-8000-000000000303"),
        job_state=JobState.CREATED,
    )
    success = RpcResponse(
        1,
        REQUEST_ID,
        result=CreateJobResult(ticket, CAPABILITY),
    )
    inspected = RpcResponse(
        1,
        REQUEST_ID,
        result=InspectJobResult(_snapshot()),
    )
    failure = RpcResponse(
        1,
        REQUEST_ID,
        failure=RpcFailure(RpcErrorCode.UNAUTHENTICATED),
    )
    assert success.ok and inspected.ok and not failure.ok
    with pytest.raises(ValueError):
        RpcResponse(1, REQUEST_ID)
    with pytest.raises(ValueError):
        RpcResponse(
            1,
            REQUEST_ID,
            result=InspectJobResult(_snapshot()),
            failure=RpcFailure(RpcErrorCode.INTERNAL),
        )
    rendered = repr(success)
    assert CAPABILITY.value not in rendered

