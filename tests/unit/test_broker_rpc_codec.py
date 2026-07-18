import json
from datetime import UTC, datetime
from pathlib import Path
import struct
from uuid import UUID

import pytest

from openloop.broker_rpc.codec import (
    MAX_RPC_FRAME_BYTES,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from openloop.broker_rpc.capability import JobCapability
from openloop.broker_rpc.errors import RpcErrorCode, RpcProtocolProblem
from openloop.broker_rpc.identity import WorkloadIdentityToken, WorkloadIntent
from openloop.broker_rpc.models import (
    RPC_VERSION,
    CreateJobPayload,
    RunningGenerationAccess,
    RpcRequest,
    RpcResponse,
    StartSegmentPayload,
    StartSegmentResult,
)
from openloop.broker_rpc.errors import RpcFailure


REQUEST_ID = UUID("00000000-0000-4000-8000-000000000311")


def _request():
    return RpcRequest(
        version=RPC_VERSION,
        request_id=REQUEST_ID,
        method=WorkloadIntent.CREATE_JOB,
        identity_token=WorkloadIdentityToken("header.payload.signature"),
        job_capability=None,
        payload=CreateJobPayload("rpc-create-key-01"),
    )


def _frame(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack(">I", len(data)) + data


def test_request_has_frozen_compact_json_frame_fixture():
    encoded = encode_request(_request())
    body = encoded[4:].decode("utf-8")
    assert int.from_bytes(encoded[:4], "big") == len(encoded) - 4
    assert body == (
        '{"identity_token":"header.payload.signature","job_capability":null,'
        '"method":"CREATE_JOB","payload":{"idempotency_key":"rpc-create-key-01"},'
        '"request_id":"00000000-0000-4000-8000-000000000311","version":2}'
    )
    assert decode_request(encoded) == _request()


@pytest.mark.parametrize(
    "frame",
    [
        b"\x00\x00\x00\x00",
        struct.pack(">I", MAX_RPC_FRAME_BYTES + 1),
        b"\x00\x00\x00\x05{}",
        _frame("{}") + b"trailing",
        b"\x00\x00\x00\x01\xff",
        _frame('{"version":2,"version":2}'),
        _frame('{"version":2.0}'),
        _frame('{"value":NaN}'),
        _frame(json.dumps({"nested": [[[[[[[[[["deep"]]]]]]]]]]})),
    ],
)
def test_frame_decoder_rejects_malformed_or_ambiguous_input(frame):
    with pytest.raises(RpcProtocolProblem):
        decode_request(frame)


def test_request_decoder_rejects_unknown_fields_and_noncanonical_uuid():
    values = {
        "version": RPC_VERSION,
        "request_id": "AAAAAAAA-0000-4000-8000-000000000311",
        "method": "CREATE_JOB",
        "identity_token": "header.payload.signature",
        "job_capability": None,
        "payload": {"idempotency_key": "rpc-create-key-01"},
    }
    with pytest.raises(RpcProtocolProblem):
        decode_request(_frame(json.dumps(values)))
    values["request_id"] = str(REQUEST_ID)
    values["unexpected"] = True
    with pytest.raises(RpcProtocolProblem):
        decode_request(_frame(json.dumps(values)))


def test_failure_response_round_trip_is_typed_and_safe():
    response = RpcResponse(
        RPC_VERSION,
        REQUEST_ID,
        failure=RpcFailure(RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED),
    )
    assert decode_response(encode_response(response)) == response
    assert b"traceback" not in encode_response(response).lower()


def test_version_one_is_rejected_after_strict_v2_upgrade():
    value = json.loads(encode_request(_request())[4:])
    value["version"] = 1
    with pytest.raises(RpcProtocolProblem) as captured:
        decode_request(_frame(json.dumps(value)))
    assert captured.value.code is RpcErrorCode.UNSUPPORTED_VERSION


def test_start_request_and_access_response_round_trip_exactly():
    job_id = UUID("00000000-0000-4000-8000-000000000312")
    conversation_id = UUID("00000000-0000-4000-8000-000000000313")
    operation_id = UUID("00000000-0000-4000-8000-000000000314")
    request = RpcRequest(
        RPC_VERSION,
        REQUEST_ID,
        WorkloadIntent.START_SEGMENT,
        WorkloadIdentityToken("header.payload.signature"),
        JobCapability("A" * 43),
        StartSegmentPayload(job_id, 0, "rpc-start-key-0001"),
    )
    access = RunningGenerationAccess(
        job_id=job_id,
        conversation_id=conversation_id,
        generation=1,
        deadline=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        socket_path=Path(f"/run/openloop/jobs/{job_id}/1/agent.sock"),
        relay_capability="r" * 43,
        session_api_key="s" * 43,
    )
    response = RpcResponse(
        RPC_VERSION,
        REQUEST_ID,
        result=StartSegmentResult(operation_id, False, access),
    )

    assert decode_request(encode_request(request)) == request
    assert decode_response(encode_response(response)) == response
    encoded = encode_response(response)
    assert b'"type":"START_SEGMENT"' in encoded
    assert b'"expected_generation":0' in encode_request(request)
