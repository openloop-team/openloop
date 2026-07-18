import json
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
from openloop.broker_rpc.errors import RpcErrorCode, RpcProtocolProblem
from openloop.broker_rpc.identity import WorkloadIdentityToken, WorkloadIntent
from openloop.broker_rpc.models import (
    CreateJobPayload,
    RpcRequest,
    RpcResponse,
)
from openloop.broker_rpc.errors import RpcFailure


REQUEST_ID = UUID("00000000-0000-4000-8000-000000000311")


def _request():
    return RpcRequest(
        version=1,
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
        '"request_id":"00000000-0000-4000-8000-000000000311","version":1}'
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
        _frame('{"version":1,"version":1}'),
        _frame('{"version":1.0}'),
        _frame('{"value":NaN}'),
        _frame(json.dumps({"nested": [[[[[[[[[["deep"]]]]]]]]]]})),
    ],
)
def test_frame_decoder_rejects_malformed_or_ambiguous_input(frame):
    with pytest.raises(RpcProtocolProblem):
        decode_request(frame)


def test_request_decoder_rejects_unknown_fields_and_noncanonical_uuid():
    values = {
        "version": 1,
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
        1,
        REQUEST_ID,
        failure=RpcFailure(RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED),
    )
    assert decode_response(encode_response(response)) == response
    assert b"traceback" not in encode_response(response).lower()
