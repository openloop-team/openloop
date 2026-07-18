"""Strict bounded JSON framing for broker RPC."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import struct
from typing import Any
from uuid import UUID

from openloop.broker.models import (
    BrokerOwner,
    CommandKind,
    GenerationSnapshot,
    GenerationState,
    JobSnapshot,
    JobState,
    OperationTicket,
    ReleaseTarget,
    TerminalOutcome,
    validate_timestamp,
)

from .capability import JobCapability
from .errors import RpcErrorCode, RpcFailure, RpcProtocolProblem
from .identity import WorkloadIdentityToken, WorkloadIntent
from .models import (
    CreateJobPayload,
    CreateJobResult,
    InspectJobPayload,
    InspectJobResult,
    RPC_VERSION,
    RunningGenerationAccess,
    RpcRequest,
    RpcResponse,
    StartSegmentPayload,
    StartSegmentResult,
)


MAX_RPC_FRAME_BYTES = 32 * 1024
MAX_RPC_JSON_DEPTH = 8
MAX_RPC_JSON_NODES = 256


def _reject_float(_value: str) -> None:
    raise RpcProtocolProblem()


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RpcProtocolProblem()
        value[key] = item
    return value


def _validate_shape(value: object, *, depth: int = 0) -> int:
    if depth > MAX_RPC_JSON_DEPTH:
        raise RpcProtocolProblem()
    nodes = 1
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RpcProtocolProblem()
            nodes += _validate_shape(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            nodes += _validate_shape(item, depth=depth + 1)
    elif value is not None and type(value) not in {bool, int, str}:
        raise RpcProtocolProblem()
    if nodes > MAX_RPC_JSON_NODES:
        raise RpcProtocolProblem()
    return nodes


def _decode_frame(frame: bytes) -> dict[str, object]:
    if not isinstance(frame, bytes) or len(frame) < 4:
        raise RpcProtocolProblem(RpcErrorCode.MALFORMED_FRAME)
    length = struct.unpack(">I", frame[:4])[0]
    if length == 0 or length > MAX_RPC_FRAME_BYTES or len(frame) != length + 4:
        raise RpcProtocolProblem(RpcErrorCode.MALFORMED_FRAME)
    try:
        text = frame[4:].decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_object,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except RpcProtocolProblem:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RpcProtocolProblem(RpcErrorCode.MALFORMED_FRAME) from error
    _validate_shape(value)
    if not isinstance(value, dict):
        raise RpcProtocolProblem()
    return value


def _encode_frame(value: dict[str, object]) -> bytes:
    try:
        body = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RpcProtocolProblem() from error
    if not 1 <= len(body) <= MAX_RPC_FRAME_BYTES:
        raise RpcProtocolProblem(RpcErrorCode.MALFORMED_FRAME)
    return struct.pack(">I", len(body)) + body


def _exact(value: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise RpcProtocolProblem()
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RpcProtocolProblem()
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise RpcProtocolProblem()
    return value


def _uuid(value: object) -> UUID:
    text = _text(value)
    try:
        parsed = UUID(text)
    except ValueError as error:
        raise RpcProtocolProblem() from error
    if str(parsed) != text:
        raise RpcProtocolProblem()
    return parsed


def _optional_uuid(value: object) -> UUID | None:
    return None if value is None else _uuid(value)


def _timestamp(value: object) -> datetime:
    text = _text(value)
    try:
        parsed = datetime.fromisoformat(text)
        validate_timestamp("RPC timestamp", parsed)
    except (TypeError, ValueError) as error:
        raise RpcProtocolProblem() from error
    if parsed.isoformat() != text:
        raise RpcProtocolProblem()
    return parsed


def _request_dict(request: RpcRequest) -> dict[str, object]:
    if request.method is WorkloadIntent.CREATE_JOB:
        assert isinstance(request.payload, CreateJobPayload)
        payload: dict[str, object] = {
            "idempotency_key": request.payload.idempotency_key
        }
    elif request.method is WorkloadIntent.INSPECT_JOB:
        assert isinstance(request.payload, InspectJobPayload)
        payload = {"job_id": str(request.payload.job_id)}
    else:
        assert request.method is WorkloadIntent.START_SEGMENT
        assert isinstance(request.payload, StartSegmentPayload)
        payload = {
            "job_id": str(request.payload.job_id),
            "expected_generation": request.payload.expected_generation,
            "idempotency_key": request.payload.idempotency_key,
        }
    return {
        "version": request.version,
        "request_id": str(request.request_id),
        "method": request.method.value,
        "identity_token": request.identity_token.value,
        "job_capability": (
            request.job_capability.value if request.job_capability else None
        ),
        "payload": payload,
    }


def encode_request(request: RpcRequest) -> bytes:
    if not isinstance(request, RpcRequest):
        raise TypeError("request must be RpcRequest")
    return _encode_frame(_request_dict(request))


def decode_request(frame: bytes) -> RpcRequest:
    value = _exact(
        _decode_frame(frame),
        {
            "version",
            "request_id",
            "method",
            "identity_token",
            "job_capability",
            "payload",
        },
    )
    version = _integer(value["version"])
    if version != RPC_VERSION:
        raise RpcProtocolProblem(RpcErrorCode.UNSUPPORTED_VERSION)
    try:
        method = WorkloadIntent(_text(value["method"]))
        token = WorkloadIdentityToken(_text(value["identity_token"]))
        capability = (
            None
            if value["job_capability"] is None
            else JobCapability(_text(value["job_capability"]))
        )
        if method is WorkloadIntent.CREATE_JOB:
            payload_value = _exact(value["payload"], {"idempotency_key"})
            payload = CreateJobPayload(_text(payload_value["idempotency_key"]))
        elif method is WorkloadIntent.INSPECT_JOB:
            payload_value = _exact(value["payload"], {"job_id"})
            payload = InspectJobPayload(_uuid(payload_value["job_id"]))
        else:
            payload_value = _exact(
                value["payload"],
                {"job_id", "expected_generation", "idempotency_key"},
            )
            payload = StartSegmentPayload(
                _uuid(payload_value["job_id"]),
                _integer(payload_value["expected_generation"]),
                _text(payload_value["idempotency_key"]),
            )
        return RpcRequest(
            version=version,
            request_id=_uuid(value["request_id"]),
            method=method,
            identity_token=token,
            job_capability=capability,
            payload=payload,
        )
    except RpcProtocolProblem:
        raise
    except (TypeError, ValueError, UnicodeError) as error:
        raise RpcProtocolProblem() from error


def _ticket_dict(ticket: OperationTicket) -> dict[str, object]:
    return {
        "operation_id": str(ticket.operation_id),
        "command": ticket.command.value,
        "job_id": str(ticket.job_id) if ticket.job_id else None,
        "conversation_id": str(ticket.conversation_id) if ticket.conversation_id else None,
        "generation": ticket.generation,
        "job_state": ticket.job_state.value if ticket.job_state else None,
        "generation_state": (
            ticket.generation_state.value if ticket.generation_state else None
        ),
        "replayed": ticket.replayed,
    }


def _decode_ticket(value: object) -> OperationTicket:
    item = _exact(
        value,
        {
            "operation_id",
            "command",
            "job_id",
            "conversation_id",
            "generation",
            "job_state",
            "generation_state",
            "replayed",
        },
    )
    replayed = item["replayed"]
    if type(replayed) is not bool:
        raise RpcProtocolProblem()
    generation = item["generation"]
    if generation is not None:
        generation = _integer(generation)
    try:
        return OperationTicket(
            operation_id=_uuid(item["operation_id"]),
            command=CommandKind(_text(item["command"])),
            job_id=_optional_uuid(item["job_id"]),
            conversation_id=_optional_uuid(item["conversation_id"]),
            generation=generation,
            job_state=(
                JobState(_text(item["job_state"]))
                if item["job_state"] is not None
                else None
            ),
            generation_state=(
                GenerationState(_text(item["generation_state"]))
                if item["generation_state"] is not None
                else None
            ),
            replayed=replayed,
        )
    except (TypeError, ValueError) as error:
        raise RpcProtocolProblem() from error


def _generation_dict(value: GenerationSnapshot) -> dict[str, object]:
    return {
        "generation": value.generation,
        "state": value.state.value,
        "revision": value.revision,
        "previous_job_state": value.previous_job_state.value,
        "start_operation_id": str(value.start_operation_id),
        "pending_operation_id": (
            str(value.pending_operation_id) if value.pending_operation_id else None
        ),
        "runtime_key_version": value.runtime_key_version,
        "durable_key_version": value.durable_key_version,
        "execution_lease_deadline": value.execution_lease_deadline.isoformat(),
        "barrier_id": value.barrier_id,
        "receipt_id": value.receipt_id,
        "release_target": value.release_target.value if value.release_target else None,
        "release_terminal_outcome": (
            value.release_terminal_outcome.value
            if value.release_terminal_outcome
            else None
        ),
        "failure_reason_code": value.failure_reason_code,
        "created_at": value.created_at.isoformat(),
        "updated_at": value.updated_at.isoformat(),
    }


_GENERATION_FIELDS = {
    "generation",
    "state",
    "revision",
    "previous_job_state",
    "start_operation_id",
    "pending_operation_id",
    "runtime_key_version",
    "durable_key_version",
    "execution_lease_deadline",
    "barrier_id",
    "receipt_id",
    "release_target",
    "release_terminal_outcome",
    "failure_reason_code",
    "created_at",
    "updated_at",
}


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _decode_generation(value: object) -> GenerationSnapshot:
    item = _exact(value, _GENERATION_FIELDS)
    try:
        return GenerationSnapshot(
            generation=_integer(item["generation"]),
            state=GenerationState(_text(item["state"])),
            revision=_integer(item["revision"]),
            previous_job_state=JobState(_text(item["previous_job_state"])),
            start_operation_id=_uuid(item["start_operation_id"]),
            pending_operation_id=_optional_uuid(item["pending_operation_id"]),
            runtime_key_version=_optional_text(item["runtime_key_version"]),
            durable_key_version=_optional_text(item["durable_key_version"]),
            execution_lease_deadline=_timestamp(item["execution_lease_deadline"]),
            barrier_id=_optional_text(item["barrier_id"]),
            receipt_id=_optional_text(item["receipt_id"]),
            release_target=(
                ReleaseTarget(_text(item["release_target"]))
                if item["release_target"] is not None
                else None
            ),
            release_terminal_outcome=(
                TerminalOutcome(_text(item["release_terminal_outcome"]))
                if item["release_terminal_outcome"] is not None
                else None
            ),
            failure_reason_code=_optional_text(item["failure_reason_code"]),
            created_at=_timestamp(item["created_at"]),
            updated_at=_timestamp(item["updated_at"]),
        )
    except (TypeError, ValueError) as error:
        raise RpcProtocolProblem() from error


def _snapshot_dict(value: JobSnapshot) -> dict[str, object]:
    return {
        "job_id": str(value.job_id),
        "conversation_id": str(value.conversation_id),
        "tenant_id": value.owner.tenant_id,
        "workload_subject": value.owner.workload_subject,
        "profile": value.profile,
        "runtime_driver": value.runtime_driver,
        "durable_state_driver": value.durable_state_driver,
        "state": value.state.value,
        "revision": value.revision,
        "generation": value.generation,
        "current_generation": value.current_generation,
        "pending_operation_id": (
            str(value.pending_operation_id) if value.pending_operation_id else None
        ),
        "durable_key_version": value.durable_key_version,
        "terminal_outcome": (
            value.terminal_outcome.value if value.terminal_outcome else None
        ),
        "created_at": value.created_at.isoformat(),
        "updated_at": value.updated_at.isoformat(),
        "generation_record": (
            _generation_dict(value.generation_record)
            if value.generation_record
            else None
        ),
    }


_SNAPSHOT_FIELDS = {
    "job_id",
    "conversation_id",
    "tenant_id",
    "workload_subject",
    "profile",
    "runtime_driver",
    "durable_state_driver",
    "state",
    "revision",
    "generation",
    "current_generation",
    "pending_operation_id",
    "durable_key_version",
    "terminal_outcome",
    "created_at",
    "updated_at",
    "generation_record",
}


def _decode_snapshot(value: object) -> JobSnapshot:
    item = _exact(value, _SNAPSHOT_FIELDS)
    current_generation = item["current_generation"]
    if current_generation is not None:
        current_generation = _integer(current_generation)
    generation_record = item["generation_record"]
    try:
        return JobSnapshot(
            job_id=_uuid(item["job_id"]),
            conversation_id=_uuid(item["conversation_id"]),
            owner=BrokerOwner(
                _text(item["tenant_id"]), _text(item["workload_subject"])
            ),
            profile=_text(item["profile"]),
            runtime_driver=_text(item["runtime_driver"]),
            durable_state_driver=_text(item["durable_state_driver"]),
            state=JobState(_text(item["state"])),
            revision=_integer(item["revision"]),
            generation=_integer(item["generation"]),
            current_generation=current_generation,
            pending_operation_id=_optional_uuid(item["pending_operation_id"]),
            durable_key_version=_optional_text(item["durable_key_version"]),
            terminal_outcome=(
                TerminalOutcome(_text(item["terminal_outcome"]))
                if item["terminal_outcome"] is not None
                else None
            ),
            created_at=_timestamp(item["created_at"]),
            updated_at=_timestamp(item["updated_at"]),
            generation_record=(
                _decode_generation(generation_record)
                if generation_record is not None
                else None
            ),
        )
    except (TypeError, ValueError) as error:
        raise RpcProtocolProblem() from error


def _access_dict(value: RunningGenerationAccess) -> dict[str, object]:
    return {
        "job_id": str(value.job_id),
        "conversation_id": str(value.conversation_id),
        "generation": value.generation,
        "deadline": value.deadline.isoformat(),
        "socket_path": str(value.socket_path),
        "relay_capability": value.relay_capability,
        "session_api_key": value.session_api_key,
    }


def _decode_access(value: object) -> RunningGenerationAccess:
    item = _exact(
        value,
        {
            "job_id",
            "conversation_id",
            "generation",
            "deadline",
            "socket_path",
            "relay_capability",
            "session_api_key",
        },
    )
    try:
        return RunningGenerationAccess(
            job_id=_uuid(item["job_id"]),
            conversation_id=_uuid(item["conversation_id"]),
            generation=_integer(item["generation"]),
            deadline=_timestamp(item["deadline"]),
            socket_path=Path(_text(item["socket_path"])),
            relay_capability=_text(item["relay_capability"]),
            session_api_key=_text(item["session_api_key"]),
        )
    except (TypeError, ValueError) as error:
        raise RpcProtocolProblem() from error


def _response_dict(response: RpcResponse) -> dict[str, object]:
    result = None
    error = None
    if isinstance(response.result, CreateJobResult):
        result = {
            "type": WorkloadIntent.CREATE_JOB.value,
            "ticket": _ticket_dict(response.result.ticket),
            "capability": response.result.capability.value,
        }
    elif isinstance(response.result, InspectJobResult):
        result = {
            "type": WorkloadIntent.INSPECT_JOB.value,
            "snapshot": _snapshot_dict(response.result.snapshot),
            "access": (
                _access_dict(response.result.access)
                if response.result.access is not None
                else None
            ),
        }
    elif isinstance(response.result, StartSegmentResult):
        result = {
            "type": WorkloadIntent.START_SEGMENT.value,
            "operation_id": str(response.result.operation_id),
            "replayed": response.result.replayed,
            "access": _access_dict(response.result.access),
        }
    else:
        assert response.failure is not None
        error = {"code": response.failure.code.value}
    return {
        "version": response.version,
        "request_id": str(response.request_id),
        "ok": response.ok,
        "result": result,
        "error": error,
    }


def encode_response(response: RpcResponse) -> bytes:
    if not isinstance(response, RpcResponse):
        raise TypeError("response must be RpcResponse")
    return _encode_frame(_response_dict(response))


def decode_response(frame: bytes) -> RpcResponse:
    value = _exact(
        _decode_frame(frame),
        {"version", "request_id", "ok", "result", "error"},
    )
    version = _integer(value["version"])
    if version != RPC_VERSION:
        raise RpcProtocolProblem(RpcErrorCode.UNSUPPORTED_VERSION)
    request_id = _uuid(value["request_id"])
    ok = value["ok"]
    if type(ok) is not bool:
        raise RpcProtocolProblem()
    try:
        if ok:
            if value["error"] is not None:
                raise RpcProtocolProblem()
            result = value["result"]
            if not isinstance(result, dict):
                raise RpcProtocolProblem()
            result_type = result.get("type")
            if result_type == WorkloadIntent.CREATE_JOB.value:
                item = _exact(result, {"type", "ticket", "capability"})
                decoded = CreateJobResult(
                    _decode_ticket(item["ticket"]),
                    JobCapability(_text(item["capability"])),
                )
            elif result_type == WorkloadIntent.INSPECT_JOB.value:
                item = _exact(result, {"type", "snapshot", "access"})
                decoded = InspectJobResult(
                    _decode_snapshot(item["snapshot"]),
                    (
                        _decode_access(item["access"])
                        if item["access"] is not None
                        else None
                    ),
                )
            elif result_type == WorkloadIntent.START_SEGMENT.value:
                item = _exact(
                    result,
                    {"type", "operation_id", "replayed", "access"},
                )
                replayed = item["replayed"]
                if type(replayed) is not bool:
                    raise RpcProtocolProblem()
                decoded = StartSegmentResult(
                    _uuid(item["operation_id"]),
                    replayed,
                    _decode_access(item["access"]),
                )
            else:
                raise RpcProtocolProblem()
            return RpcResponse(version, request_id, result=decoded)
        if value["result"] is not None:
            raise RpcProtocolProblem()
        error = _exact(value["error"], {"code"})
        return RpcResponse(
            version,
            request_id,
            failure=RpcFailure(RpcErrorCode(_text(error["code"]))),
        )
    except RpcProtocolProblem:
        raise
    except (TypeError, ValueError) as error:
        raise RpcProtocolProblem() from error
