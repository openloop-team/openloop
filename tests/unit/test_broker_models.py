from dataclasses import FrozenInstanceError, asdict
from datetime import UTC, datetime, timedelta
from enum import Enum
from uuid import UUID

import pytest

from openloop.broker.errors import (
    ConcurrentMutation,
    IdempotencyConflict,
    InvalidTransition,
    JobNotFound,
    MigrationProblem,
    MigrationVersionError,
    OperationMismatch,
    OwnerMismatch,
    ReceiptField,
    ReceiptBindingMismatch,
    StaleGeneration,
    TransitionEntity,
)
from openloop.broker.models import (
    AuditRecord,
    BrokerOwner,
    CommandKind,
    GenerationRecord,
    GenerationState,
    JobRecord,
    JobState,
    OperationRecord,
    OperationResult,
    OperationSource,
    OperationStatus,
    OperationTicket,
    ReleaseTarget,
    TerminalOutcome,
    VerifiedCheckpointReceipt,
    project_job_snapshot,
    project_recovery_snapshot,
    validate_base_commit,
    validate_bigint,
    validate_idempotency_key,
    validate_identifier,
    validate_lease_seconds,
    validate_opaque_ref,
    validate_positive_bigint,
    validate_sha256,
    validate_token,
)


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
JOB_ID = UUID("00000000-0000-4000-8000-000000000001")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000002")
START_OPERATION_ID = UUID("00000000-0000-4000-8000-000000000003")
PENDING_OPERATION_ID = UUID("00000000-0000-4000-8000-000000000004")
CAPABILITY_DIGEST = "a" * 64
DURABLE_DIGEST = "b" * 64
RUNTIME_REF = "runtime://protected-handle"
DURABLE_REF = "durable://protected-handle"


@pytest.mark.parametrize(
    ("enum_type", "values"),
    [
        (JobState, ["created", "active", "parked", "finalizing", "terminal"]),
        (
            GenerationState,
            [
                "starting",
                "running",
                "quiescing",
                "quiesced",
                "releasing",
                "released",
                "abandoned",
            ],
        ),
        (OperationStatus, ["pending", "completed", "failed"]),
        (OperationSource, ["caller", "internal"]),
        (ReleaseTarget, ["parked", "finalizing"]),
        (TerminalOutcome, ["success", "cancelled", "failed"]),
        (
            CommandKind,
            [
                "create_job",
                "begin_start",
                "mark_running",
                "abandon_generation",
                "begin_quiesce",
                "mark_quiesced",
                "begin_release",
                "mark_released",
                "begin_finalize",
                "mark_terminal",
            ],
        ),
    ],
)
def test_enum_values_are_stable_strings(enum_type: type[Enum], values: list[str]):
    assert [member.value for member in enum_type] == values
    assert all(str(member) == member.value for member in enum_type)


@pytest.mark.parametrize(
    ("tenant_id", "workload_subject"),
    [
        ("t", "w"),
        ("t" * 128, "w" * 256),
        ("é" * 64, "界" * 85),
    ],
)
def test_owner_accepts_exact_utf8_byte_bounds(tenant_id, workload_subject):
    owner = BrokerOwner(tenant_id=tenant_id, workload_subject=workload_subject)
    assert owner.tenant_id == tenant_id
    assert owner.workload_subject == workload_subject


@pytest.mark.parametrize(
    ("tenant_id", "workload_subject"),
    [
        ("", "workload"),
        ("t" * 129, "workload"),
        ("é" * 65, "workload"),
        ("tenant", ""),
        ("tenant", "w" * 257),
        ("tenant", "界" * 86),
        ("tenant\n", "workload"),
        ("tenant", "work\x00load"),
        ("tenant", "work\u0085load"),
    ],
)
def test_owner_rejects_empty_oversize_and_control_values(
    tenant_id, workload_subject
):
    with pytest.raises(ValueError):
        BrokerOwner(tenant_id=tenant_id, workload_subject=workload_subject)


@pytest.mark.parametrize("value", ["x" * 16, "!" * 128, "abcDEF-._~0123456"])
def test_idempotency_key_accepts_visible_ascii_at_bounds(value):
    assert validate_idempotency_key(value) == value


@pytest.mark.parametrize(
    "value",
    ["x" * 15, "x" * 129, "contains space 12", "line\nbreak-value", "é" * 16],
)
def test_idempotency_key_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        validate_idempotency_key(value)


@pytest.mark.parametrize("value", ["a", "runtime-1", "driver_v2", "0" * 64])
def test_token_accepts_lowercase_contract(value):
    assert validate_token("profile", value) == value


@pytest.mark.parametrize(
    "value", ["", "A", "has.dot", "has/slash", "has space", "x" * 65]
)
def test_token_rejects_noncanonical_values(value):
    with pytest.raises(ValueError):
        validate_token("profile", value)


def test_identifier_and_opaque_reference_bounds_are_utf8_bytes():
    assert validate_identifier("barrier", "界" * 85) == "界" * 85
    assert validate_opaque_ref("runtime_ref", "界" * 341) == "界" * 341
    with pytest.raises(ValueError):
        validate_identifier("barrier", "界" * 86)
    with pytest.raises(ValueError):
        validate_opaque_ref("runtime_ref", "界" * 342)
    with pytest.raises(ValueError):
        validate_opaque_ref("runtime_ref", "opaque\x7fhandle")


@pytest.mark.parametrize("value", ["0" * 64, "abcdef0123456789" * 4])
def test_sha256_accepts_exact_lowercase_hex(value):
    assert validate_sha256("digest", value) == value


@pytest.mark.parametrize(
    "value", ["a" * 63, "a" * 65, "A" * 64, "g" * 64, "00" * 31 + "zz"]
)
def test_sha256_rejects_noncanonical_values(value):
    with pytest.raises(ValueError):
        validate_sha256("digest", value)


@pytest.mark.parametrize("value", ["a" * 40, "b" * 64])
def test_base_commit_accepts_sha1_or_sha256(value):
    assert validate_base_commit(value) == value


@pytest.mark.parametrize("value", ["a" * 39, "a" * 41, "A" * 40, "z" * 64])
def test_base_commit_rejects_other_encodings(value):
    with pytest.raises(ValueError):
        validate_base_commit(value)


@pytest.mark.parametrize("value", [0, 1, 2**63 - 1])
def test_bigint_accepts_nonnegative_postgres_range(value):
    assert validate_bigint("count", value) == value


@pytest.mark.parametrize("value", [-1, 2**63, 1.0, True])
def test_bigint_rejects_out_of_range_or_non_integer(value):
    with pytest.raises((TypeError, ValueError)):
        validate_bigint("count", value)


@pytest.mark.parametrize("value", [1, 2**63 - 1])
def test_positive_bigint_accepts_positive_postgres_range(value):
    assert validate_positive_bigint("revision", value) == value


@pytest.mark.parametrize("value", [0, -1, 2**63, 1.0, False])
def test_positive_bigint_rejects_nonpositive_or_non_integer(value):
    with pytest.raises((TypeError, ValueError)):
        validate_positive_bigint("revision", value)


@pytest.mark.parametrize("value", [1, 86_400])
def test_lease_seconds_accepts_inclusive_bounds(value):
    assert validate_lease_seconds(value) == value


@pytest.mark.parametrize("value", [0, 86_401, 1.5, True])
def test_lease_seconds_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError)):
        validate_lease_seconds(value)


def _receipt() -> VerifiedCheckpointReceipt:
    return VerifiedCheckpointReceipt(
        issuer="checkpoint_issuer",
        receipt_id="receipt-0001",
        tenant_id="tenant-a",
        job_id=JOB_ID,
        conversation_id=CONVERSATION_ID,
        generation=1,
        barrier_id="barrier-0001",
        artifact_id="artifact-0001",
        base_commit="c" * 40,
        ciphertext_sha256="d" * 64,
        plaintext_sha256="e" * 64,
        byte_count=1024,
        store_version="store-v1",
        envelope_version="envelope-v1",
        key_version="key-v1",
        durable_write_sequence=7,
    )


def _job() -> JobRecord:
    return JobRecord(
        job_id=JOB_ID,
        conversation_id=CONVERSATION_ID,
        owner=BrokerOwner("tenant-a", "workload-a"),
        profile="default",
        runtime_driver="docker",
        durable_state_driver="postgres",
        state=JobState.ACTIVE,
        revision=3,
        generation=1,
        current_generation=1,
        pending_operation_id=PENDING_OPERATION_ID,
        durable_state_ref=DURABLE_REF,
        durable_key_version="key-v1",
        durable_digest=DURABLE_DIGEST,
        terminal_outcome=None,
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=2),
    )


def _generation() -> GenerationRecord:
    return GenerationRecord(
        job_id=JOB_ID,
        generation=1,
        state=GenerationState.RELEASING,
        revision=4,
        previous_job_state=JobState.CREATED,
        start_operation_id=START_OPERATION_ID,
        pending_operation_id=PENDING_OPERATION_ID,
        runtime_ref=RUNTIME_REF,
        durable_state_ref=DURABLE_REF,
        runtime_key_version="runtime-key-v1",
        durable_key_version="durable-key-v1",
        capability_digest=CAPABILITY_DIGEST,
        durable_digest=DURABLE_DIGEST,
        execution_lease_deadline=NOW + timedelta(minutes=30),
        barrier_id="barrier-0001",
        receipt=_receipt(),
        release_target=ReleaseTarget.PARKED,
        release_terminal_outcome=None,
        failure_reason_code=None,
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=3),
    )


def test_verified_receipt_has_only_bounded_metadata_and_is_frozen():
    receipt = _receipt()
    assert not hasattr(receipt, "signature")
    assert not hasattr(receipt, "body")
    assert not hasattr(receipt, "metadata")
    with pytest.raises(FrozenInstanceError):
        receipt.receipt_id = "changed"  # type: ignore[misc]


def test_records_are_frozen_and_protected_fields_are_repr_redacted():
    job = _job()
    generation = _generation()
    for record in (job, generation):
        with pytest.raises(FrozenInstanceError):
            record.revision = 99  # type: ignore[misc]
    rendered = repr((job, generation))
    assert RUNTIME_REF not in rendered
    assert DURABLE_REF not in rendered
    assert CAPABILITY_DIGEST not in rendered
    assert DURABLE_DIGEST not in rendered
    assert _receipt().plaintext_sha256 not in rendered


def test_public_projection_omits_protected_recovery_values():
    snapshot = project_job_snapshot(_job(), _generation())
    rendered = repr(snapshot)
    serialized = repr(asdict(snapshot))
    for protected in (
        RUNTIME_REF,
        DURABLE_REF,
        CAPABILITY_DIGEST,
        DURABLE_DIGEST,
        _receipt().plaintext_sha256,
        _receipt().ciphertext_sha256,
    ):
        assert protected not in rendered
        assert protected not in serialized
    assert snapshot.job_id == JOB_ID
    assert snapshot.generation_record is not None
    assert snapshot.generation_record.barrier_id == "barrier-0001"
    assert snapshot.generation_record.receipt_id == "receipt-0001"


def test_recovery_projection_retains_trusted_values_but_redacts_repr():
    snapshot = project_recovery_snapshot(_job(), _generation())
    assert snapshot.durable_state_ref == DURABLE_REF
    assert snapshot.durable_digest == DURABLE_DIGEST
    assert snapshot.generation_record is not None
    assert snapshot.generation_record.runtime_ref == RUNTIME_REF
    assert snapshot.generation_record.capability_digest == CAPABILITY_DIGEST
    assert snapshot.generation_record.durable_digest == DURABLE_DIGEST
    assert snapshot.generation_record.receipt == _receipt()
    rendered = repr(snapshot)
    for protected in (RUNTIME_REF, DURABLE_REF, CAPABILITY_DIGEST, DURABLE_DIGEST):
        assert protected not in rendered


def test_operation_and_audit_models_are_immutable_and_safe_to_render():
    owner = BrokerOwner("tenant-a", "workload-a")
    ticket = OperationTicket(
        operation_id=START_OPERATION_ID,
        command=CommandKind.BEGIN_START,
        job_id=JOB_ID,
        conversation_id=CONVERSATION_ID,
        generation=1,
        job_state=JobState.CREATED,
        generation_state=GenerationState.STARTING,
    )
    result = OperationResult(
        operation_id=START_OPERATION_ID,
        command=CommandKind.MARK_RUNNING,
        job_id=JOB_ID,
        generation=1,
        job_state=JobState.ACTIVE,
        generation_state=GenerationState.RUNNING,
    )
    operation = OperationRecord(
        operation_id=START_OPERATION_ID,
        owner=owner,
        source=OperationSource.CALLER,
        idempotency_key="caller-key-00001",
        command=CommandKind.BEGIN_START,
        request_digest="f" * 64,
        job_id=JOB_ID,
        generation=1,
        status=OperationStatus.COMPLETED,
        intent_ticket=ticket,
        completion_result=result,
        created_at=NOW,
        updated_at=NOW,
    )
    audit = AuditRecord(
        sequence=1,
        command=CommandKind.MARK_RUNNING,
        owner=owner,
        job_id=JOB_ID,
        generation=1,
        operation_id=START_OPERATION_ID,
        before_job_state=JobState.CREATED,
        after_job_state=JobState.ACTIVE,
        before_generation_state=GenerationState.STARTING,
        after_generation_state=GenerationState.RUNNING,
        reason_code=None,
        created_at=NOW,
    )
    assert operation.intent_ticket == ticket
    assert operation.completion_result == result
    assert audit.after_job_state is JobState.ACTIVE
    with pytest.raises(FrozenInstanceError):
        operation.status = OperationStatus.FAILED  # type: ignore[misc]


def test_operation_source_controls_idempotency_key_presence():
    base = dict(
        operation_id=START_OPERATION_ID,
        owner=BrokerOwner("tenant-a", "workload-a"),
        command=CommandKind.BEGIN_START,
        request_digest="f" * 64,
        job_id=JOB_ID,
        generation=1,
        status=OperationStatus.PENDING,
        intent_ticket=OperationTicket(
            operation_id=START_OPERATION_ID,
            command=CommandKind.BEGIN_START,
            job_id=JOB_ID,
            generation=1,
            job_state=JobState.CREATED,
            generation_state=GenerationState.STARTING,
        ),
        completion_result=None,
        created_at=NOW,
        updated_at=NOW,
    )
    with pytest.raises(ValueError, match="idempotency"):
        OperationRecord(source=OperationSource.CALLER, idempotency_key=None, **base)
    with pytest.raises(ValueError, match="idempotency"):
        OperationRecord(
            source=OperationSource.INTERNAL,
            idempotency_key="internal-key-0001",
            **base,
        )


def test_naive_timestamps_are_rejected():
    values = _job().__dict__ if hasattr(_job(), "__dict__") else None
    assert values is None  # slotted records do not expose a mutable dict
    with pytest.raises(ValueError, match="timezone"):
        JobRecord(
            job_id=JOB_ID,
            conversation_id=CONVERSATION_ID,
            owner=BrokerOwner("tenant-a", "workload-a"),
            profile="default",
            runtime_driver="docker",
            durable_state_driver="postgres",
            state=JobState.CREATED,
            revision=1,
            generation=0,
            current_generation=None,
            pending_operation_id=None,
            durable_state_ref=None,
            durable_key_version=None,
            durable_digest=None,
            terminal_outcome=None,
            created_at=datetime(2026, 7, 17, 12, 0),
            updated_at=NOW,
        )


@pytest.mark.parametrize(
    "error",
    [
        JobNotFound(JOB_ID),
        OwnerMismatch(JOB_ID),
        StaleGeneration(JOB_ID, expected=1, actual=2),
        InvalidTransition(
            TransitionEntity.JOB, JobState.ACTIVE, CommandKind.BEGIN_START
        ),
        IdempotencyConflict(),
        OperationMismatch(START_OPERATION_ID),
        ReceiptBindingMismatch(ReceiptField.BARRIER_ID),
        ConcurrentMutation(JOB_ID),
        MigrationVersionError(2, MigrationProblem.FUTURE_VERSION),
    ],
)
def test_domain_errors_render_only_safe_identifiers_and_enums(error):
    rendered = f"{error!s} {error!r}"
    for protected in (
        RUNTIME_REF,
        DURABLE_REF,
        CAPABILITY_DIGEST,
        DURABLE_DIGEST,
        _receipt().plaintext_sha256,
        "raw database exception",
    ):
        assert protected not in rendered


def test_domain_errors_reject_free_form_rendered_context():
    with pytest.raises(TypeError):
        ReceiptBindingMismatch(RUNTIME_REF)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        InvalidTransition(  # type: ignore[arg-type]
            RUNTIME_REF, JobState.ACTIVE, CommandKind.BEGIN_START
        )
