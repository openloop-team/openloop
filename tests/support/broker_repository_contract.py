"""Implementation-neutral fixtures for broker repository contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from openloop.broker.ledger import BrokerLedger
from openloop.broker.models import (
    BrokerOwner,
    ReleaseTarget,
    TerminalOutcome,
    VerifiedCheckpointReceipt,
)


OWNER = BrokerOwner("tenant-contract", "workload-contract")
OTHER_OWNER = BrokerOwner("tenant-other", "workload-other")
CAPABILITY_DIGEST = "a" * 64
DURABLE_DIGEST = "b" * 64


class SequenceIds:
    def __init__(self, start: int = 1) -> None:
        self.next_value = start

    def __call__(self) -> UUID:
        value = UUID(f"00000000-0000-4000-8000-{self.next_value:012d}")
        self.next_value += 1
        return value


class MutableClock:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(2026, 7, 17, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def receipt_for(
    *,
    job_id: UUID,
    conversation_id: UUID,
    generation: int,
    barrier_id: str,
    suffix: str,
) -> VerifiedCheckpointReceipt:
    return VerifiedCheckpointReceipt(
        issuer="checkpoint_issuer",
        receipt_id=f"receipt-{suffix}",
        tenant_id=OWNER.tenant_id,
        job_id=job_id,
        conversation_id=conversation_id,
        generation=generation,
        barrier_id=barrier_id,
        artifact_id=f"artifact-{suffix}",
        base_commit="c" * 40,
        ciphertext_sha256="d" * 64,
        plaintext_sha256="e" * 64,
        byte_count=1024,
        store_version="store-v1",
        envelope_version="envelope-v1",
        key_version="key-v1",
        durable_write_sequence=generation,
    )


async def mark_generation_running(
    ledger: BrokerLedger,
    *,
    job_id: UUID,
    operation_id: UUID,
    generation: int,
):
    return await ledger.mark_running(
        OWNER,
        operation_id,
        job_id,
        generation,
        f"runtime://generation-{generation}",
        f"durable://generation-{generation}",
        f"runtime-key-v{generation}",
        f"durable-key-v{generation}",
        CAPABILITY_DIGEST,
        DURABLE_DIGEST,
    )


async def quiesce_generation(
    ledger: BrokerLedger,
    *,
    job_id: UUID,
    generation: int,
    suffix: str,
):
    barrier_id = f"barrier-{suffix}"
    ticket = await ledger.begin_quiesce(
        OWNER,
        f"contract-quiesce-{suffix}",
        job_id,
        generation,
        barrier_id,
    )
    result = await ledger.mark_quiesced(
        OWNER, ticket.operation_id, job_id, generation
    )
    return ticket, result, barrier_id


@dataclass(frozen=True, slots=True)
class LifecycleTrace:
    job_id: UUID
    conversation_id: UUID
    first_start_operation_id: UUID
    second_start_operation_id: UUID
    terminal_operation_id: UUID
    snapshots: tuple[object, ...]


async def exercise_complete_lifecycle(ledger: BrokerLedger) -> LifecycleTrace:
    snapshots = []
    created = await ledger.create_job(
        OWNER,
        "contract-create-0001",
        "default",
        "docker",
        "postgres",
    )
    snapshots.append(await ledger.inspect_job(OWNER, created.job_id))

    first = await ledger.begin_start(
        OWNER, "contract-start-00001", created.job_id, 0, 30
    )
    snapshots.append(await ledger.inspect_job(OWNER, created.job_id))
    await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=first.operation_id,
        generation=1,
    )
    snapshots.append(await ledger.inspect_job(OWNER, created.job_id))
    _, _, barrier = await quiesce_generation(
        ledger, job_id=created.job_id, generation=1, suffix="0001"
    )
    receipt = receipt_for(
        job_id=created.job_id,
        conversation_id=created.conversation_id,
        generation=1,
        barrier_id=barrier,
        suffix="0001",
    )
    release = await ledger.begin_release(
        OWNER,
        "contract-release-0001",
        created.job_id,
        1,
        receipt,
        ReleaseTarget.PARKED,
    )
    snapshots.append(await ledger.inspect_job(OWNER, created.job_id))
    await ledger.mark_released(OWNER, release.operation_id, created.job_id, 1)
    snapshots.append(await ledger.inspect_job(OWNER, created.job_id))

    second = await ledger.begin_start(
        OWNER, "contract-start-00002", created.job_id, 1, 60
    )
    await mark_generation_running(
        ledger,
        job_id=created.job_id,
        operation_id=second.operation_id,
        generation=2,
    )
    _, _, barrier = await quiesce_generation(
        ledger, job_id=created.job_id, generation=2, suffix="0002"
    )
    receipt = receipt_for(
        job_id=created.job_id,
        conversation_id=created.conversation_id,
        generation=2,
        barrier_id=barrier,
        suffix="0002",
    )
    release = await ledger.begin_release(
        OWNER,
        "contract-release-0002",
        created.job_id,
        2,
        receipt,
        ReleaseTarget.FINALIZING,
        TerminalOutcome.SUCCESS,
    )
    await ledger.mark_released(OWNER, release.operation_id, created.job_id, 2)
    snapshots.append(await ledger.inspect_job(OWNER, created.job_id))
    finalizing = await ledger.begin_finalize(
        OWNER,
        "contract-finalize-01",
        created.job_id,
        2,
        TerminalOutcome.SUCCESS,
    )
    terminal = await ledger.mark_terminal(
        OWNER, finalizing.operation_id, created.job_id
    )
    snapshots.append(await ledger.inspect_job(OWNER, created.job_id))
    return LifecycleTrace(
        job_id=created.job_id,
        conversation_id=created.conversation_id,
        first_start_operation_id=first.operation_id,
        second_start_operation_id=second.operation_id,
        terminal_operation_id=terminal.operation_id,
        snapshots=tuple(snapshots),
    )
