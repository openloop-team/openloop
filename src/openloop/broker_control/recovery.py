"""Bounded, idempotent recovery of persisted broker lifecycle intent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from openloop.broker.errors import BrokerError
from openloop.broker.ledger import BrokerLedger
from openloop.broker.models import (
    GenerationState,
    JobState,
    RecoveryCandidate,
    RecoverySnapshot,
    ReleaseTarget,
    TerminalOutcome,
    VerifiedCheckpointReceipt,
)
from openloop.broker_rpc.coordinator import (
    SegmentCoordinatorCode,
    SegmentCoordinatorProblem,
)

from .coordinator import BrokerSegmentCoordinator
from .receipts import (
    CheckpointReceiptKey,
    CheckpointReceiptLocator,
    CheckpointReceiptVerifier,
    receipt_key,
)


class RecoveryOutcome(str, Enum):
    REPAIRED = "repaired"
    DEFERRED = "deferred"
    STALE = "stale"
    FAILED_CLOSED = "failed_closed"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


class _ReceiptLookupStatus(Enum):
    MISSING = "missing"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    VALID = "valid"


RECOVERY_REASON_CODES = frozenset(
    {
        "execution_lease_expired",
        "execution_lease_active",
        "quiesce_completed",
        "checkpoint_pending",
        "checkpoint_evidence_invalid",
        "checkpoint_lookup_unavailable",
        "checkpoint_expired",
        "release_completed",
        "finalize_completed",
        "state_changed",
        "recovery_error",
    }
)


@dataclass(frozen=True, slots=True)
class RecoveryItemReport:
    job_id: UUID
    generation: int
    source_job_state: JobState
    source_generation_state: GenerationState | None
    outcome: RecoveryOutcome
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, UUID):
            raise TypeError("job_id must be a UUID")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("generation must be an integer")
        if self.generation < 0:
            raise ValueError("generation must be nonnegative")
        if not isinstance(self.source_job_state, JobState):
            raise TypeError("source_job_state must be a JobState")
        if self.source_generation_state is not None and not isinstance(
            self.source_generation_state, GenerationState
        ):
            raise TypeError(
                "source_generation_state must be a GenerationState or None"
            )
        if not isinstance(self.outcome, RecoveryOutcome):
            raise TypeError("outcome must be a RecoveryOutcome")
        if self.reason_code not in RECOVERY_REASON_CODES:
            raise ValueError("reason_code is not a fixed recovery reason")


@dataclass(frozen=True, slots=True)
class RecoveryPassReport:
    items: tuple[RecoveryItemReport, ...]
    repaired: int
    deferred: int
    stale: int
    failed_closed: int
    error: int

    @classmethod
    def from_items(
        cls, items: tuple[RecoveryItemReport, ...]
    ) -> "RecoveryPassReport":
        if not isinstance(items, tuple) or any(
            not isinstance(item, RecoveryItemReport) for item in items
        ):
            raise TypeError("items must be RecoveryItemReport values")
        counts = {outcome: 0 for outcome in RecoveryOutcome}
        for item in items:
            counts[item.outcome] += 1
        return cls(
            items=items,
            repaired=counts[RecoveryOutcome.REPAIRED],
            deferred=counts[RecoveryOutcome.DEFERRED],
            stale=counts[RecoveryOutcome.STALE],
            failed_closed=counts[RecoveryOutcome.FAILED_CLOSED],
            error=counts[RecoveryOutcome.ERROR],
        )

    @property
    def total(self) -> int:
        return len(self.items)


class BrokerLifecycleReconciler:
    """Finish durable checkpoint lifecycle intent without caller credentials."""

    def __init__(
        self,
        *,
        ledger: BrokerLedger,
        coordinator: BrokerSegmentCoordinator,
        receipt_locator: CheckpointReceiptLocator,
        receipt_verifier: CheckpointReceiptVerifier,
        page_limit: int = 100,
    ) -> None:
        if not isinstance(ledger, BrokerLedger):
            raise TypeError("ledger must be BrokerLedger")
        if not isinstance(coordinator, BrokerSegmentCoordinator):
            raise TypeError("coordinator must be BrokerSegmentCoordinator")
        if not isinstance(receipt_locator, CheckpointReceiptLocator):
            raise TypeError("receipt_locator must implement CheckpointReceiptLocator")
        if not isinstance(receipt_verifier, CheckpointReceiptVerifier):
            raise TypeError("receipt_verifier must be CheckpointReceiptVerifier")
        if isinstance(page_limit, bool) or not isinstance(page_limit, int):
            raise TypeError("page_limit must be an integer")
        if not 1 <= page_limit <= 1000:
            raise ValueError("page_limit must be between 1 and 1000")
        self._ledger = ledger
        self._coordinator = coordinator
        self._locator = receipt_locator
        self._verifier = receipt_verifier
        self._page_limit = page_limit

    async def run_pass(self) -> RecoveryPassReport:
        reports: list[RecoveryItemReport] = []
        cursor: UUID | None = None
        while True:
            candidates = await self._ledger.scan_recovery_candidates(
                cursor, self._page_limit
            )
            if not candidates:
                break
            for candidate in candidates:
                reports.append(await self._recover_one(candidate))
            cursor = candidates[-1].job_id
            if len(candidates) < self._page_limit:
                break
        return RecoveryPassReport.from_items(tuple(reports))

    @staticmethod
    def _report(
        candidate: RecoveryCandidate,
        outcome: RecoveryOutcome,
        reason_code: str,
    ) -> RecoveryItemReport:
        return RecoveryItemReport(
            job_id=candidate.job_id,
            generation=candidate.generation,
            source_job_state=candidate.job_state,
            source_generation_state=candidate.generation_state,
            outcome=outcome,
            reason_code=reason_code,
        )

    async def _recover_one(
        self, candidate: RecoveryCandidate
    ) -> RecoveryItemReport:
        try:
            snapshot = await self._ledger.inspect_job_for_recovery(
                candidate.owner, candidate.job_id
            )
            return await self._dispatch(candidate, snapshot)
        except SegmentCoordinatorProblem as error:
            if error.code not in {
                SegmentCoordinatorCode.STATE_CONFLICT,
                SegmentCoordinatorCode.IDEMPOTENCY_CONFLICT,
            }:
                return self._report(
                    candidate, RecoveryOutcome.ERROR, "recovery_error"
                )
            return await self._after_conflict(candidate)
        except BrokerError:
            return await self._after_conflict(candidate)
        except Exception:
            return self._report(
                candidate, RecoveryOutcome.ERROR, "recovery_error"
            )

    async def _dispatch(
        self, candidate: RecoveryCandidate, snapshot: RecoverySnapshot
    ) -> RecoveryItemReport:
        if snapshot.state is JobState.FINALIZING:
            await self._finish_finalizing(snapshot)
            return self._report(
                candidate, RecoveryOutcome.REPAIRED, "finalize_completed"
            )
        generation = snapshot.generation_record
        if (
            snapshot.state is not JobState.ACTIVE
            or generation is None
            or snapshot.current_generation != generation.generation
        ):
            return self._report(
                candidate, RecoveryOutcome.STALE, "state_changed"
            )
        if generation.state is GenerationState.RUNNING:
            return await self._running(candidate, snapshot)
        if generation.state is GenerationState.QUIESCING:
            return await self._quiescing(candidate, snapshot)
        if generation.state is GenerationState.QUIESCED:
            return await self._quiesced(candidate, snapshot)
        if generation.state is GenerationState.RELEASING:
            return await self._releasing(candidate, snapshot)
        return self._report(candidate, RecoveryOutcome.STALE, "state_changed")

    async def _running(
        self, candidate: RecoveryCandidate, snapshot: RecoverySnapshot
    ) -> RecoveryItemReport:
        generation = snapshot.generation_record
        assert generation is not None
        if candidate.observed_at < generation.execution_lease_deadline:
            return self._report(
                candidate, RecoveryOutcome.DEFERRED, "execution_lease_active"
            )
        await self._coordinator.release_for_recovery(snapshot)
        await self._ledger.abandon_generation(
            snapshot.owner,
            snapshot.job_id,
            generation.generation,
            GenerationState.RUNNING,
            "execution_lease_expired",
            TerminalOutcome.FAILED,
        )
        await self._finish_finalizing(
            await self._ledger.inspect_job_for_recovery(
                snapshot.owner, snapshot.job_id
            )
        )
        return self._report(
            candidate,
            RecoveryOutcome.FAILED_CLOSED,
            "execution_lease_expired",
        )

    async def _quiescing(
        self, candidate: RecoveryCandidate, snapshot: RecoverySnapshot
    ) -> RecoveryItemReport:
        generation = snapshot.generation_record
        assert generation is not None
        if candidate.observed_at < generation.execution_lease_deadline:
            operation_id = generation.pending_operation_id
            if operation_id is None or snapshot.pending_operation_id != operation_id:
                return self._report(
                    candidate, RecoveryOutcome.ERROR, "recovery_error"
                )
            await self._coordinator.quiesce_for_recovery(snapshot)
            await self._ledger.mark_quiesced(
                snapshot.owner,
                operation_id,
                snapshot.job_id,
                generation.generation,
            )
            return self._report(
                candidate, RecoveryOutcome.REPAIRED, "quiesce_completed"
            )
        await self._coordinator.release_for_recovery(snapshot)
        await self._ledger.abandon_generation(
            snapshot.owner,
            snapshot.job_id,
            generation.generation,
            GenerationState.QUIESCING,
            "checkpoint_expired",
            TerminalOutcome.FAILED,
        )
        await self._finish_finalizing(
            await self._ledger.inspect_job_for_recovery(
                snapshot.owner, snapshot.job_id
            )
        )
        return self._report(
            candidate, RecoveryOutcome.FAILED_CLOSED, "checkpoint_expired"
        )

    @staticmethod
    def _receipt_lookup_key(snapshot: RecoverySnapshot) -> CheckpointReceiptKey:
        generation = snapshot.generation_record
        if generation is None or generation.barrier_id is None:
            raise ValueError("checkpoint identity is incomplete")
        return CheckpointReceiptKey(
            tenant_id=snapshot.owner.tenant_id,
            job_id=snapshot.job_id,
            conversation_id=snapshot.conversation_id,
            generation=generation.generation,
            barrier_id=generation.barrier_id,
        )

    async def _find_receipt(
        self, snapshot: RecoverySnapshot
    ) -> tuple[_ReceiptLookupStatus, VerifiedCheckpointReceipt | None]:
        key = self._receipt_lookup_key(snapshot)
        try:
            token = await self._locator.lookup(key)
        except Exception:
            return _ReceiptLookupStatus.UNAVAILABLE, None
        if token is None:
            return _ReceiptLookupStatus.MISSING, None
        try:
            verified = self._verifier.verify(token)
            if receipt_key(verified) != key:
                return _ReceiptLookupStatus.INVALID, None
            return _ReceiptLookupStatus.VALID, verified
        except Exception:
            return _ReceiptLookupStatus.INVALID, None

    async def _quiesced(
        self, candidate: RecoveryCandidate, snapshot: RecoverySnapshot
    ) -> RecoveryItemReport:
        generation = snapshot.generation_record
        assert generation is not None
        status, receipt = await self._find_receipt(snapshot)
        if status is _ReceiptLookupStatus.VALID:
            assert receipt is not None
            await self._park(snapshot, receipt)
            return self._report(
                candidate, RecoveryOutcome.REPAIRED, "release_completed"
            )
        if candidate.observed_at < generation.execution_lease_deadline:
            if status is _ReceiptLookupStatus.MISSING:
                return self._report(
                    candidate, RecoveryOutcome.DEFERRED, "checkpoint_pending"
                )
            if status is _ReceiptLookupStatus.UNAVAILABLE:
                return self._report(
                    candidate,
                    RecoveryOutcome.ERROR,
                    "checkpoint_lookup_unavailable",
                )
            return self._report(
                candidate,
                RecoveryOutcome.ERROR,
                "checkpoint_evidence_invalid",
            )

        await self._coordinator.release_for_recovery(snapshot)
        late_status, late_receipt = await self._find_receipt(snapshot)
        if late_status is _ReceiptLookupStatus.UNAVAILABLE:
            return self._report(
                candidate,
                RecoveryOutcome.ERROR,
                "checkpoint_lookup_unavailable",
            )
        current = await self._ledger.inspect_job_for_recovery(
            snapshot.owner, snapshot.job_id
        )
        current_generation = current.generation_record
        if (
            current.state is not JobState.ACTIVE
            or current_generation is None
            or current_generation.generation != generation.generation
            or current_generation.state is not GenerationState.QUIESCED
        ):
            return self._report(
                candidate, RecoveryOutcome.STALE, "state_changed"
            )
        if late_status is _ReceiptLookupStatus.VALID:
            assert late_receipt is not None
            await self._park(current, late_receipt)
            return self._report(
                candidate, RecoveryOutcome.REPAIRED, "release_completed"
            )
        await self._ledger.abandon_generation(
            current.owner,
            current.job_id,
            current_generation.generation,
            GenerationState.QUIESCED,
            "checkpoint_expired",
            TerminalOutcome.FAILED,
        )
        await self._finish_finalizing(
            await self._ledger.inspect_job_for_recovery(
                current.owner, current.job_id
            )
        )
        return self._report(
            candidate, RecoveryOutcome.FAILED_CLOSED, "checkpoint_expired"
        )

    async def _park(
        self, snapshot: RecoverySnapshot, receipt: VerifiedCheckpointReceipt
    ) -> None:
        generation = snapshot.generation_record
        if generation is None:
            raise ValueError("generation is required")
        ticket = await self._ledger.begin_internal_release(
            snapshot.owner,
            snapshot.job_id,
            generation.generation,
            receipt,
            ReleaseTarget.PARKED,
        )
        releasing = await self._ledger.inspect_job_for_recovery(
            snapshot.owner, snapshot.job_id
        )
        await self._coordinator.release_for_recovery(releasing)
        await self._ledger.mark_released(
            snapshot.owner,
            ticket.operation_id,
            snapshot.job_id,
            generation.generation,
        )

    async def _releasing(
        self, candidate: RecoveryCandidate, snapshot: RecoverySnapshot
    ) -> RecoveryItemReport:
        generation = snapshot.generation_record
        assert generation is not None
        if (
            generation.receipt is None
            or generation.release_target is None
            or generation.pending_operation_id is None
            or snapshot.pending_operation_id != generation.pending_operation_id
            or receipt_key(generation.receipt) != self._receipt_lookup_key(snapshot)
        ):
            return self._report(
                candidate, RecoveryOutcome.ERROR, "recovery_error"
            )
        await self._coordinator.release_for_recovery(snapshot)
        await self._ledger.mark_released(
            snapshot.owner,
            generation.pending_operation_id,
            snapshot.job_id,
            generation.generation,
        )
        return self._report(
            candidate, RecoveryOutcome.REPAIRED, "release_completed"
        )

    async def _finish_finalizing(self, snapshot: RecoverySnapshot) -> None:
        if (
            snapshot.state is not JobState.FINALIZING
            or snapshot.current_generation is not None
            or snapshot.terminal_outcome is None
        ):
            raise ValueError("job is not stably finalizing")
        operation_id = snapshot.pending_operation_id
        if operation_id is None:
            ticket = await self._ledger.begin_internal_finalize(
                snapshot.owner,
                snapshot.job_id,
                snapshot.generation,
                snapshot.terminal_outcome,
            )
            operation_id = ticket.operation_id
        await self._ledger.mark_terminal(
            snapshot.owner, operation_id, snapshot.job_id
        )

    async def _after_conflict(
        self, candidate: RecoveryCandidate
    ) -> RecoveryItemReport:
        try:
            snapshot = await self._ledger.inspect_job_for_recovery(
                candidate.owner, candidate.job_id
            )
        except Exception:
            return self._report(
                candidate, RecoveryOutcome.ERROR, "recovery_error"
            )
        if snapshot.state in {JobState.PARKED, JobState.TERMINAL}:
            return self._report(
                candidate, RecoveryOutcome.REPAIRED, "state_changed"
            )
        return self._report(candidate, RecoveryOutcome.STALE, "state_changed")


__all__ = [
    "BrokerLifecycleReconciler",
    "RECOVERY_REASON_CODES",
    "RecoveryItemReport",
    "RecoveryOutcome",
    "RecoveryPassReport",
]
