"""Authenticated, capability-scoped application boundary for broker RPC v2."""

from __future__ import annotations

import asyncio
import math
from uuid import UUID

from openloop.broker.errors import IdempotencyConflict, JobNotFound, OwnerMismatch
from openloop.broker.ledger import BrokerLedger

from .audit import (
    AuditDecision,
    AuditReason,
    PeerCredentials,
    RpcAuditRecord,
    RpcAuditSink,
)
from .capability import CapabilityProblem, JobCapability, JobCapabilityAuthority
from .coordinator import (
    BrokerRpcPolicy,
    SegmentCoordinator,
    SegmentCoordinatorCode,
    SegmentCoordinatorProblem,
)
from .errors import RpcErrorCode, RpcFailure
from .identity import (
    IdentityProblem,
    WorkloadIdentityVerifier,
    WorkloadIntent,
    WorkloadPrincipal,
)
from .limits import TokenBucketLimiter
from .models import (
    RPC_VERSION,
    CreateJobPayload,
    CreateJobResult,
    FinalizeJobPayload,
    FinalizeJobResult,
    InspectJobPayload,
    InspectJobResult,
    QuiesceSegmentPayload,
    QuiesceSegmentResult,
    ReleaseSegmentPayload,
    ReleaseSegmentResult,
    RpcRequest,
    RpcResponse,
    RpcResult,
    RunningGenerationAccess,
    StartSegmentPayload,
    StartSegmentResult,
)


_COORDINATOR_FAILURES = {
    SegmentCoordinatorCode.IDEMPOTENCY_CONFLICT: (
        RpcErrorCode.IDEMPOTENCY_CONFLICT,
        AuditDecision.DENIED,
        AuditReason.IDEMPOTENCY_CONFLICT,
    ),
    SegmentCoordinatorCode.STATE_CONFLICT: (
        RpcErrorCode.STATE_CONFLICT,
        AuditDecision.DENIED,
        AuditReason.STATE_CONFLICT,
    ),
    SegmentCoordinatorCode.INVALID_RECEIPT: (
        RpcErrorCode.INVALID_RECEIPT,
        AuditDecision.DENIED,
        AuditReason.INVALID_RECEIPT,
    ),
    SegmentCoordinatorCode.RUNTIME_UNAVAILABLE: (
        RpcErrorCode.RUNTIME_UNAVAILABLE,
        AuditDecision.ERROR,
        AuditReason.RUNTIME_UNAVAILABLE,
    ),
    SegmentCoordinatorCode.DEADLINE_EXCEEDED: (
        RpcErrorCode.DEADLINE_EXCEEDED,
        AuditDecision.DENIED,
        AuditReason.DEADLINE_EXCEEDED,
    ),
    SegmentCoordinatorCode.INTERNAL: (
        RpcErrorCode.INTERNAL,
        AuditDecision.ERROR,
        AuditReason.INTERNAL,
    ),
}


class BrokerRpcApplication:
    def __init__(
        self,
        *,
        ledger: BrokerLedger,
        identity_verifier: WorkloadIdentityVerifier,
        capability_authority: JobCapabilityAuthority,
        audit_sink: RpcAuditSink,
        policy: BrokerRpcPolicy,
        segment_coordinator: SegmentCoordinator,
        principal_limiter: TokenBucketLimiter | None = None,
        audit_timeout_seconds: float = 2.0,
    ) -> None:
        if not isinstance(ledger, BrokerLedger):
            raise TypeError("ledger must be BrokerLedger")
        if not isinstance(identity_verifier, WorkloadIdentityVerifier):
            raise TypeError("identity_verifier must be WorkloadIdentityVerifier")
        if not isinstance(capability_authority, JobCapabilityAuthority):
            raise TypeError("capability_authority must be JobCapabilityAuthority")
        if not isinstance(audit_sink, RpcAuditSink):
            raise TypeError("audit_sink must implement RpcAuditSink")
        if not isinstance(policy, BrokerRpcPolicy):
            raise TypeError("policy must be BrokerRpcPolicy")
        if not isinstance(segment_coordinator, SegmentCoordinator):
            raise TypeError(
                "segment_coordinator must implement SegmentCoordinator"
            )
        if principal_limiter is not None and not isinstance(
            principal_limiter, TokenBucketLimiter
        ):
            raise TypeError("principal_limiter must be TokenBucketLimiter")
        if (
            isinstance(audit_timeout_seconds, bool)
            or not isinstance(audit_timeout_seconds, (int, float))
            or not math.isfinite(float(audit_timeout_seconds))
            or audit_timeout_seconds <= 0
        ):
            raise ValueError("audit_timeout_seconds must be finite and positive")
        self._ledger = ledger
        self._identity_verifier = identity_verifier
        self._capability_authority = capability_authority
        self._audit_sink = audit_sink
        self._policy = policy
        self._segment_coordinator = segment_coordinator
        self._principal_limiter = principal_limiter
        self._audit_timeout_seconds = float(audit_timeout_seconds)

    @staticmethod
    def _response(
        request: RpcRequest,
        *,
        result: RpcResult | None = None,
        failure: RpcErrorCode | None = None,
    ) -> RpcResponse:
        return RpcResponse(
            RPC_VERSION,
            request.request_id,
            result=result,
            failure=RpcFailure(failure) if failure is not None else None,
        )

    async def _audit(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
        decision: AuditDecision,
        reason: AuditReason,
        *,
        job_id: UUID | None = None,
        operation_id: UUID | None = None,
    ) -> bool:
        try:
            async with asyncio.timeout(self._audit_timeout_seconds):
                await self._audit_sink.append(
                    RpcAuditRecord(
                        request_id=request.request_id,
                        method=request.method,
                        decision=decision,
                        reason=reason,
                        peer=peer,
                        principal=principal,
                        job_id=job_id,
                        operation_id=operation_id,
                    )
                )
            return True
        except Exception:
            # Authenticated calls fail closed when their durable audit cannot be
            # recorded. The exception and record contain no bearer credentials.
            return False

    async def _failure(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
        code: RpcErrorCode,
        decision: AuditDecision,
        reason: AuditReason,
        *,
        job_id: UUID | None = None,
        operation_id: UUID | None = None,
    ) -> RpcResponse:
        if not await self._audit(
            request,
            peer,
            principal,
            decision,
            reason,
            job_id=job_id,
            operation_id=operation_id,
        ):
            code = RpcErrorCode.INTERNAL
        return self._response(request, failure=code)

    async def handle(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        *,
        principal_limiter: TokenBucketLimiter | None = None,
    ) -> RpcResponse:
        if not isinstance(request, RpcRequest):
            raise TypeError("request must be RpcRequest")
        if not isinstance(peer, PeerCredentials):
            raise TypeError("peer must be PeerCredentials")
        if principal_limiter is not None and not isinstance(
            principal_limiter, TokenBucketLimiter
        ):
            raise TypeError("principal_limiter must be TokenBucketLimiter")
        try:
            principal = self._identity_verifier.verify(request.identity_token)
        except IdentityProblem:
            # No claims from an unauthenticated token are trusted for durable
            # audit attribution.
            return self._response(request, failure=RpcErrorCode.UNAUTHENTICATED)
        except Exception:
            return self._response(request, failure=RpcErrorCode.INTERNAL)

        limiter = principal_limiter or self._principal_limiter
        if limiter is not None and not await limiter.allow(
            (
                principal.owner.tenant_id,
                principal.owner.workload_subject,
                principal.worker_instance_id,
                principal.assignment_id,
            )
        ):
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.OVERLOADED,
                AuditDecision.DENIED,
                AuditReason.OVERLOADED,
            )

        if request.method not in principal.intents:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.METHOD_NOT_ALLOWED,
                AuditDecision.DENIED,
                AuditReason.MISSING_INTENT,
            )

        if request.method is WorkloadIntent.CREATE_JOB:
            return await self._create_job(request, peer, principal)
        if request.method is WorkloadIntent.START_SEGMENT:
            return await self._start_segment(request, peer, principal)
        if request.method is WorkloadIntent.QUIESCE_SEGMENT:
            return await self._quiesce_segment(request, peer, principal)
        if request.method is WorkloadIntent.RELEASE_SEGMENT:
            return await self._release_segment(request, peer, principal)
        if request.method is WorkloadIntent.FINALIZE_JOB:
            return await self._finalize_job(request, peer, principal)
        if request.method is WorkloadIntent.INSPECT_JOB:
            return await self._inspect_job(request, peer, principal)
        return await self._failure(
            request,
            peer,
            principal,
            RpcErrorCode.METHOD_NOT_ALLOWED,
            AuditDecision.DENIED,
            AuditReason.MISSING_INTENT,
        )

    async def _create_job(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
    ) -> RpcResponse:
        payload = request.payload
        if not isinstance(payload, CreateJobPayload):
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
            )
        ticket = None
        try:
            ticket = await self._ledger.create_authorized_job(
                principal.owner,
                payload.idempotency_key,
                self._policy.profile,
                self._policy.runtime_driver,
                self._policy.durable_state_driver,
                principal.required_isolation,
                self._capability_authority.issue_metadata,
            )
            if ticket.job_id is None:
                raise RuntimeError("CREATE_JOB returned no job ID")
            authorization = await self._ledger.inspect_job_authorization(
                principal.owner, ticket.job_id
            )
            capability = self._capability_authority.derive(
                principal.owner,
                ticket.job_id,
                authorization.authorization,
            )
        except IdempotencyConflict:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.IDEMPOTENCY_CONFLICT,
                AuditDecision.DENIED,
                AuditReason.IDEMPOTENCY_CONFLICT,
            )
        except Exception:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=ticket.job_id if ticket is not None else None,
                operation_id=(
                    ticket.operation_id if ticket is not None else None
                ),
            )

        if not await self._audit(
            request,
            peer,
            principal,
            AuditDecision.ALLOWED,
            AuditReason.ALLOWED,
            job_id=ticket.job_id,
            operation_id=ticket.operation_id,
        ):
            return self._response(request, failure=RpcErrorCode.INTERNAL)
        return self._response(
            request,
            result=CreateJobResult(ticket=ticket, capability=capability),
        )

    async def _start_segment(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
    ) -> RpcResponse:
        payload = request.payload
        capability = request.job_capability
        if not isinstance(payload, StartSegmentPayload) or capability is None:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
            )
        try:
            authorization = await self._ledger.inspect_job_authorization(
                principal.owner, payload.job_id
            )
            if not principal.isolation_mode.allows(
                authorization.minimum_isolation
            ):
                raise PermissionError
            if not self._capability_authority.verify(
                principal.owner,
                payload.job_id,
                authorization.authorization,
                capability,
            ):
                raise PermissionError
        except (JobNotFound, OwnerMismatch, PermissionError):
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                AuditDecision.DENIED,
                AuditReason.NOT_FOUND_OR_UNAUTHORIZED,
                job_id=payload.job_id,
            )
        except CapabilityProblem:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=payload.job_id,
            )
        except Exception:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=payload.job_id,
            )

        try:
            result = await self._segment_coordinator.start_segment(
                principal.owner, payload
            )
            if not isinstance(result, StartSegmentResult):
                raise TypeError("coordinator returned an invalid start result")
        except SegmentCoordinatorProblem as error:
            code, decision, reason = _COORDINATOR_FAILURES[error.code]
            return await self._failure(
                request,
                peer,
                principal,
                code,
                decision,
                reason,
                job_id=payload.job_id,
                operation_id=error.operation_id,
            )
        except Exception:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=payload.job_id,
            )

        if not await self._audit(
            request,
            peer,
            principal,
            AuditDecision.ALLOWED,
            AuditReason.ALLOWED,
            job_id=payload.job_id,
            operation_id=result.operation_id,
        ):
            return self._response(request, failure=RpcErrorCode.INTERNAL)
        return self._response(request, result=result)

    async def _authorize_job(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
        job_id: UUID,
        capability: JobCapability,
    ) -> RpcResponse | None:
        try:
            authorization = await self._ledger.inspect_job_authorization(
                principal.owner, job_id
            )
            if not principal.isolation_mode.allows(
                authorization.minimum_isolation
            ):
                raise PermissionError
            if not self._capability_authority.verify(
                principal.owner,
                job_id,
                authorization.authorization,
                capability,
            ):
                raise PermissionError
        except (JobNotFound, OwnerMismatch, PermissionError):
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                AuditDecision.DENIED,
                AuditReason.NOT_FOUND_OR_UNAUTHORIZED,
                job_id=job_id,
            )
        except Exception:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=job_id,
            )
        return None

    async def _quiesce_segment(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
    ) -> RpcResponse:
        payload = request.payload
        capability = request.job_capability
        if not isinstance(payload, QuiesceSegmentPayload) or not isinstance(
            capability, JobCapability
        ):
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
            )
        denied = await self._authorize_job(
            request, peer, principal, payload.job_id, capability
        )
        if denied is not None:
            return denied
        try:
            result = await self._segment_coordinator.quiesce_segment(
                principal.owner, payload
            )
            if not isinstance(result, QuiesceSegmentResult):
                raise TypeError("coordinator returned an invalid quiesce result")
        except SegmentCoordinatorProblem as error:
            code, decision, reason = _COORDINATOR_FAILURES[error.code]
            return await self._failure(
                request,
                peer,
                principal,
                code,
                decision,
                reason,
                job_id=payload.job_id,
                operation_id=error.operation_id,
            )
        except Exception:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=payload.job_id,
            )
        if not await self._audit(
            request,
            peer,
            principal,
            AuditDecision.ALLOWED,
            AuditReason.ALLOWED,
            job_id=payload.job_id,
            operation_id=result.operation_id,
        ):
            return self._response(request, failure=RpcErrorCode.INTERNAL)
        return self._response(request, result=result)

    async def _release_segment(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
    ) -> RpcResponse:
        payload = request.payload
        capability = request.job_capability
        if not isinstance(payload, ReleaseSegmentPayload) or not isinstance(
            capability, JobCapability
        ):
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
            )
        denied = await self._authorize_job(
            request, peer, principal, payload.job_id, capability
        )
        if denied is not None:
            return denied
        try:
            result = await self._segment_coordinator.release_segment(
                principal.owner, payload
            )
            if not isinstance(result, ReleaseSegmentResult):
                raise TypeError("coordinator returned an invalid release result")
        except SegmentCoordinatorProblem as error:
            code, decision, reason = _COORDINATOR_FAILURES[error.code]
            return await self._failure(
                request,
                peer,
                principal,
                code,
                decision,
                reason,
                job_id=payload.job_id,
                operation_id=error.operation_id,
            )
        except Exception:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=payload.job_id,
            )
        if not await self._audit(
            request,
            peer,
            principal,
            AuditDecision.ALLOWED,
            AuditReason.ALLOWED,
            job_id=payload.job_id,
            operation_id=result.operation_id,
        ):
            return self._response(request, failure=RpcErrorCode.INTERNAL)
        return self._response(request, result=result)

    async def _finalize_job(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
    ) -> RpcResponse:
        payload = request.payload
        capability = request.job_capability
        if not isinstance(payload, FinalizeJobPayload) or not isinstance(
            capability, JobCapability
        ):
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
            )
        denied = await self._authorize_job(
            request, peer, principal, payload.job_id, capability
        )
        if denied is not None:
            return denied
        try:
            result = await self._segment_coordinator.finalize_job(
                principal.owner, payload
            )
            if not isinstance(result, FinalizeJobResult):
                raise TypeError("coordinator returned an invalid finalize result")
        except SegmentCoordinatorProblem as error:
            code, decision, reason = _COORDINATOR_FAILURES[error.code]
            return await self._failure(
                request,
                peer,
                principal,
                code,
                decision,
                reason,
                job_id=payload.job_id,
                operation_id=error.operation_id,
            )
        except Exception:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=payload.job_id,
            )
        if not await self._audit(
            request,
            peer,
            principal,
            AuditDecision.ALLOWED,
            AuditReason.ALLOWED,
            job_id=payload.job_id,
            operation_id=result.operation_id,
        ):
            return self._response(request, failure=RpcErrorCode.INTERNAL)
        return self._response(request, result=result)

    async def _inspect_job(
        self,
        request: RpcRequest,
        peer: PeerCredentials,
        principal: WorkloadPrincipal,
    ) -> RpcResponse:
        payload = request.payload
        capability = request.job_capability
        if not isinstance(payload, InspectJobPayload) or capability is None:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
            )
        try:
            authorization = await self._ledger.inspect_job_authorization(
                principal.owner, payload.job_id
            )
            if not principal.isolation_mode.allows(
                authorization.minimum_isolation
            ):
                raise PermissionError
            if not self._capability_authority.verify(
                principal.owner,
                payload.job_id,
                authorization.authorization,
                capability,
            ):
                raise PermissionError
            snapshot = await self._ledger.inspect_job(
                principal.owner, payload.job_id
            )
            access = await self._segment_coordinator.inspect_running_access(
                principal.owner, payload.job_id
            )
            if access is not None and not isinstance(
                access, RunningGenerationAccess
            ):
                raise TypeError("coordinator returned invalid running access")
        except (JobNotFound, OwnerMismatch, PermissionError):
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                AuditDecision.DENIED,
                AuditReason.NOT_FOUND_OR_UNAUTHORIZED,
                job_id=payload.job_id,
            )
        except CapabilityProblem:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=payload.job_id,
            )
        except Exception:
            return await self._failure(
                request,
                peer,
                principal,
                RpcErrorCode.INTERNAL,
                AuditDecision.ERROR,
                AuditReason.INTERNAL,
                job_id=payload.job_id,
            )

        if not await self._audit(
            request,
            peer,
            principal,
            AuditDecision.ALLOWED,
            AuditReason.ALLOWED,
            job_id=payload.job_id,
        ):
            return self._response(request, failure=RpcErrorCode.INTERNAL)
        return self._response(
            request, result=InspectJobResult(snapshot=snapshot, access=access)
        )
