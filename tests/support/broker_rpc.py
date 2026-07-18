from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openloop.broker.ledger import BrokerLedger
from openloop.broker.memory import InMemoryBrokerRepository
from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker_control.coordinator import BrokerSegmentCoordinator
from openloop.broker_control.durable import LocalDurableStateAdapter
from openloop.broker_control.secrets import (
    RuntimeSecretAuthority,
    RuntimeSecretRootRing,
)
from openloop.broker_rpc.application import BrokerRpcApplication, BrokerRpcPolicy
from openloop.broker_rpc.audit import InMemoryRpcAuditSink
from openloop.broker_rpc.capability import (
    CapabilityRootRing,
    JobCapabilityAuthority,
)
from openloop.broker_rpc.coordinator import (
    SegmentCoordinatorCode,
    SegmentCoordinatorProblem,
)
from openloop.broker_rpc.identity import (
    WorkloadIdentityIssuer,
    WorkloadIdentityVerifier,
    WorkloadIntent,
)
from openloop.broker_runtime.contract import RuntimeDriver
from openloop.broker_runtime.memory import InMemoryRuntimeDriver

from .broker_repository_contract import MutableClock, SequenceIds


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


class DisabledSegmentCoordinator:
    async def start_segment(self, owner, payload):
        raise SegmentCoordinatorProblem(SegmentCoordinatorCode.INTERNAL)

    async def inspect_running_access(self, owner, job_id):
        return None


@dataclass(frozen=True, slots=True)
class BrokerRpcTestFixture:
    application: BrokerRpcApplication
    issuer: WorkloadIdentityIssuer
    repository: InMemoryBrokerRepository
    audit: InMemoryRpcAuditSink
    ledger: BrokerLedger
    coordinator: object
    runtime: RuntimeDriver | None = None
    durable: LocalDurableStateAdapter | None = None

    def identity_provider(
        self,
        owner: BrokerOwner,
        *,
        isolation: IsolationMode = IsolationMode.DEDICATED,
        required: IsolationMode = IsolationMode.SHARED,
    ):
        worker_instance_id = uuid4()
        assignment_id = uuid4()

        def provide(intent: WorkloadIntent):
            return self.issuer.issue(
                owner=owner,
                worker_instance_id=worker_instance_id,
                assignment_id=assignment_id,
                isolation_mode=isolation,
                required_isolation=required,
                intents={intent},
            )

        return provide


def broker_rpc_test_fixture(
    *,
    state_root: Path | None = None,
    runtime_driver: RuntimeDriver | None = None,
) -> BrokerRpcTestFixture:
    if runtime_driver is not None and state_root is None:
        raise ValueError("runtime_driver requires a trusted state_root")
    private_key = Ed25519PrivateKey.generate()
    issuer = WorkloadIdentityIssuer(
        private_key=private_key,
        key_id="issuer-v1",
        issuer="openloop-control",
        audience="openloop:broker-control",
        clock=lambda: NOW,
    )
    verifier = WorkloadIdentityVerifier(
        public_keys={"issuer-v1": private_key.public_key()},
        issuer="openloop-control",
        audience="openloop:broker-control",
        clock=lambda: NOW,
    )
    clock = MutableClock(NOW)
    repository = InMemoryBrokerRepository(clock=clock)
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=5000))
    capability = JobCapabilityAuthority(
        CapabilityRootRing(
            {"cap-v1": bytes(range(32))}, current_version="cap-v1"
        )
    )
    audit = InMemoryRpcAuditSink(clock=lambda: NOW)
    runtime = None
    durable = None
    coordinator: object = DisabledSegmentCoordinator()
    if state_root is not None:
        runtime = runtime_driver or InMemoryRuntimeDriver(clock=clock)
        durable = LocalDurableStateAdapter(
            state_root=state_root,
            uid=os.getuid(),
            gid=os.getgid(),
        )
        coordinator = BrokerSegmentCoordinator(
            ledger=ledger,
            policy=BrokerRpcPolicy("default", "docker", "local", 300),
            runtime_driver=runtime,
            secret_authority=RuntimeSecretAuthority(
                RuntimeSecretRootRing(
                    {"runtime-v1": bytes(range(32))},
                    current_version="runtime-v1",
                )
            ),
            durable_state_adapter=durable,
            clock=clock,
        )
    application = BrokerRpcApplication(
        ledger=ledger,
        identity_verifier=verifier,
        capability_authority=capability,
        audit_sink=audit,
        policy=BrokerRpcPolicy("default", "docker", "local", 300),
        segment_coordinator=coordinator,
    )
    return BrokerRpcTestFixture(
        application,
        issuer,
        repository,
        audit,
        ledger,
        coordinator,
        runtime,
        durable,
    )
