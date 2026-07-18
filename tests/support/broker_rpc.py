from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openloop.broker.ledger import BrokerLedger
from openloop.broker.memory import InMemoryBrokerRepository
from openloop.broker.models import BrokerOwner, IsolationMode
from openloop.broker_rpc.application import BrokerRpcApplication, BrokerRpcPolicy
from openloop.broker_rpc.audit import InMemoryRpcAuditSink
from openloop.broker_rpc.capability import (
    CapabilityRootRing,
    JobCapabilityAuthority,
)
from openloop.broker_rpc.identity import (
    WorkloadIdentityIssuer,
    WorkloadIdentityVerifier,
    WorkloadIntent,
)

from .broker_repository_contract import MutableClock, SequenceIds


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class BrokerRpcTestFixture:
    application: BrokerRpcApplication
    issuer: WorkloadIdentityIssuer
    repository: InMemoryBrokerRepository
    audit: InMemoryRpcAuditSink

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


def broker_rpc_test_fixture() -> BrokerRpcTestFixture:
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
    repository = InMemoryBrokerRepository(clock=MutableClock(NOW))
    ledger = BrokerLedger(repository, id_factory=SequenceIds(start=5000))
    capability = JobCapabilityAuthority(
        CapabilityRootRing(
            {"cap-v1": bytes(range(32))}, current_version="cap-v1"
        )
    )
    audit = InMemoryRpcAuditSink(clock=lambda: NOW)
    application = BrokerRpcApplication(
        ledger=ledger,
        identity_verifier=verifier,
        capability_authority=capability,
        audit_sink=audit,
        policy=BrokerRpcPolicy("default", "docker", "postgres"),
    )
    return BrokerRpcTestFixture(application, issuer, repository, audit)
