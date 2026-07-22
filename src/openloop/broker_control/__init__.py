"""Privileged composition adapters for broker-owned workload generations."""

from .coordinator import BrokerSegmentCoordinator
from .receipts import (
    CheckpointReceiptIssuer,
    CheckpointReceiptKey,
    CheckpointReceiptLocator,
    CheckpointReceiptProblem,
    CheckpointReceiptVerifier,
    receipt_key,
)
from .local_receipts import (
    LocalCheckpointReceiptConflict,
    LocalCheckpointReceiptProblem,
    LocalCheckpointReceiptStore,
    ReadOnlyCheckpointReceiptLocator,
    canonical_checkpoint_key_json,
    checkpoint_artifact_identity,
    checkpoint_digest,
)
from .durable import (
    DurableStateDescriptor,
    LocalDurableBinding,
    LocalDurableStateAdapter,
    LocalDurableStateProblem,
)
from .secrets import (
    DerivedRuntimeSecrets,
    RuntimeSecretAuthority,
    RuntimeSecretProblem,
    RuntimeSecretRootRing,
)
from .recovery import (
    BrokerLifecycleReconciler,
    RECOVERY_REASON_CODES,
    RecoveryItemReport,
    RecoveryOutcome,
    RecoveryPassReport,
)
from .workspace_ingress import (
    LocalWorkspaceIngress,
    StagedWorkspace,
    WorkspaceIngressProblem,
)

__all__ = [
    "BrokerLifecycleReconciler",
    "BrokerSegmentCoordinator",
    "CheckpointReceiptIssuer",
    "CheckpointReceiptKey",
    "CheckpointReceiptLocator",
    "CheckpointReceiptProblem",
    "CheckpointReceiptVerifier",
    "LocalCheckpointReceiptConflict",
    "LocalCheckpointReceiptProblem",
    "LocalCheckpointReceiptStore",
    "ReadOnlyCheckpointReceiptLocator",
    "canonical_checkpoint_key_json",
    "checkpoint_artifact_identity",
    "checkpoint_digest",
    "receipt_key",
    "DerivedRuntimeSecrets",
    "DurableStateDescriptor",
    "LocalDurableBinding",
    "LocalDurableStateAdapter",
    "LocalDurableStateProblem",
    "RuntimeSecretAuthority",
    "RuntimeSecretProblem",
    "RuntimeSecretRootRing",
    "RECOVERY_REASON_CODES",
    "RecoveryItemReport",
    "RecoveryOutcome",
    "RecoveryPassReport",
    "LocalWorkspaceIngress",
    "StagedWorkspace",
    "WorkspaceIngressProblem",
]
