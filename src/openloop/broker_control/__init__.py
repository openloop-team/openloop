"""Privileged composition adapters for broker-owned workload generations."""

from .coordinator import BrokerSegmentCoordinator
from .durable import (
    DurableStateDescriptor,
    LocalDurableBinding,
    LocalDurableStateAdapter,
    LocalDurableStateProblem,
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
from .receipts import (
    CheckpointReceiptIssuer,
    CheckpointReceiptKey,
    CheckpointReceiptLocator,
    CheckpointReceiptProblem,
    CheckpointReceiptVerifier,
    receipt_key,
)
from .recovery import (
    RECOVERY_REASON_CODES,
    BrokerLifecycleReconciler,
    RecoveryItemReport,
    RecoveryOutcome,
    RecoveryPassReport,
)
from .secrets import (
    DerivedRuntimeSecrets,
    RuntimeSecretAuthority,
    RuntimeSecretProblem,
    RuntimeSecretRootRing,
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
