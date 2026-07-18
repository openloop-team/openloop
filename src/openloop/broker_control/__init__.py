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

__all__ = [
    "BrokerSegmentCoordinator",
    "CheckpointReceiptIssuer",
    "CheckpointReceiptKey",
    "CheckpointReceiptLocator",
    "CheckpointReceiptProblem",
    "CheckpointReceiptVerifier",
    "receipt_key",
    "DerivedRuntimeSecrets",
    "DurableStateDescriptor",
    "LocalDurableBinding",
    "LocalDurableStateAdapter",
    "LocalDurableStateProblem",
    "RuntimeSecretAuthority",
    "RuntimeSecretProblem",
    "RuntimeSecretRootRing",
]
