"""Privileged composition adapters for broker-owned workload generations."""

from .coordinator import BrokerSegmentCoordinator
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
    "DerivedRuntimeSecrets",
    "DurableStateDescriptor",
    "LocalDurableBinding",
    "LocalDurableStateAdapter",
    "LocalDurableStateProblem",
    "RuntimeSecretAuthority",
    "RuntimeSecretProblem",
    "RuntimeSecretRootRing",
]
