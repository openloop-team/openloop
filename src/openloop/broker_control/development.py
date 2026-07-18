"""Explicit development composition helpers for local Docker durable state."""

from __future__ import annotations

from openloop.broker_runtime import DockerRuntimeConfig

from .durable import LocalDurableStateAdapter, LocalDurableStateProblem


def validate_local_durable_binding(
    adapter: LocalDurableStateAdapter,
    config: DockerRuntimeConfig,
) -> None:
    if not isinstance(adapter, LocalDurableStateAdapter):
        raise TypeError("adapter must be LocalDurableStateAdapter")
    if not isinstance(config, DockerRuntimeConfig):
        raise TypeError("config must be DockerRuntimeConfig")
    binding = adapter.binding
    if (
        binding.state_root != config.state_root
        or binding.uid != config.uid
        or binding.gid != config.gid
    ):
        raise LocalDurableStateProblem()


def local_durable_adapter_for_docker(
    config: DockerRuntimeConfig,
) -> LocalDurableStateAdapter:
    if not isinstance(config, DockerRuntimeConfig):
        raise TypeError("config must be DockerRuntimeConfig")
    adapter = LocalDurableStateAdapter(
        state_root=config.state_root,
        uid=config.uid,
        gid=config.gid,
    )
    validate_local_durable_binding(adapter, config)
    return adapter


__all__ = [
    "local_durable_adapter_for_docker",
    "validate_local_durable_binding",
]
