"""Immutable role configuration derived at process composition boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import SecretStr

from openloop.config import (
    DEFAULT_EXTERNAL_BROKER_CHECKPOINT_RECEIPT_ROOT,
    DEFAULT_EXTERNAL_BROKER_CONTAINER_ROOT,
    DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR,
    DEFAULT_EXTERNAL_BROKER_INGRESS_ROOT,
    DEFAULT_EXTERNAL_BROKER_RUNTIME_ROOT,
    DEFAULT_EXTERNAL_BROKER_STATE_ROOT,
    BrokerSettings,
    CoprocessBrokerSettings,
    RuntimeSettings,
)

BrokerMode = Literal["coprocess", "external"]


__all__ = [
    "BrokerClientConfig",
    "BrokerMode",
    "BrokerServiceConfig",
    "DEFAULT_EXTERNAL_BROKER_CHECKPOINT_RECEIPT_ROOT",
    "DEFAULT_EXTERNAL_BROKER_CONTAINER_ROOT",
    "DEFAULT_EXTERNAL_BROKER_CONTROL_SOCKET_DIR",
    "DEFAULT_EXTERNAL_BROKER_INGRESS_ROOT",
    "DEFAULT_EXTERNAL_BROKER_RUNTIME_ROOT",
    "DEFAULT_EXTERNAL_BROKER_STATE_ROOT",
]


def _required_path(
    name: str,
    value: str | None,
) -> Path:
    if not value:
        raise ValueError(f"{name} is required when the broker is enabled")
    return Path(value)


@dataclass(frozen=True, slots=True)
class BrokerClientConfig:
    """The application-side subset needed to construct a broker client."""

    mode: BrokerMode
    control_socket_dir: Path
    ingress_root: Path
    checkpoint_receipt_root: Path | None
    shared_data_gid: int | None
    identity_private_key: SecretStr | None = field(repr=False)
    identity_key_id: str
    identity_issuer: str
    identity_audience: str
    receipt_roots: Mapping[str, SecretStr] = field(repr=False)
    receipt_current_version: str
    receipt_domain: str

    @classmethod
    def from_runtime_settings(
        cls,
        settings: RuntimeSettings,
        *,
        coprocess_settings: CoprocessBrokerSettings | None = None,
    ) -> BrokerClientConfig:
        if not isinstance(settings, RuntimeSettings):
            raise TypeError("settings must be RuntimeSettings")
        external = settings.broker_mode == "external"
        if external:
            control_socket_dir = _required_path(
                "broker_control_socket_dir",
                settings.broker_control_socket_dir,
            )
            ingress_root = _required_path(
                "broker_ingress_root",
                settings.broker_ingress_root,
            )
            checkpoint_receipt_root = _required_path(
                "broker_checkpoint_receipt_root",
                settings.broker_checkpoint_receipt_root,
            )
            shared_data_gid = settings.broker_shared_data_gid
            missing = [
                name
                for name, value in (
                    (
                        "broker_identity_private_key",
                        settings.broker_identity_private_key,
                    ),
                    ("broker_shared_data_gid", settings.broker_shared_data_gid),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"external broker mode requires {', '.join(missing)} to be set"
                )
        else:
            if not isinstance(coprocess_settings, CoprocessBrokerSettings):
                raise ValueError(
                    "coprocess broker mode requires CoprocessBrokerSettings"
                )
            control_socket_dir = _required_path(
                "broker_control_socket_dir",
                coprocess_settings.broker_control_socket_dir,
            )
            ingress_root = (
                _required_path(
                    "broker_runtime_root",
                    coprocess_settings.broker_runtime_root,
                )
                / ".workspace-ingress"
            )
            checkpoint_receipt_root = None
            shared_data_gid = coprocess_settings.broker_shared_data_gid
            if (
                coprocess_settings.broker_identity_issuer
                != settings.broker_identity_issuer
                or coprocess_settings.broker_identity_audience
                != settings.broker_identity_audience
                or coprocess_settings.broker_receipt_domain
                != settings.broker_receipt_domain
            ):
                raise ValueError(
                    "coprocess runtime and broker trust-domain settings must match"
                )

        if checkpoint_receipt_root is not None and shared_data_gid is None:
            raise ValueError(
                "broker_shared_data_gid is required when "
                "broker_checkpoint_receipt_root is set"
            )

        return cls(
            mode=settings.broker_mode,
            control_socket_dir=control_socket_dir,
            ingress_root=ingress_root,
            checkpoint_receipt_root=checkpoint_receipt_root,
            shared_data_gid=shared_data_gid,
            identity_private_key=settings.broker_identity_private_key,
            identity_key_id=settings.broker_identity_key_id,
            identity_issuer=settings.broker_identity_issuer,
            identity_audience=settings.broker_identity_audience,
            receipt_roots=MappingProxyType(dict(settings.broker_receipt_roots)),
            receipt_current_version=settings.broker_receipt_current_version,
            receipt_domain=settings.broker_receipt_domain,
        )

    @property
    def external(self) -> bool:
        return self.mode == "external"


@dataclass(frozen=True, slots=True)
class BrokerServiceConfig:
    """The broker-service subset needed to construct the privileged graph."""

    mode: BrokerMode
    control_socket_dir: Path
    state_root: Path
    runtime_root: Path
    ingress_root: Path
    checkpoint_receipt_root: Path | None
    shared_data_gid: int | None
    expected_app_uid: int | None
    capability_roots: Mapping[str, SecretStr] = field(repr=False)
    capability_current_version: str
    runtime_roots: Mapping[str, SecretStr] = field(repr=False)
    runtime_current_version: str
    identity_public_keys: Mapping[str, str]
    receipt_public_keys: Mapping[str, str]
    receipt_domain: str
    identity_issuer: str
    identity_audience: str
    execution_lease_seconds: int
    generation_deadline_seconds: int

    @classmethod
    def from_broker_settings(cls, settings: BrokerSettings) -> BrokerServiceConfig:
        if not isinstance(settings, BrokerSettings):
            raise TypeError("settings must be BrokerSettings")
        runtime_root = _required_path(
            "broker_runtime_root",
            settings.broker_runtime_root,
        )
        control_socket_dir = _required_path(
            "broker_control_socket_dir",
            settings.broker_control_socket_dir,
        )
        ingress_root = _required_path(
            "broker_ingress_root",
            settings.broker_ingress_root,
        )
        checkpoint_receipt_root = _required_path(
            "broker_checkpoint_receipt_root",
            settings.broker_checkpoint_receipt_root,
        )
        missing = [
            name
            for name, value in (
                ("broker_shared_data_gid", settings.broker_shared_data_gid),
                ("broker_expected_app_uid", settings.broker_expected_app_uid),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"external broker requires {', '.join(missing)} to be set")

        return cls(
            mode="external",
            control_socket_dir=control_socket_dir,
            state_root=_required_path(
                "broker_state_root",
                settings.broker_state_root,
            ),
            runtime_root=runtime_root,
            ingress_root=ingress_root,
            checkpoint_receipt_root=checkpoint_receipt_root,
            shared_data_gid=settings.broker_shared_data_gid,
            expected_app_uid=settings.broker_expected_app_uid,
            capability_roots=MappingProxyType(dict(settings.broker_capability_roots)),
            capability_current_version=settings.broker_capability_current_version,
            runtime_roots=MappingProxyType(dict(settings.broker_runtime_roots)),
            runtime_current_version=settings.broker_runtime_current_version,
            identity_public_keys=MappingProxyType(
                dict(settings.broker_identity_public_keys)
            ),
            receipt_public_keys=MappingProxyType(
                dict(settings.broker_receipt_public_keys)
            ),
            receipt_domain=settings.broker_receipt_domain,
            identity_issuer=settings.broker_identity_issuer,
            identity_audience=settings.broker_identity_audience,
            execution_lease_seconds=settings.broker_execution_lease_seconds,
            generation_deadline_seconds=(settings.broker_generation_deadline_seconds),
        )

    @classmethod
    def from_coprocess_settings(
        cls, settings: CoprocessBrokerSettings
    ) -> BrokerServiceConfig:
        if not isinstance(settings, CoprocessBrokerSettings):
            raise TypeError("settings must be CoprocessBrokerSettings")
        runtime_root = _required_path(
            "broker_runtime_root",
            settings.broker_runtime_root,
        )
        return cls(
            mode="coprocess",
            control_socket_dir=_required_path(
                "broker_control_socket_dir",
                settings.broker_control_socket_dir,
            ),
            state_root=_required_path(
                "broker_state_root",
                settings.broker_state_root,
            ),
            runtime_root=runtime_root,
            ingress_root=runtime_root / ".workspace-ingress",
            checkpoint_receipt_root=None,
            shared_data_gid=settings.broker_shared_data_gid,
            expected_app_uid=None,
            capability_roots=MappingProxyType(dict(settings.broker_capability_roots)),
            capability_current_version=settings.broker_capability_current_version,
            runtime_roots=MappingProxyType(dict(settings.broker_runtime_roots)),
            runtime_current_version=settings.broker_runtime_current_version,
            identity_public_keys=MappingProxyType({}),
            receipt_public_keys=MappingProxyType({}),
            receipt_domain=settings.broker_receipt_domain,
            identity_issuer=settings.broker_identity_issuer,
            identity_audience=settings.broker_identity_audience,
            execution_lease_seconds=settings.broker_execution_lease_seconds,
            generation_deadline_seconds=(settings.broker_generation_deadline_seconds),
        )

    @property
    def external(self) -> bool:
        return self.mode == "external"
