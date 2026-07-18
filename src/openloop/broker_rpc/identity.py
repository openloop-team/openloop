"""Strict short-lived Ed25519 workload identities for broker control RPC."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from openloop.broker.models import (
    BrokerOwner,
    IsolationMode,
    validate_identifier,
    validate_timestamp,
    validate_uuid,
)


MAX_IDENTITY_TOKEN_BYTES = 8192
MAX_IDENTITY_INTENTS = 16
MAX_IDENTITY_LIFETIME_SECONDS = 300
IDENTITY_CLOCK_SKEW_SECONDS = 30


class WorkloadIntent(str, Enum):
    CREATE_JOB = "CREATE_JOB"
    INSPECT_JOB = "INSPECT_JOB"
    START_SEGMENT = "START_SEGMENT"
    QUIESCE_SEGMENT = "QUIESCE_SEGMENT"
    RELEASE_SEGMENT = "RELEASE_SEGMENT"
    FINALIZE_JOB = "FINALIZE_JOB"


class IdentityProblem(Exception):
    def __init__(self) -> None:
        super().__init__("workload identity rejected")


@dataclass(frozen=True, slots=True)
class WorkloadIdentityToken:
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("identity token must be a string")
        encoded = self.value.encode("ascii", errors="strict")
        if not 1 <= len(encoded) <= MAX_IDENTITY_TOKEN_BYTES:
            raise ValueError("identity token length is invalid")
        if any(not 33 <= byte <= 126 for byte in encoded):
            raise ValueError("identity token must be visible ASCII")


@dataclass(frozen=True, slots=True)
class WorkloadPrincipal:
    owner: BrokerOwner
    worker_instance_id: UUID
    assignment_id: UUID
    isolation_mode: IsolationMode
    required_isolation: IsolationMode
    intents: frozenset[WorkloadIntent]
    key_id: str
    jwt_id: UUID
    issued_at: int
    not_before: int
    expires_at: int


def _timestamp(clock: Callable[[], datetime]) -> int:
    now = clock()
    validate_timestamp("identity clock", now)
    return int(now.astimezone(UTC).timestamp())


def _canonical_uuid(name: str, value: object) -> UUID:
    if not isinstance(value, str):
        raise IdentityProblem()
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as error:
        raise IdentityProblem() from error
    if str(parsed) != value:
        raise IdentityProblem()
    return parsed


def _int_claim(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IdentityProblem()
    return value


def _validated_intents(values: object) -> frozenset[WorkloadIntent]:
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_IDENTITY_INTENTS:
        raise IdentityProblem()
    if any(not isinstance(value, str) for value in values):
        raise IdentityProblem()
    if len(set(values)) != len(values):
        raise IdentityProblem()
    try:
        return frozenset(WorkloadIntent(value) for value in values)
    except ValueError as error:
        raise IdentityProblem() from error


class WorkloadIdentityIssuer:
    def __init__(
        self,
        *,
        private_key: Ed25519PrivateKey,
        key_id: str,
        issuer: str,
        audience: str,
        clock: Callable[[], datetime],
        id_factory: Callable[[], UUID] = uuid4,
        ttl_seconds: int = MAX_IDENTITY_LIFETIME_SECONDS,
    ) -> None:
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("private_key must be Ed25519PrivateKey")
        validate_identifier("key_id", key_id)
        validate_identifier("issuer", issuer)
        validate_identifier("audience", audience)
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or not 1 <= ttl_seconds <= MAX_IDENTITY_LIFETIME_SECONDS
        ):
            raise ValueError("identity lifetime must be 1-300 seconds")
        if not callable(clock) or not callable(id_factory):
            raise TypeError("identity clock and id_factory must be callable")
        self._private_key = private_key
        self._key_id = key_id
        self._issuer = issuer
        self._audience = audience
        self._clock = clock
        self._id_factory = id_factory
        self._ttl_seconds = ttl_seconds

    def issue(
        self,
        *,
        owner: BrokerOwner,
        worker_instance_id: UUID,
        assignment_id: UUID,
        isolation_mode: IsolationMode,
        required_isolation: IsolationMode,
        intents: Iterable[WorkloadIntent],
    ) -> WorkloadIdentityToken:
        if not isinstance(owner, BrokerOwner):
            raise TypeError("owner must be a BrokerOwner")
        validate_uuid("worker_instance_id", worker_instance_id)
        validate_uuid("assignment_id", assignment_id)
        if not isinstance(isolation_mode, IsolationMode) or not isinstance(
            required_isolation, IsolationMode
        ):
            raise TypeError("identity isolation fields must be IsolationMode")
        if not isolation_mode.allows(required_isolation):
            raise ValueError("required isolation exceeds actual placement")
        if not isinstance(intents, (set, frozenset, tuple, list)):
            raise TypeError("intents must be a bounded collection")
        intent_set = frozenset(intents)
        if (
            not 1 <= len(intent_set) <= MAX_IDENTITY_INTENTS
            or any(not isinstance(intent, WorkloadIntent) for intent in intent_set)
        ):
            raise ValueError("identity intents are invalid")
        jwt_id = validate_uuid("jwt_id", self._id_factory())
        issued_at = _timestamp(self._clock)
        payload = {
            "iss": self._issuer,
            "aud": self._audience,
            "sub": owner.workload_subject,
            "tenant_id": owner.tenant_id,
            "worker_instance_id": str(worker_instance_id),
            "assignment_id": str(assignment_id),
            "isolation_mode": isolation_mode.value,
            "required_isolation": required_isolation.value,
            "intents": sorted(intent.value for intent in intent_set),
            "jti": str(jwt_id),
            "iat": issued_at,
            "nbf": issued_at,
            "exp": issued_at + self._ttl_seconds,
        }
        encoded = jwt.encode(
            payload,
            self._private_key,
            algorithm="EdDSA",
            headers={"kid": self._key_id, "typ": "JWT"},
        )
        return WorkloadIdentityToken(encoded)


class WorkloadIdentityVerifier:
    _HEADER_FIELDS = frozenset({"alg", "typ", "kid"})
    _CLAIM_FIELDS = frozenset(
        {
            "iss",
            "aud",
            "sub",
            "tenant_id",
            "worker_instance_id",
            "assignment_id",
            "isolation_mode",
            "required_isolation",
            "intents",
            "jti",
            "iat",
            "nbf",
            "exp",
        }
    )

    def __init__(
        self,
        *,
        public_keys: Mapping[str, Ed25519PublicKey],
        issuer: str,
        audience: str,
        clock: Callable[[], datetime],
    ) -> None:
        validate_identifier("issuer", issuer)
        validate_identifier("audience", audience)
        if not isinstance(public_keys, Mapping) or not public_keys:
            raise ValueError("at least one verification key is required")
        keys: dict[str, Ed25519PublicKey] = {}
        for key_id, key in public_keys.items():
            validate_identifier("key_id", key_id)
            if not isinstance(key, Ed25519PublicKey):
                raise TypeError("verification keys must be Ed25519PublicKey")
            keys[key_id] = key
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._public_keys = keys
        self._issuer = issuer
        self._audience = audience
        self._clock = clock

    def verify(self, token: WorkloadIdentityToken) -> WorkloadPrincipal:
        if not isinstance(token, WorkloadIdentityToken):
            raise TypeError("token must be WorkloadIdentityToken")
        try:
            header = jwt.get_unverified_header(token.value)
            if set(header) != self._HEADER_FIELDS:
                raise IdentityProblem()
            if header["alg"] != "EdDSA" or header["typ"] != "JWT":
                raise IdentityProblem()
            key_id = header["kid"]
            validate_identifier("key_id", key_id)
            key = self._public_keys.get(key_id)
            if key is None:
                raise IdentityProblem()
            claims = jwt.decode(
                token.value,
                key,
                algorithms=["EdDSA"],
                options={
                    "verify_aud": False,
                    "verify_iss": False,
                    "verify_exp": False,
                    "verify_nbf": False,
                    "verify_iat": False,
                },
            )
            if set(claims) != self._CLAIM_FIELDS:
                raise IdentityProblem()
            if claims["iss"] != self._issuer or claims["aud"] != self._audience:
                raise IdentityProblem()
            owner = BrokerOwner(claims["tenant_id"], claims["sub"])
            worker_id = _canonical_uuid(
                "worker_instance_id", claims["worker_instance_id"]
            )
            assignment_id = _canonical_uuid(
                "assignment_id", claims["assignment_id"]
            )
            jwt_id = _canonical_uuid("jti", claims["jti"])
            isolation_mode = IsolationMode(claims["isolation_mode"])
            required_isolation = IsolationMode(claims["required_isolation"])
            if not isolation_mode.allows(required_isolation):
                raise IdentityProblem()
            intents = _validated_intents(claims["intents"])
            issued_at = _int_claim(claims["iat"])
            not_before = _int_claim(claims["nbf"])
            expires_at = _int_claim(claims["exp"])
            now = _timestamp(self._clock)
            if not 1 <= expires_at - issued_at <= MAX_IDENTITY_LIFETIME_SECONDS:
                raise IdentityProblem()
            if not issued_at <= not_before < expires_at:
                raise IdentityProblem()
            if issued_at > now + IDENTITY_CLOCK_SKEW_SECONDS:
                raise IdentityProblem()
            if not_before > now + IDENTITY_CLOCK_SKEW_SECONDS:
                raise IdentityProblem()
            if expires_at <= now - IDENTITY_CLOCK_SKEW_SECONDS:
                raise IdentityProblem()
            return WorkloadPrincipal(
                owner=owner,
                worker_instance_id=worker_id,
                assignment_id=assignment_id,
                isolation_mode=isolation_mode,
                required_isolation=required_isolation,
                intents=intents,
                key_id=key_id,
                jwt_id=jwt_id,
                issued_at=issued_at,
                not_before=not_before,
                expires_at=expires_at,
            )
        except IdentityProblem:
            raise
        except (
            jwt.PyJWTError,
            KeyError,
            TypeError,
            ValueError,
            UnicodeError,
        ) as error:
            raise IdentityProblem() from error
