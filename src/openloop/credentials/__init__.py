"""Credential resolution seam — see :mod:`openloop.credentials.resolver`."""

from openloop.credentials.resolver import (
    CredentialError,
    CredentialResolver,
    CredentialScope,
    EnvCredentialResolver,
    GitHubAppResolver,
)

__all__ = [
    "CredentialError",
    "CredentialResolver",
    "CredentialScope",
    "EnvCredentialResolver",
    "GitHubAppResolver",
]
