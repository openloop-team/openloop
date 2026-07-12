"""Core tool types: the Tool protocol, results, and invocation outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from openloop.approvals.store import ApprovalRequest


@dataclass(slots=True)
class ToolResult:
    ok: bool
    summary: str
    data: dict = field(default_factory=dict)


@dataclass(slots=True)
class ActionSpec:
    """Describes an action to the model: a description + JSON-schema args.

    A typed action (docs/typed-tool-args.md §3) additionally names the pydantic
    ``model`` its ``parameters`` were GENERATED from — the gateway then parses
    raw args through it before anything durable exists — and the integer
    ``version`` of that args contract, stamped onto durable records so a
    consumer can refuse a record written under an older contract. Both stay
    ``None`` for untyped actions (MCP passthrough), which keep the permissive
    subset validation of :func:`validate_args`.
    """

    description: str
    parameters: dict
    model: type | None = None
    version: int | None = None


@dataclass(slots=True)
class Invocation:
    """Outcome of asking the gateway to run an action."""

    # executed | started | pending_approval | forbidden | denied | invalid
    # ("invalid" = the args failed the action's declared schema; nothing was
    # persisted or executed — the caller/model can correct the args and retry)
    status: str
    result: ToolResult | None = None
    approval: ApprovalRequest | None = None
    message: str | None = None


@runtime_checkable
class Tool(Protocol):
    """A native or MCP-backed connector exposing permissioned actions."""

    name: str

    def supported_permissions(self) -> set[str]: ...

    def describe(self, permission: str) -> ActionSpec: ...

    async def execute(self, permission: str, args: dict) -> ToolResult: ...


def split_action(action: str) -> tuple[str, str]:
    """Split ``github.issues:write`` into ``("github", "issues:write")``."""
    tool, _, permission = action.partition(".")
    if not tool or not permission:
        raise ValueError(f"malformed action {action!r}; expected '<tool>.<perm>'")
    return tool, permission


def format_validation_error(exc) -> str:
    """One human/model-readable line from a pydantic ``ValidationError``.

    Keeps the field path so the model can correct the exact argument, and drops
    pydantic's ``url``/input echoes — raw input values must not round-trip into
    an error message that gets persisted or shown to an approver.
    """
    problems = []
    for error in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(part) for part in error.get("loc", ()))
        msg = error.get("msg", "invalid value")
        problems.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(problems) or "invalid arguments"


# The JSON-schema primitive types our validator understands. Anything else
# (unknown type names, anyOf, $ref, …) is deliberately not enforced.
_SCHEMA_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _matches_type(value, expected: str) -> bool:
    py_type = _SCHEMA_TYPES.get(expected)
    if py_type is None:
        return True  # a type name outside the subset — never stricter than declared
    if expected in ("integer", "number") and isinstance(value, bool):
        return False  # bool is an int subclass in Python; JSON schema disagrees
    return isinstance(value, py_type)


def validate_args(parameters: dict, args: dict) -> list[str]:
    """Check ``args`` against an action's declared parameter schema.

    The gateway calls this once per invocation, after ``prepare_args``
    normalization and before anything is persisted or executed, so a tool's
    declared contract (:class:`ActionSpec.parameters`) is actually enforced on
    every path — including workflow-backed tools whose ``execute()`` is never
    called. Returns human-readable problems (empty = valid).

    Deliberately a narrow subset of JSON schema: ``required`` membership,
    primitive ``type`` checks, and string ``minLength``. Constructs outside the
    subset (``anyOf``, ``$ref``, unknown type names, MCP servers' richer
    schemas) are ignored — enforcement must never be stricter than what this
    validator provably understands, so unknown shapes degrade to permissive.
    """
    problems: list[str] = []
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        return problems
    required = parameters.get("required")
    for name in required if isinstance(required, list) else []:
        if isinstance(name, str) and name not in args:
            problems.append(f"missing required argument {name!r}")
    properties = parameters.get("properties")
    for name, spec in (properties if isinstance(properties, dict) else {}).items():
        if name not in args or not isinstance(spec, dict):
            continue
        value = args[name]
        expected = spec.get("type")
        if isinstance(expected, str) and not _matches_type(value, expected):
            problems.append(f"argument {name!r} must be of type {expected}")
            continue
        min_length = spec.get("minLength")
        if (
            isinstance(min_length, int)
            and isinstance(value, str)
            and len(value) < min_length
        ):
            problems.append(
                f"argument {name!r} must not be empty"
                if min_length == 1
                else f"argument {name!r} must be at least {min_length} characters"
            )
    return problems
