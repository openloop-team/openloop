"""The neutral workspace-task contract: gate, profile, and durable task state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Gate(str, Enum):
    """Where a profile's approval boundary sits. A profile's value is a floor;
    agent config may add gating, never remove it."""

    START = "start"
    EFFECT = "effect"  # reserved for mid-task proposed effects (Gate F); unused in Stage 1
    NONE = "none"


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """A profile declaration: its entry action, typed args, gate, and grants."""

    name: str
    entry_action: str
    args_model: type
    gate: Gate
    capabilities: frozenset[str]
