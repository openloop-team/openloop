"""Typed task outcomes and their mapping onto surface-neutral deliverables.

The union is a property of the task core, never derived from the entry action
(design decision 2): an EvidenceBundle and a PullRequest are peers, so a future
task can end with an outcome its entry profile did not start with."""

from __future__ import annotations

from dataclasses import dataclass

from openloop.deliverable import Artifact, Deliverable, Prose


@dataclass(slots=True)
class Diagnosis:
    text: str


@dataclass(slots=True)
class EvidenceBundle:
    summary: str
    findings: str
    title: str = "Investigation findings"
    filename: str = "findings.md"


@dataclass(slots=True)
class PullRequest:
    repo: str
    branch: str
    pr_number: int | None
    pr_url: str | None
    summary: str


@dataclass(slots=True)
class Failed:
    status: str
    error: str


TaskOutcome = Diagnosis | EvidenceBundle | PullRequest | Failed


def to_deliverable(outcome: TaskOutcome) -> Deliverable:
    """Map a typed outcome onto Prose | Artifact. No new Deliverable variant."""
    if isinstance(outcome, Diagnosis):
        return Prose(text=outcome.text)
    if isinstance(outcome, EvidenceBundle):
        return Artifact(
            content=outcome.findings,
            title=outcome.title,
            filename=outcome.filename,
            summary=outcome.summary,
            snippet_type="markdown",
        )
    if isinstance(outcome, PullRequest):
        return Prose(text=outcome.summary)
    if isinstance(outcome, Failed):
        return Prose(text=f"The task failed ({outcome.status}): {outcome.error}")
    raise TypeError(f"unknown outcome type {type(outcome).__name__}")
