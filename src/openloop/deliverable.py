"""Surface-neutral deliverables — the semantic result layer between producers
and surface renderers.

A producer (the session runner today; tool connectors later) describes *what*
the outcome is; a surface's :class:`~openloop.sessions.delivery.SurfaceDelivery`
decides *how* to render it — Slack turns :class:`Prose` into a server-rendered
Markdown message and an :class:`Artifact` into a hosted snippet with a summary
message, another surface may do otherwise. Producers must never encode a
surface dialect (Slack mrkdwn, Block Kit) here: standard Markdown is the
neutral representation for prose, and structure belongs in typed fields.

The union is deliberately small. New variants (e.g. a tool-result card) are
added when a real producer needs them, not speculatively — every variant is a
rendering obligation for every surface.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Prose", "Artifact", "Deliverable", "coerce"]


@dataclass(slots=True)
class Prose:
    """A conversational answer in standard (GitHub-style) Markdown."""

    text: str


@dataclass(slots=True)
class Artifact:
    """Bulk content (diff, log, report) best hosted natively by the surface.

    ``summary`` is short standard-Markdown prose posted conversationally
    alongside the hosted content — it, not the artifact body, is what a reader
    sees in the thread. ``snippet_type`` is a syntax-highlighting hint using
    Slack snippet-type names ("diff", "python", "text"); surfaces without the
    concept ignore it.
    """

    content: str
    title: str
    filename: str
    summary: str
    snippet_type: str | None = None


Deliverable = Prose | Artifact


def coerce(result: "Deliverable | str") -> Deliverable:
    """Widen plain strings (the historical seam type) into :class:`Prose`."""
    if isinstance(result, str):
        return Prose(text=result)
    return result
