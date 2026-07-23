"""Read-only repository investigation profile (roadmap Stage 1, profile #2).

Mirrors the builtin coding worker's model+context seam, but read-only: it asks
the model to answer a question over a repository snapshot and returns an
EvidenceBundle (findings + provenance). It applies no diff, opens no PR, and
needs no sandbox — the outcome is an answer, never an effect."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openloop.tasks.outcomes import EvidenceBundle

INVESTIGATE_ARGS_VERSION = 1


class InvestigateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(min_length=1, description="owner/repo to investigate")
    question: str = Field(min_length=1, description="the question to answer")
    ref: str | None = Field(default=None, description="branch/ref (default main)")

    @field_validator("repo", "question", mode="before")
    @classmethod
    def _strip(cls, value):
        return value.strip() if isinstance(value, str) else value


def _parse_findings(text: str) -> tuple[str, str]:
    """Split the model reply into (summary, findings). Tolerant of missing tags."""
    summary, findings = "", text.strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.upper().startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif line.upper().startswith("FINDINGS:"):
            findings = "\n".join(lines[i + 1 :]).strip()
            break
    return summary or "Investigation complete.", findings


class RepoInvestigator:
    def __init__(self, model: str, *, gateway=None, max_context_bytes: int = 60_000):
        self.model = model
        self._gateway = gateway
        self.max_context_bytes = max_context_bytes

    def _completer(self):
        if self._gateway is None:
            from openloop.models.gateway import ModelGateway

            self._gateway = ModelGateway()
        return self._gateway

    async def investigate(self, workspace: Path, question: str, repo: str):
        context = self._repo_context(workspace)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a read-only code investigator. Answer the question "
                    "using the repository snapshot. Cite evidence as `path:line`. "
                    "Respond with exactly:\nSUMMARY: <one line>\nFINDINGS:\n<markdown "
                    "bullets, each citing path:line>. Do not propose code changes."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nRepository {repo}:\n{context}"
                ),
            },
        ]
        resp = await self._completer().complete(self.model, messages)
        summary, findings = _parse_findings(resp.text)
        return EvidenceBundle(summary=summary, findings=findings), resp

    def _repo_context(self, workspace: Path) -> str:
        parts: list[str] = []
        budget = self.max_context_bytes
        for path in sorted(workspace.rglob("*")):
            if ".git" in path.parts or not path.is_file():
                continue
            try:
                text = path.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.relative_to(workspace)
            chunk = f"\n=== {rel} ===\n{text}"
            if len(chunk) > budget:
                break
            budget -= len(chunk)
            parts.append(chunk)
        return "".join(parts)
