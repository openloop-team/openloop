"""Durable-compatibility action-name aliases.

A leaf module (no imports from ``gateway``/``policy``) so that both can
depend on it without a circular import: ``gateway`` already imports
``policy`` (for ``is_allowed``), so ``policy`` importing the alias map back
out of ``gateway`` would be a cycle. Living here, it is imported one-way by
both.

The coding worker's legacy action string resolves to its new home as the
``code`` profile of ``workspace_task`` (Stage 1 migration). Every place an
action name is matched — the allowlist check, approval-record resolution,
and the model-facing tool surface — must treat the legacy and canonical
spellings as interchangeable, so existing agent YAML and in-flight durable
approval records keep working without edits.
"""

from __future__ import annotations

ACTION_ALIASES: dict[str, str] = {
    "coding_worker.pr:write": "workspace_task.code:write",
}


def _canonical_action(action: str) -> str:
    return ACTION_ALIASES.get(action, action)
