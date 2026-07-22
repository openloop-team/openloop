"""Minimal `openloop` CLI.

First slice supports validating/inspecting agent config-as-code. `apply` simply
validates for now — registering against a live control plane lands with the
durable runtime (see roadmap).
"""

from __future__ import annotations

import argparse
import sys

from openloop.agents import load_agent
from openloop.agents.loader import AgentConfigError


def _cmd_agents_apply(args: argparse.Namespace) -> int:
    try:
        agent = load_agent(args.file)
    except AgentConfigError as exc:
        # `apply` stays read-only; when the failure is a missing/invalid id,
        # point at the command that fixes it instead of a validation dump.
        if _lacks_issued_identity(args.file):
            print(
                f"error: {args.file} has no issued identity — run "
                f"openloop agents id issue -f {args.file}",
                file=sys.stderr,
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    surfaces = ", ".join(s.type for s in agent.spec.surfaces) or "none"
    print(
        f"ok: agent {agent.metadata.name!r} "
        f"(workspace {agent.metadata.workspace!r}) validated — "
        f"surfaces: {surfaces}; default model: {agent.spec.model_policy.default}"
    )
    print("note: apply currently validates only; live registration is WIP.")
    return 0


def _lacks_issued_identity(file: str) -> bool:
    """Whether a strict-load failure is explained by a missing/invalid id."""
    import re
    from pathlib import Path

    import yaml

    try:
        raw = yaml.safe_load(Path(file).read_text())
    except (OSError, yaml.YAMLError):
        return False
    meta = raw.get("metadata") if isinstance(raw, dict) else None
    if not isinstance(meta, dict):
        return False
    return not (
        isinstance(meta.get("id"), str)
        and re.match(r"^[0-9a-f]{32}$", meta["id"])
    )


def _cmd_agents_id_issue(args: argparse.Namespace) -> int:
    """Issue a durable identity (minted ``uuid4().hex``) into an agent YAML.

    Authoring stays config-as-code: this command only stamps identity into an
    existing hand-authored file, with a surgical text insert that preserves
    comments and formatting. A present, well-formed, unique id is kept
    untouched (idempotent); duplicates and malformed ids refuse. The file's
    directory is the uniqueness authority — the loader's duplicate-id check
    at load time is the guarantee, this is the friendlier early gate.
    """
    import re
    import uuid
    from pathlib import Path

    import yaml

    id_pattern = re.compile(r"^[0-9a-f]{32}$")
    path = Path(args.file)
    try:
        text = path.read_text()
        raw = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as exc:
        print(f"error: {path}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(raw, dict) or not isinstance(raw.get("metadata"), dict):
        print(
            f"error: {path}: expected a YAML mapping with a metadata mapping",
            file=sys.stderr,
        )
        return 1
    metadata = raw["metadata"]
    name = metadata.get("name")

    # Sibling files (raw-read; unparseable ones are the loader's problem).
    sibling_ids: dict[str, Path] = {}
    sibling_names: dict[str, Path] = {}
    siblings = {*path.parent.glob("*.yaml"), *path.parent.glob("*.yml")}
    for sibling in sorted(siblings):
        if sibling.resolve() == path.resolve():
            continue
        try:
            other = yaml.safe_load(sibling.read_text())
        except (OSError, yaml.YAMLError):
            continue
        meta = other.get("metadata") if isinstance(other, dict) else None
        if not isinstance(meta, dict):
            continue
        if isinstance(meta.get("id"), str):
            sibling_ids.setdefault(meta["id"], sibling)
        if isinstance(meta.get("name"), str):
            sibling_names.setdefault(meta["name"], sibling)

    existing = metadata.get("id")
    if existing is not None:
        existing = str(existing)
        if not id_pattern.match(existing):
            print(
                f"error: {path}: existing id is not a valid identity; "
                "remove it to re-issue",
                file=sys.stderr,
            )
            return 1
        if existing in sibling_ids:
            print(
                f"error: id {existing} already used by "
                f"{sibling_ids[existing].name} — remove the line to mint a "
                "fresh one",
                file=sys.stderr,
            )
            return 1
        print(f"agent {name} already has identity {existing}")
        return 0

    if name in sibling_names:
        print(
            f"error: duplicate agent name {name!r} "
            f"(also {sibling_names[name]})",
            file=sys.stderr,
        )
        return 1

    minted = uuid.uuid4().hex
    inserted = _insert_metadata_id(text, minted)
    if inserted is None:
        print(
            f"error: {path}: could not find the metadata block to insert "
            "the id into",
            file=sys.stderr,
        )
        return 1
    new_text, preview = inserted

    print(f"{path} has no durable identity. I'll insert this line under metadata:")
    print()
    print(preview)
    print()
    print(
        "This becomes the agent's permanent identity "
        "(billing scope + spend guard key)."
    )
    if not args.yes:
        if not sys.stdin.isatty():
            print(
                "error: re-run with --yes to insert non-interactively",
                file=sys.stderr,
            )
            return 1
        answer = input(f"Insert it into {path}? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("aborted — file unchanged")
            return 1
    path.write_text(new_text)
    print(f"issued identity {minted} — remember to commit")
    return 0


def _insert_metadata_id(text: str, minted: str) -> tuple[str, str] | None:
    """Insert ``id: <minted>`` after the last metadata child, as a pure text
    edit (no safe_dump round-trip — the files' comments must survive).

    Returns ``(new_text, diff_preview)``, or ``None`` when no insertable
    metadata block exists.
    """
    import re

    lines = text.splitlines(keepends=True)
    meta_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(r"^(\s*)metadata:\s*(#.*)?$", line)
        ),
        None,
    )
    if meta_idx is None:
        return None
    meta_indent = len(lines[meta_idx]) - len(lines[meta_idx].lstrip())
    child_prefix = None
    last_child = None
    for j in range(meta_idx + 1, len(lines)):
        stripped = lines[j].strip()
        if not stripped:
            continue  # a blank line may sit inside the block; the next
            # non-blank line decides whether the block continues
        indent = len(lines[j]) - len(lines[j].lstrip())
        if indent <= meta_indent:
            break
        last_child = j
        if child_prefix is None and re.match(r"^(name|workspace):", stripped):
            child_prefix = lines[j][:indent]
    if child_prefix is None or last_child is None:
        return None
    id_line = f"{child_prefix}id: {minted}\n"
    preview = "".join(
        f"    {line.rstrip()}\n" for line in lines[meta_idx : last_child + 1]
    ) + f"+   {child_prefix}id: {minted}"
    new_lines = [*lines[: last_child + 1], id_line, *lines[last_child + 1 :]]
    return "".join(new_lines), preview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openloop")
    sub = parser.add_subparsers(dest="group", required=True)

    agents = sub.add_parser("agents", help="manage agent config-as-code")
    agents_sub = agents.add_subparsers(dest="action", required=True)

    apply = agents_sub.add_parser("apply", help="validate an agent YAML file")
    apply.add_argument("-f", "--file", required=True, help="path to agent YAML")
    apply.set_defaults(func=_cmd_agents_apply)

    # `id` is a group so sibling actions (e.g. `id show`) can land later.
    agents_id = agents_sub.add_parser("id", help="manage durable agent identity")
    agents_id_sub = agents_id.add_subparsers(dest="id_action", required=True)
    issue = agents_id_sub.add_parser(
        "issue", help="stamp a minted durable identity into an agent YAML file"
    )
    issue.add_argument("-f", "--file", required=True, help="path to agent YAML")
    issue.add_argument(
        "-y", "--yes", action="store_true", help="insert without prompting"
    )
    issue.set_defaults(func=_cmd_agents_id_issue)

    slack = sub.add_parser("slack", help="run the Slack surface")
    slack_sub = slack.add_subparsers(dest="action", required=True)
    socket = slack_sub.add_parser(
        "socket", help="run Slack over Socket Mode (no public URL)"
    )
    socket.set_defaults(func=_cmd_slack_socket)

    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_slack_socket(args: argparse.Namespace) -> int:
    from openloop.surfaces.slack_socket import main as run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
