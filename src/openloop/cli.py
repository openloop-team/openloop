"""Minimal `openloop` CLI.

First slice supports validating/inspecting agent config-as-code. `apply` simply
validates for now — registering against a live control plane lands with the
durable runtime (see roadmap).

`analysis stage` is the sealed-analysis worker's operator staging path: input
bytes are staged out-of-band by a trusted operator/harness under a freshly
generated high-entropy ``input_ref`` — a capability token whose possession is
the authorization (job-agnostic lookup; job_id is purely run identity and is
never carried in args). The model references it as a ``staged`` entry in the
``inputs`` list (docs/sealed-analysis-worker.md §7; the full rehearsal
walkthrough lives in docs/analysis-rehearsal.md).
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


def _cmd_analysis_stage(args: argparse.Namespace) -> int:
    """Stage operator-provided input bytes under a fresh capability ref.

    Filenames are restricted to one path component (the flat input contract),
    so a directory tree rides as a single ``git archive`` tarball the
    generated program extracts inside the sandbox. The ``input_ref`` is
    generated here with high entropy — possession of the printed ref is the
    authorization, so operator-chosen (guessable) refs are deliberately not
    accepted.
    """
    import asyncio
    import secrets
    import subprocess
    from pathlib import Path

    from openloop.analysis import InputFile, InputManifest
    from openloop.config import get_settings

    input_ref = f"staged:{secrets.token_urlsafe(24)}"
    staged: list[InputFile] = []
    try:
        if args.archive:
            directory = Path(args.archive).resolve()
            archive = subprocess.run(
                ["git", "archive", "--format=tar", "HEAD"],
                cwd=directory,
                check=True,
                capture_output=True,
            )
            staged.append(InputFile(f"{directory.name}.tar", archive.stdout))
        for name in args.files:
            path = Path(name)
            staged.append(InputFile(path.name, path.read_bytes()))
        if not staged:
            print(
                "error: nothing to stage — pass files and/or --archive DIR",
                file=sys.stderr,
            )
            return 1
        manifest = InputManifest(input_ref=input_ref, files=tuple(staged))
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="replace").strip()
        print(f"error: git archive failed: {detail}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Only the durable store makes sense here: staging into a process-local
    # in-memory store would be invisible to the runtime that later materializes
    # the inputs.
    settings = get_settings()
    if settings.effective_storage_mode == "memory":
        print(
            "error: cross-process staging needs the durable input store. Set "
            "STORAGE_MODE=postgres and DATABASE_URL so the CLI and the "
            "runtime share it.",
            file=sys.stderr,
        )
        return 1
    from openloop.analysis.postgres import PostgresInputStore

    store = PostgresInputStore()
    try:
        asyncio.run(
            _stage_manifest(
                store,
                manifest,
                dsn=settings.database_url,
                min_size=settings.postgres_pool_min_size,
                max_size=settings.postgres_pool_max_size,
            )
        )
    except Exception as exc:  # noqa: BLE001 — operator-facing tool, no traceback spam
        print(f"error: staging failed: {exc}", file=sys.stderr)
        return 1

    total = sum(len(file.content) for file in manifest.files)
    print(
        f"staged {len(manifest.files)} file(s), {total} bytes, "
        f"as input_ref {input_ref!r}"
    )
    for file in manifest.files:
        print(f"  - {file.name} ({len(file.content)} bytes)")
    print(
        "\ninvoke it through the tools API (human approval still applies; the "
        "ref is a capability token — anyone holding it can request an "
        "analysis over the staged bytes):\n"
        "\n"
        "  curl -sX POST http://localhost:8000/tools/invoke \\\n"
        "    -H 'content-type: application/json' \\\n"
        '    -d \'{"action": "analysis.report:write", "requested_by": "cli",\n'
        '         "args": {"instruction": "<the analysis question>",\n'
        '                  "inputs": [{"source": "staged", '
        f'"input_ref": "{input_ref}"}}]}}}}\'\n'
        "\n"
        "then approve the returned approval_id:\n"
        "\n"
        "  curl -sX POST http://localhost:8000/approvals/<approval_id>/resolve \\\n"
        "    -H 'content-type: application/json' \\\n"
        '    -d \'{"approver": "<an approver from the agent yaml>", "approve": true}\''
    )
    return 0


async def _stage_manifest(
    store, manifest, *, dsn: str, min_size: int, max_size: int
) -> None:
    """Stage through a durable store, owning one process-scoped pool."""
    from openloop.postgres import create_pool

    pool = await create_pool(dsn, min_size=min_size, max_size=max_size)
    try:
        await store.setup(pool)
        await store.stage(manifest)
    finally:
        await store.close()
        await pool.close()


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

    analysis = sub.add_parser("analysis", help="sealed analysis worker utilities")
    analysis_sub = analysis.add_subparsers(dest="action", required=True)
    stage = analysis_sub.add_parser(
        "stage",
        help=(
            "stage input bytes under a generated capability ref "
            "(trusted operator path)"
        ),
    )
    stage.add_argument(
        "--archive",
        metavar="DIR",
        help=(
            "stage `git archive HEAD` of DIR as one <dirname>.tar "
            "(committed content only)"
        ),
    )
    stage.add_argument(
        "files", nargs="*", help="individual files, staged under their bare names"
    )
    stage.set_defaults(func=_cmd_analysis_stage)

    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_slack_socket(args: argparse.Namespace) -> int:
    from openloop.surfaces.slack_socket import main as run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
