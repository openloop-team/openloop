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

    # Mirrors app.build_analysis_input_store's backend dispatch, without
    # importing openloop.app (whose import boots the whole application).
    # Only the durable store makes sense here: staging into a process-local
    # in-memory store would be invisible to the runtime that later
    # materializes the inputs.
    settings = get_settings()
    if settings.memory_backend != "postgres":
        print(
            "error: cross-process staging needs the durable input store. Set "
            "MEMORY_BACKEND=postgres and DATABASE_URL so the CLI and the "
            "runtime share it.",
            file=sys.stderr,
        )
        return 1
    from openloop.analysis.postgres import PostgresInputStore

    store = PostgresInputStore(settings.database_url)
    try:
        asyncio.run(_stage_manifest(store, manifest))
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


async def _stage_manifest(store, manifest) -> None:
    """Stage through a possibly-durable store, owning its setup/teardown."""
    setup = getattr(store, "setup", None)
    if setup is not None:
        await setup()
    try:
        await store.stage(manifest)
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            await close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openloop")
    sub = parser.add_subparsers(dest="group", required=True)

    agents = sub.add_parser("agents", help="manage agent config-as-code")
    agents_sub = agents.add_subparsers(dest="action", required=True)

    apply = agents_sub.add_parser("apply", help="validate an agent YAML file")
    apply.add_argument("-f", "--file", required=True, help="path to agent YAML")
    apply.set_defaults(func=_cmd_agents_apply)

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
