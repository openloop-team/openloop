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

    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_slack_socket(args: argparse.Namespace) -> int:
    from openloop.surfaces.slack_socket import main as run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
