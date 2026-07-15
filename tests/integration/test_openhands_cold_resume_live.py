"""Live proof of the pinned OpenHands cold-resume mechanism.

Opt in with ``OPENLOOP_RUN_OPENHANDS_LIVE=1``. The test uses the real
digest-pinned agent-server twice per decision, but a deterministic local
OpenAI-compatible endpoint, so it spends no provider tokens.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

if importlib.util.find_spec("openhands.workspace") is None:
    pytest.skip("OpenHands optional dependency is not installed", allow_module_level=True)

from openhands.sdk import Agent, Conversation, LLM, Tool
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent, UserRejectObservation
from openhands.sdk.security.confirmation_policy import AlwaysConfirm
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

from openloop.tools.openhands_artifacts import (
    WorkspaceArtifact,
    WorkspaceArtifactIdentity,
    WorkspaceArtifactManifest,
    WorkspaceArtifactStore,
)
from openloop.tools.openhands_docker import (
    CONVERSATION_LEASE_TTL_SECONDS,
    HardenedDockerWorkspace,
    HardenedDockerWorkspaceError,
)
from openloop.tools.openhands_state import OpenHandsKeyDeriver, OpenHandsStateLayout


pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("OPENLOOP_RUN_OPENHANDS_LIVE") != "1",
        reason="set OPENLOOP_RUN_OPENHANDS_LIVE=1 for the real OpenHands proof",
    ),
]

_REJECTION = "User rejected the pending action in Slack"


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    server: "_FakeOpenAIServer"

    def log_message(self, *_args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self._json({"object": "list", "data": []})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size) or b"{}")
        with self.server.calls_lock:
            self.server.calls.append({"path": self.path, "body": request})
            is_completion = self.path.rstrip("/").endswith(
                ("/chat/completions", "/responses")
            )
            if is_completion:
                self.server.completion_calls += 1
                if request.get("tools"):
                    self.server.agent_calls += 1
            number = self.server.agent_calls

        if not is_completion:
            self._json({"input_tokens": 1, "total_tokens": 1})
            return

        if not request.get("tools"):
            message = {"role": "assistant", "content": "🔧 Cold resume proof"}
            finish_reason = "stop"
            response_id = f"chatcmpl-title-{self.server.completion_calls}"
        elif number == 1:
            command = (
                "printf 'accepted-once\\n' >> /workspace/resumed.txt && "
                "printf 'Cold resume proof\\nRecovered after container removal.\\n' "
                "> /workspace/OPENLOOP_PR.md"
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_cold_resume_decision",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps(
                                {"command": command, "timeout": 10}
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
            response_id = f"chatcmpl-proof-{number}"
        else:
            message = {
                "role": "assistant",
                "content": "Cold resume proof complete.",
            }
            finish_reason = "stop"
            response_id = f"chatcmpl-proof-{number}"

        self._json(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    def _json(self, value: object) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeOpenAIServer(ThreadingHTTPServer):
    calls: list[dict]
    calls_lock: threading.Lock
    completion_calls: int
    agent_calls: int


@contextmanager
def _fake_openai():
    server = _FakeOpenAIServer(("0.0.0.0", 0), _FakeOpenAIHandler)
    server.calls = []
    server.calls_lock = threading.Lock()
    server.completion_calls = 0
    server.agent_calls = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode().strip()


def _repository(root: Path) -> tuple[Path, str]:
    source = root / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.email", "proof@example.invalid")
    _git(source, "config", "user.name", "Cold Resume Proof")
    (source / "tracked.txt").write_text("base\n")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "base")
    base_commit = _git(source, "rev-parse", "HEAD")
    # A deterministic pre-pause workspace change makes the paused delta
    # non-empty without relying on an earlier unconfirmed tool action.
    (source / "prepause.txt").write_text("survives parking\n")
    return source, base_commit


def _agent(fake: _FakeOpenAIServer) -> Agent:
    llm = LLM(
        model="openai/gpt-4o-mini",
        api_key="proof-only",
        base_url=f"http://host.docker.internal:{fake.server_port}/v1",
        num_retries=0,
        timeout=30,
        input_cost_per_token=0,
        output_cost_per_token=0,
    )
    return Agent(
        llm=llm,
        tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
    )


def _restore(
    store: WorkspaceArtifactStore,
    artifact: WorkspaceArtifact,
    identity: WorkspaceArtifactIdentity,
    repo: Path,
) -> WorkspaceArtifactManifest:
    with store.open_verified(artifact, identity) as verified:
        payload = verified.stream.read()
        if payload:
            _git(repo, "apply", "--binary", "-", input_bytes=payload)
        return verified.manifest


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_pause_remove_restore_attach_and_decide(tmp_path, decision):
    source, base_commit = _repository(tmp_path)
    layout = OpenHandsStateLayout(tmp_path / "state")
    keys = OpenHandsKeyDeriver.from_base64(
        base64.b64encode(bytes(range(32))).decode(), master_key_id="proof-key"
    )
    adapter = HardenedDockerWorkspace(layout=layout, keys=keys)
    store = WorkspaceArtifactStore(layout, keys, scratch_root=tmp_path / "scratch")
    conversation_id = uuid.uuid4()
    job_id = f"{decision}-proof"
    paths = layout.for_job(job_id)

    first_workspace = first_conversation = None
    second_workspace = second_conversation = None
    with _fake_openai() as fake:
        agent = _agent(fake)
        try:
            first_workspace = adapter.create(source, job_id)
            first_container = first_workspace._container_id
            first_conversation = Conversation(
                agent=agent,
                workspace=first_workspace,
                conversation_id=conversation_id,
                max_iteration_per_run=10,
                visualizer=None,
                delete_on_close=False,
            )
            first_conversation.send_message("Create the requested proof files.")
            # send_message lazily initializes the remote agent and its state;
            # set the policy afterwards so initialization cannot replace it.
            first_conversation.set_confirmation_policy(AlwaysConfirm())
            info = first_workspace.client.get(
                f"/api/conversations/{conversation_id}"
            )
            info.raise_for_status()
            assert info.json()["confirmation_policy"]["kind"] == "AlwaysConfirm"
            first_conversation.run(timeout=90)
            advertised_tools = [
                tool.get("function", {}).get("name")
                for request in fake.calls
                for tool in request["body"].get("tools", [])
            ]
            event_summary = [
                (type(event).__name__, repr(event)[:500])
                for event in first_conversation.state.events
                if type(event).__name__
                not in {"ConversationStateUpdateEvent", "SystemPromptEvent"}
            ]
            request_summary = [
                {
                    "path": request["path"],
                    "tools": [
                        tool.get("function", {}).get("name")
                        for tool in request["body"].get("tools", [])
                    ],
                    "roles": [
                        message.get("role")
                        for message in request["body"].get("messages", [])
                    ],
                    "last_message": repr(
                        request["body"].get("messages", [{}])[-1]
                    )[:800],
                }
                for request in fake.calls
            ]
            assert first_conversation.state.execution_status == (
                ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
            ), {
                "advertised_tools": advertised_tools,
                "model_calls": fake.agent_calls,
                "request_paths": [request["path"] for request in fake.calls],
                "requests": request_summary,
                "events": event_summary,
            }
            paused_metrics = (
                first_conversation.conversation_stats.get_combined_metrics()
            ).accumulated_token_usage

            paused_bytes = io.BytesIO()
            paused_archive = adapter.stream_git_delta(
                first_workspace, paused_bytes, base_ref=base_commit
            )
            paused_identity = WorkspaceArtifactIdentity(
                job_id, str(conversation_id), "segment-1", "paused"
            )
            paused_artifact = store.put_atomic(
                paused_identity,
                io.BytesIO(paused_bytes.getvalue()),
                WorkspaceArtifactManifest(
                    format="git-delta", base_commit=paused_archive.base_commit
                ),
            )

            first_conversation.close()
            first_conversation = None
            first_workspace.cleanup()
            first_workspace = None

            # Move the mutable base ref after parking. Resume must ignore it.
            (source / "prepause.txt").unlink()
            (source / "tracked.txt").write_text("advanced\n")
            _git(source, "add", "-A")
            _git(source, "commit", "-m", "advance base")

            fresh = tmp_path / "fresh"
            subprocess.run(
                ["git", "clone", str(source), str(fresh)],
                check=True,
                capture_output=True,
            )
            _git(fresh, "checkout", "-B", "resume-job", base_commit)
            paused_manifest = _restore(
                store, paused_artifact, paused_identity, fresh
            )
            assert paused_manifest.base_commit == base_commit
            assert (fresh / "tracked.txt").read_text() == "base\n"
            assert (fresh / "prepause.txt").read_text() == "survives parking\n"

            second_workspace = adapter.create(fresh, job_id)
            second_container = second_workspace._container_id
            assert second_container != first_container
            second_conversation = adapter.attach_conversation(
                second_workspace,
                agent=agent,
                conversation_id=conversation_id,
                max_iterations=10,
            )
            assert second_conversation.state.execution_status == (
                ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
            )
            if decision == "reject":
                second_conversation.reject_pending_actions(_REJECTION)
            second_conversation.run(timeout=90)
            assert second_conversation.state.execution_status == (
                ConversationExecutionStatus.FINISHED
            )

            events = list(second_conversation.state.events)
            prompts = [
                event
                for event in events
                if isinstance(event, MessageEvent) and event.source == "user"
            ]
            assert len(prompts) == 1
            rejections = [
                event for event in events if isinstance(event, UserRejectObservation)
            ]
            if decision == "accept":
                assert (fresh / "resumed.txt").read_text() == "accepted-once\n"
                assert rejections == []
                pr_lines = (fresh / "OPENLOOP_PR.md").read_text().splitlines()
                pr_title, pr_body = pr_lines[0], "\n".join(pr_lines[1:])
                (fresh / "OPENLOOP_PR.md").unlink()
            else:
                assert not (fresh / "resumed.txt").exists()
                assert len(rejections) == 1
                assert rejections[0].rejection_reason == _REJECTION
                pr_title = "Rejected cold resume proof"
                pr_body = "The rejected action was not executed."

            final_metrics = (
                second_conversation.conversation_stats.get_combined_metrics()
            ).accumulated_token_usage
            assert final_metrics.prompt_tokens > paused_metrics.prompt_tokens

            final_bytes = io.BytesIO()
            final_archive = adapter.stream_git_delta(
                second_workspace, final_bytes, base_ref=base_commit
            )
            final_identity = WorkspaceArtifactIdentity(
                job_id, str(conversation_id), "segment-2", "final"
            )
            final_artifact = store.put_atomic(
                final_identity,
                io.BytesIO(final_bytes.getvalue()),
                WorkspaceArtifactManifest(
                    format="git-delta",
                    base_commit=final_archive.base_commit,
                    pr_title=pr_title,
                    pr_body=pr_body,
                ),
            )

            second_conversation.close()
            second_conversation = None
            second_workspace.cleanup()
            second_workspace = None

            reconstructed = tmp_path / "reconstructed"
            subprocess.run(
                ["git", "clone", str(source), str(reconstructed)],
                check=True,
                capture_output=True,
            )
            _git(reconstructed, "checkout", "-B", "recovered-job", base_commit)
            final_manifest = _restore(
                store, final_artifact, final_identity, reconstructed
            )
            assert final_manifest.pr_title == pr_title
            assert final_manifest.pr_body == pr_body
            assert not (reconstructed / "OPENLOOP_PR.md").exists()
            assert (reconstructed / "prepause.txt").read_text() == (
                "survives parking\n"
            )
            assert (reconstructed / "resumed.txt").exists() is (
                decision == "accept"
            )
            assert fake.agent_calls == 2
        finally:
            if first_conversation is not None:
                first_conversation.close()
            if first_workspace is not None:
                first_workspace.cleanup()
            if second_conversation is not None:
                second_conversation.close()
            if second_workspace is not None:
                second_workspace.cleanup()
            shutil.rmtree(paths.root, ignore_errors=True)


def test_unclean_stop_fails_closed_until_conversation_lease_expires(tmp_path):
    source, _ = _repository(tmp_path)
    layout = OpenHandsStateLayout(tmp_path / "state")
    keys = OpenHandsKeyDeriver.from_base64(
        base64.b64encode(bytes(range(32))).decode(), master_key_id="proof-key"
    )
    adapter = HardenedDockerWorkspace(layout=layout, keys=keys)
    conversation_id = uuid.uuid4()
    job_id = "stale-lease-proof"
    paths = layout.for_job(job_id)

    first_workspace = first_conversation = None
    immediate_workspace = recovered_workspace = recovered_conversation = None
    with _fake_openai() as fake:
        agent = _agent(fake)
        try:
            first_workspace = adapter.create(source, job_id)
            first_conversation = Conversation(
                agent=agent,
                workspace=first_workspace,
                conversation_id=conversation_id,
                max_iteration_per_run=10,
                visualizer=None,
                delete_on_close=False,
            )
            first_conversation.send_message("Create the requested proof files.")
            first_conversation.set_confirmation_policy(AlwaysConfirm())
            first_conversation.run(timeout=90)
            assert first_conversation.state.execution_status == (
                ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
            )

            subprocess.run(
                ["docker", "kill", first_workspace._container_id],
                check=True,
                capture_output=True,
            )
            # The --rm container is gone; prevent upstream cleanup from trying
            # to stop its stale ID. Closing only drains the client/WebSocket and
            # cannot release the server-side lease after this simulated crash.
            first_workspace._container_id = None
            first_conversation.close()
            first_conversation = None
            first_workspace = None

            immediate_workspace = adapter.create(source, job_id)
            with pytest.raises(HardenedDockerWorkspaceError, match="lease"):
                adapter.attach_conversation(
                    immediate_workspace,
                    agent=agent,
                    conversation_id=conversation_id,
                    max_iterations=10,
                )
            immediate_workspace.cleanup()
            immediate_workspace = None

            time.sleep(int(CONVERSATION_LEASE_TTL_SECONDS) + 2)
            recovered_workspace = adapter.create(source, job_id)
            recovered_conversation = adapter.attach_conversation(
                recovered_workspace,
                agent=agent,
                conversation_id=conversation_id,
                max_iterations=10,
            )
            assert recovered_conversation.state.execution_status == (
                ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
            )
        finally:
            if first_conversation is not None:
                first_conversation.close()
            if first_workspace is not None:
                first_workspace.cleanup()
            if immediate_workspace is not None:
                immediate_workspace.cleanup()
            if recovered_conversation is not None:
                recovered_conversation.close()
            if recovered_workspace is not None:
                recovered_workspace.cleanup()
            shutil.rmtree(paths.root, ignore_errors=True)
