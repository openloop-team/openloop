"""Provider-free OpenAI-compatible HTTP fixture for broker live canaries."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    server: "FakeOpenAIServer"

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


class FakeOpenAIServer(ThreadingHTTPServer):
    calls: list[dict]
    calls_lock: threading.Lock
    completion_calls: int
    agent_calls: int


@contextmanager
def fake_openai():
    server = FakeOpenAIServer(("0.0.0.0", 0), _FakeOpenAIHandler)
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
