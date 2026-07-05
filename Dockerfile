FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git + CA certs are needed by the coding worker, which shells out to `git`
# to clone/commit/push over HTTPS. tmux backs the OpenHands terminal tool; without
# it the SDK falls back to a subprocess terminal it warns is "less stable" (a
# plausible hang source for the agent's shell commands).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates tmux \
    && rm -rf /var/lib/apt/lists/*

# Static docker CLI so CODING_WORKER_SANDBOX=docker works from inside this
# container (sibling containers over the mounted /var/run/docker.sock — see
# docker-compose.deploy.yml). CLI only, no daemon; major version pinned so a
# breaking CLI change can't ride in on a rebuild. Without this binary the
# sandbox probe fails at boot and the coding worker is disabled (fail-closed).
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

COPY pyproject.toml README.md ./
COPY src ./src

# Install with the `redis` extra so the documented multi-replica deploy path
# (LOCK_BACKEND=redis) can actually coordinate — without it the runtime silently
# falls back to in-process locks — and the `githubapp` extra so GITHUB_APP_*
# auth (short-lived installation tokens) can sign at boot; without it the
# runtime logs GITHUB APP AUTH DISABLED and degrades to GITHUB_TOKEN. Mount the
# App private key read-only and point GITHUB_APP_PRIVATE_KEY_PATH at it.
RUN pip install --upgrade pip && pip install ".[redis,githubapp,mcp,openhands]"

COPY agents ./agents

EXPOSE 8000

CMD ["uvicorn", "openloop.app:app", "--host", "0.0.0.0", "--port", "8000"]
