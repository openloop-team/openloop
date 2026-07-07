FROM python:3.12-slim

# Run the whole container as a non-root user. The EXPERIMENTAL claude worker
# backend shells out to `claude -p --dangerously-skip-permissions`, and Claude
# Code deliberately REFUSES that flag when it detects it's running as root (a
# guardrail against unsandboxed auto-approval). The container itself is the
# sandbox here, so we drop to an unprivileged user to satisfy the check. uid/gid
# 1000 is a conventional first non-system id; pin it so bind-mount ownership is
# predictable across hosts.
RUN groupadd --gid 1000 openloop \
    && useradd --uid 1000 --gid openloop --create-home --shell /bin/bash openloop

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git + CA certs are needed by the coding worker, which shells out to `git`
# to clone/commit/push over HTTPS. tmux backs the OpenHands terminal tool; without
# it the SDK falls back to a subprocess terminal it warns is "less stable" (a
# plausible hang source for the agent's shell commands). curl stays in the image
# because the claude CLI install below (run later as the unprivileged user, which
# can't apt-get) needs it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates tmux curl \
    && rm -rf /var/lib/apt/lists/*

# Static docker CLI so CODING_WORKER_SANDBOX=docker works from inside this
# container (sibling containers over the mounted /var/run/docker.sock — see
# docker-compose.deploy.yml). CLI only, no daemon; major version pinned so a
# breaking CLI change can't ride in on a rebuild. Without this binary the
# sandbox probe fails at boot and the coding worker is disabled (fail-closed).
# NOTE: at runtime the `openloop` user needs read/write on the mounted docker socket;
# grant it by matching the host socket's group (e.g. compose `group_add`) since
# that gid is host-specific and can't be baked in here.
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

# Everything under /app was copied as root; hand it to the runtime user so the
# coding worker can write scratch state next to the app if it needs to.
RUN chown -R openloop:openloop /app

# Drop privileges for the CLI install and the running process. Claude Code's
# root check keys off the effective uid at runtime, so this USER line is what
# actually makes --dangerously-skip-permissions work.
USER openloop

# The Claude Code CLI for the EXPERIMENTAL claude worker backend
# (CODING_WORKER_BACKEND=claude), which shells out to `claude -p`. Installed via
# the native installer (`npm install -g` is deprecated); it drops a
# self-contained binary into ~/.local/bin — no Node runtime needed. `stable` is
# the stability channel; replace it with an explicit X.Y.Z to freeze the CLI
# across rebuilds (the installer also accepts `latest`). Auth is separate (a
# subscription token in CLAUDE_CODE_OAUTH_TOKEN, or a mounted ~/.claude — see
# the deploy compose). Without this binary the claude backend's probe fails at
# boot and the coding worker is disabled (fail-closed) — it never runs
# half-configured. Unused by the default builtin/openhands backends.
RUN curl -fsSL https://claude.ai/install.sh | bash -s -- stable

# The native installer puts `claude` in ~/.local/bin (/home/openloop now that we
# run as the `openloop` user), which is not on the default PATH.
ENV PATH="/home/openloop/.local/bin:${PATH}"

EXPOSE 8000

CMD ["uvicorn", "openloop.app:app", "--host", "0.0.0.0", "--port", "8000"]
