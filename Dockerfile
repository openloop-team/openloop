FROM python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217

# Run the whole container as a non-root user. The EXPERIMENTAL claude worker
# backend shells out to `claude -p --dangerously-skip-permissions`, and Claude
# Code deliberately REFUSES that flag when it detects it's running as root (a
# guardrail against unsandboxed auto-approval). The container itself is the
# sandbox here, so we drop to an unprivileged user to satisfy the check. uid/gid
# 1000 is a conventional first non-system id; pin it so bind-mount ownership is
# predictable across hosts. The external broker keeps a distinct uid while one
# pinned supplementary gid mediates only the explicitly shared filesystem
# surfaces. Keep uid 1000 stable: existing host bind mounts rely on it.
ARG OPENLOOP_BROKER_UID=10002
ARG OPENLOOP_DATA_GID=10777
RUN groupadd --gid 1000 openloop \
    && groupadd --gid "${OPENLOOP_DATA_GID}" openloop-data \
    && useradd --uid 1000 --gid openloop --create-home --shell /bin/bash openloop \
    && usermod --append --groups openloop-data openloop \
    && useradd --uid "${OPENLOOP_BROKER_UID}" --gid openloop-data \
        --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin \
        openloop-broker

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

# Static Docker CLI for the trusted external broker. In the broker composition,
# DOCKER_HOST points it at the HAProxy permission adapter's private UDS; only
# that adapter mounts the raw host socket. The runtime image carries the same
# binary because both services share this build, but receives no Docker
# endpoint. CLI only, no daemon; major version pinned so a breaking CLI change
# cannot ride in on a rebuild.
COPY --from=docker:27-cli@sha256:851f91d241214e7c6db86513b270d58776379aacc5eb9c4a87e5b47115e3065c /usr/local/bin/docker /usr/local/bin/docker

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install with the `redis` extra so the documented multi-replica deploy path
# (LOCK_BACKEND=redis) can actually coordinate — without it the runtime silently
# falls back to in-process locks — and the `githubapp` extra so GITHUB_APP_*
# auth (short-lived installation tokens) can sign at boot; without it the
# runtime logs GITHUB APP AUTH DISABLED and degrades to GITHUB_TOKEN. Mount the
# App private key read-only and point GITHUB_APP_PRIVATE_KEY_PATH at it.
#
# uv, not pip: only uv's `sync` command reads uv.lock, and mise.toml already made
# that switch for the dev and CI path. UV_PROJECT_ENVIRONMENT points the project
# environment at /usr/local so console scripts and uvicorn land on PATH — the
# default /app/.venv is on no PATH in this image. --inexact leaves packages uv
# does not own (the base image's pip) in place. --no-editable copies the package
# in: a project sync is editable by default, which would resolve imports through
# /app/src, and the chown below makes that path writable by the account the
# process runs as.
COPY --from=ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --locked --inexact --no-editable \
    --extra redis --extra githubapp --extra mcp --extra broker --group openhands

COPY agents ./agents

# Everything under /app was copied as root; hand it to the runtime user so the
# coding worker can write scratch state next to the app if it needs to.
RUN chown -R openloop:openloop /app

# Drop privileges for the CLI install and the running process. Claude Code's
# root check keys off the effective uid at runtime, so this USER line is what
# actually makes --dangerously-skip-permissions work.
USER openloop

ARG CLAUDE_CODE_VERSION=2.1.236
# The Claude Code CLI for the EXPERIMENTAL claude worker backend
# (CODING_WORKER_BACKEND=claude), which shells out to `claude -p`. Installed via
# the native installer (`npm install -g` is deprecated); it drops a
# self-contained binary into ~/.local/bin — no Node runtime needed. Pinned to
# the version in ARG CLAUDE_CODE_VERSION above, not to a moving channel;
# bumping the CLI is a deliberate commit that changes that ARG. Auth is separate (a
# subscription token in CLAUDE_CODE_OAUTH_TOKEN, or a mounted ~/.claude — see
# the deploy compose). Without this binary the claude backend's probe fails at
# boot and the coding worker is disabled (fail-closed) — it never runs
# half-configured. Unused by the default builtin/openhands backends.
RUN curl -fsSL https://claude.ai/install.sh | bash -s -- "${CLAUDE_CODE_VERSION}"

# The native installer puts `claude` in ~/.local/bin (/home/openloop now that we
# run as the `openloop` user), which is not on the default PATH.
ENV PATH="/home/openloop/.local/bin:${PATH}"

EXPOSE 8000

CMD ["uvicorn", "openloop.app:app", "--host", "0.0.0.0", "--port", "8000"]
