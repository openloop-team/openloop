# Contributing to OpenLoop

Thanks for your interest in contributing! OpenLoop is **early-stage**, so
shaping the foundations is the most valuable work right now. This guide explains
how to get involved.

> ⚠️ The codebase, APIs, and config formats are still landing and will change.
> Expect churn, and open an issue before starting anything non-trivial.

## Ways to contribute

- **Discuss design** — open an issue to propose features or weigh in on
  direction. Early architectural input is especially welcome.
- **Report bugs** — include reproduction steps, expected vs. actual behavior,
  and your environment.
- **Improve docs** — clarifications, examples, and fixes to the README or this
  guide are great first contributions.
- **Write code** — pick up an open issue or propose one.

### Areas that need help

- Agent runtime / async task pipeline
- Model adapters (LiteLLM-compatible providers)
- MCP tool gateway and native connectors (GitHub, Slack)
- Slack surface (mentions, threads, approvals)
- Memory layer (Postgres + pgvector)
- Token/cost tracking
- Documentation

## Before you start

For anything beyond a small fix, **open an issue first** so we can align on
direction before you invest time. This avoids duplicated work and PRs that head
the wrong way while the architecture is still forming.

## Development setup

> 🧪 **Preliminary.** A full setup will be documented here as the runtime lands.

```bash
git clone https://github.com/p1c2u/openloop.git
cd openloop
cp .env.example .env
# Edit tracked non-secret settings under configs/prd/, configure the
# openloop-deploy, openloop-runtime, and openloop-broker Doppler projects, then:
mise run secrets-invoke -- mise run compose-up-d
```

For the external OpenHands broker, set an absolute `OPENLOOP_BROKER_ROOT` in
`.env`, then use `mise run compose-build` and the command above. The networkless
HAProxy permission adapter—not the runtime or broker—mounts the raw Docker
socket. The broker must retain
`OPENLOOP_DATA_GID` to connect to the adapter's `root:OPENLOOP_DATA_GID` mode
`0660` UDS through its read-only volume mount. Do not add `DOCKER_GID` or a TCP
Docker endpoint; the adapter does not filter the root-equivalent Docker API.

### PostgreSQL access

Per [ADR 0007](docs/adr/0007-sqlalchemy-postgresql-interface.md), SQLAlchemy is
the only interface to PostgreSQL. Statements are written in the expression
language. A statement kept as SQL text — `text()` or `exec_driver_sql()` — needs
a comment on the line above it beginning `# sql-text:` and saying why the
expression language would obscure what the statement does. Reviewers reject text
without one; nothing enforces it automatically.

Two dialect behaviours to know before writing any statement text. A cast is
written `CAST(:name AS type)`, never `:name::type` — a bind name followed by a
colon is not a bind parameter, so the postfix form compiles to a statement with
no parameters and fails only when it runs. And every statement goes over as a
prepared statement, which carries exactly one command, so a multi-statement
script is split on statement boundaries and applied one execute at a time.

## Pull requests

- **Keep PRs small and focused.** One logical change per PR is easiest to
  review.
- **Describe the motivation.** Explain *why*, not just *what*, and note any
  tradeoffs or follow-ups.
- **Link the issue** the PR addresses (e.g. `Closes #123`).
- **Update docs** alongside behavior changes.
- **Don't commit secrets.** Keep credentials in the scoped Doppler projects,
  never in `.env`, `configs/prd/*.env`, code, or fixtures.

### Commit messages

- Use clear, descriptive messages written in the imperative mood
  (e.g. "Add Slack approval gate", not "added stuff").
- Reference issues where relevant.

## Code of conduct

Be respectful and constructive. We want a welcoming community for contributors
of all backgrounds and experience levels.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE), the same license as the project.
