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
# configure model provider keys and Slack credentials, then:
docker compose up -d
```

## Pull requests

- **Keep PRs small and focused.** One logical change per PR is easiest to
  review.
- **Describe the motivation.** Explain *why*, not just *what*, and note any
  tradeoffs or follow-ups.
- **Link the issue** the PR addresses (e.g. `Closes #123`).
- **Update docs** alongside behavior changes.
- **Don't commit secrets.** Keep credentials in `.env` (gitignored), never in
  code or fixtures.

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
