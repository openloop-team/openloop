# Compose configuration and secrets

OpenLoop keeps ordinary settings separate from credentials:

- `.env` contains only values Docker Compose needs while resolving the model.
- `configs/<env>/runtime.env` contains tracked runtime settings.
- `configs/<env>/broker.env` contains tracked broker settings and public
  verification keys.
- Secrets are injected into the Compose process.

The service-level `env_file` entries parse the tracked config files and add
their values to the corresponding container environment. They do not mount
those files into a container.

Each top-level Compose secret uses an injected environment value as its source.
Compose creates a read-only `/run/secrets/<name>` file only in services granted
that secret. OpenLoop's `Settings` reads `/run/secrets` directly, with mounted
values taking precedence over process environment and `.runtime.env`.

## Service grants

- Postgres receives only `postgres_password`.
- Runtime receives `postgres_password`, its integration credentials, and the
  runtime-owned broker identity and receipt roots.
- Broker receives `postgres_password` and its capability/runtime roots.
- The Docker socket adapter and broker initializer receive no secrets.

The broker root values are JSON objects, for example:

```json
{"key-v1":"BASE64_ENCODED_32_BYTE_VALUE"}
```

`GITHUB_APP_PRIVATE_KEY` contains the PEM text itself. Compose exposes it as
`/run/secrets/github_app_private_key`, while the runtime sets
`GITHUB_APP_PRIVATE_KEY_PATH` to that path.

The broker public verification maps are not secrets. Generate them , then store the output in
`configs/<env>/broker.env`:

```bash
uv run openloop broker keys
```
