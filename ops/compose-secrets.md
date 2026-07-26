# Secrets

Production Compose reads secret source files from
`${OPENLOOP_SECRETS_ROOT:-./secrets}` and mounts them read-only under
`/run/secrets`. The default source tree is gitignored.

```text
secrets/
├── postgres_password
├── openai_api_key
├── anthropic_api_key
├── gemini_api_key
├── groq_api_key
├── openrouter_api_key
├── slack_bot_token
├── slack_signing_secret
├── slack_app_token
├── github_token
├── github_app_private_key
├── claude_code_oauth_token
├── coding_worker_openhands_state_master_key
├── broker_identity_private_key
├── broker_receipt_roots
├── broker_capability_roots
└── broker_runtime_roots
```

`postgres_password`, the external-broker identity and root files, and any
credential for an enabled integration must contain real values. The three
broker root-map files contain JSON objects:

```json
{"key-v1":"BASE64_ENCODED_32_BYTE_VALUE"}
```

The GitHub App private-key file contains the PEM itself.
