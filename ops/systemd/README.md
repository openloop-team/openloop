# OpenLoop systemd deployment

These units run the full OpenLoop production Compose deployment on Ubuntu
26.04 LTS from `/opt/openloop`:

```text
docker-compose.deploy.yml + docker-compose.broker.yml
```

The units manage existing images only. They never pull or build during boot.
They also stop containers without deleting containers, networks, or volumes.

## Units

| Unit | Compose service | Starts after |
| --- | --- | --- |
| `openloop-postgres.service` | `postgres` | Docker |
| `openloop-docker-socket-adapter.service` | `docker-socket-adapter` | Postgres |
| `openloop-broker-init.service` | `broker-init` | Docker socket adapter |
| `openloop-broker.service` | `broker` | Broker initializer |
| `openloop-runtime.service` | `runtime` | Broker |
| `openloop.target` | complete deployment | Runtime |

The intentionally serialized order gives first boot and recovery one
deterministic path. `openloop.target` is the unit to enable at boot and the
normal whole-stack control point.

## Prerequisites

Install Docker Engine and the Docker Compose plugin using Docker's
[Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/). Docker
officially supports Ubuntu Resolute 26.04 LTS.

Verify the required commands and daemon:

```bash
test -x /usr/bin/docker
sudo /usr/bin/docker compose version
sudo systemctl is-active docker.service
```

Place the checkout at exactly `/opt/openloop`. The following files must exist
before a unit can start:

```text
/opt/openloop/docker-compose.deploy.yml
/opt/openloop/docker-compose.broker.yml
/opt/openloop/.env
/opt/openloop/configs/prd/runtime.env
/opt/openloop/configs/prd/broker.env
/etc/openloop/openloop-secrets.env
```

Create the non-secret Compose interpolation file and install the systemd secret
environment template:

```bash
cd /opt/openloop
sudo install -o root -g root -m 0644 .env.example .env
sudo install -d -o root -g root -m 0700 /etc/openloop
sudo install -o root -g root -m 0600 \
  ops/systemd/openloop-secrets.env.example \
  /etc/openloop/openloop-secrets.env
sudoedit /etc/openloop/openloop-secrets.env
```

Keep the trust domains separate:

- `.env` contains non-secret Compose interpolation and deployment-wide
  settings.
- `configs/prd/runtime.env` contains tracked non-secret runtime settings.
- `configs/prd/broker.env` contains tracked non-secret broker settings and
  derived app public keys.
- `/etc/openloop/openloop-secrets.env` contains every true secret consumed by
  the Compose process.

Protect the secret environment file from non-root reads:

```bash
sudo chown root:root /etc/openloop/openloop-secrets.env
sudo chmod 0600 /etc/openloop/openloop-secrets.env
```

Set `OPENLOOP_BROKER_ROOT` in `.env` to an absolute host path. If GitHub App
authentication is enabled, put the PEM itself in the multiline
`GITHUB_APP_PRIVATE_KEY` assignment. A single-quoted systemd EnvironmentFile
value may span multiple lines.

## Prepare images

The Compose project name must always be `openloop`. This name determines the
locally built image tags and all project resources used later by systemd.

```bash
cd /opt/openloop
sudo /usr/bin/docker compose \
  --project-name openloop \
  --file docker-compose.deploy.yml \
  --file docker-compose.broker.yml \
  config --quiet
```

Pull the two external images and build the shared OpenLoop image:

```bash
cd /opt/openloop
sudo /usr/bin/docker compose \
  --project-name openloop \
  --file docker-compose.deploy.yml \
  --file docker-compose.broker.yml \
  pull postgres docker-socket-adapter
sudo /usr/bin/docker compose \
  --project-name openloop \
  --file docker-compose.deploy.yml \
  --file docker-compose.broker.yml \
  build runtime
```

Do this before initial activation and after deploying code or Dockerfile
changes. Boot remains independent of registries and image builds.

## Install

Install the units with root ownership:

```bash
sudo install -o root -g root -m 0644 \
  /opt/openloop/ops/systemd/openloop-postgres.service \
  /opt/openloop/ops/systemd/openloop-docker-socket-adapter.service \
  /opt/openloop/ops/systemd/openloop-broker-init.service \
  /opt/openloop/ops/systemd/openloop-broker.service \
  /opt/openloop/ops/systemd/openloop-runtime.service \
  /etc/systemd/system/
sudo install -o root -g root -m 0644 \
  /opt/openloop/ops/systemd/openloop.target \
  /etc/systemd/system/
sudo systemctl daemon-reload
```

Verify the installed unit syntax and dependencies:

```bash
sudo systemd-analyze verify \
  /etc/systemd/system/openloop-postgres.service \
  /etc/systemd/system/openloop-docker-socket-adapter.service \
  /etc/systemd/system/openloop-broker-init.service \
  /etc/systemd/system/openloop-broker.service \
  /etc/systemd/system/openloop-runtime.service \
  /etc/systemd/system/openloop.target
```

Enable the aggregate target at boot and start the stack:

```bash
sudo systemctl enable --now openloop.target
```

Do not enable the five service units individually. The target's dependency
chain activates them in order.

## Verify

Check systemd activation:

```bash
sudo systemctl status \
  openloop-postgres.service \
  openloop-docker-socket-adapter.service \
  openloop-broker-init.service \
  openloop-broker.service \
  openloop-runtime.service \
  openloop.target
```

Check current container state:

```bash
cd /opt/openloop
sudo /usr/bin/docker compose \
  --project-name openloop \
  --file docker-compose.deploy.yml \
  --file docker-compose.broker.yml \
  ps
```

Postgres, the Docker socket adapter, and broker must report healthy. Runtime
must report running. The broker initializer is expected to show an exited
container with exit code zero after a successful activation.

## Routine operation

Operate the complete stack through the target:

```bash
sudo systemctl start openloop.target
sudo systemctl stop openloop.target
sudo systemctl restart openloop.target
```

Stopping the target uses `docker compose stop`. It preserves containers,
networks, the Postgres volume, and broker data.

Leaf units are independently addressable when needed:

```bash
sudo systemctl restart openloop-runtime.service
sudo systemctl stop openloop-runtime.service
sudo systemctl start openloop-runtime.service
```

Restarting a foundation unit can stop units that require it. Starting that
foundation unit again does not automatically start its reverse dependencies.
For Postgres, adapter, initializer, or broker maintenance, restart
`openloop.target` to restore the complete dependency chain.

## Logs

The systemd journal records unit activation, preflight, ordering, and Compose
command failures:

```bash
sudo journalctl -u openloop-runtime.service -u openloop-broker.service
sudo journalctl -u openloop.target --since boot
```

The long-running Compose containers are detached, so application logs remain
in Docker:

```bash
cd /opt/openloop
sudo /usr/bin/docker compose \
  --project-name openloop \
  --file docker-compose.deploy.yml \
  --file docker-compose.broker.yml \
  logs --follow postgres docker-socket-adapter broker runtime
```

## Deploy an update

From the updated `/opt/openloop` checkout:

1. Re-run the secret and quiet Compose configuration validation.
2. Pull `postgres` and `docker-socket-adapter`.
3. Build the shared image through the `runtime` service.
4. Reinstall the unit files and reload systemd.
5. Restart the aggregate target.
6. Re-check systemd and Compose state.

Refresh the installed units before activation:

```bash
sudo install -o root -g root -m 0644 \
  /opt/openloop/ops/systemd/openloop-postgres.service \
  /opt/openloop/ops/systemd/openloop-docker-socket-adapter.service \
  /opt/openloop/ops/systemd/openloop-broker-init.service \
  /opt/openloop/ops/systemd/openloop-broker.service \
  /opt/openloop/ops/systemd/openloop-runtime.service \
  /etc/systemd/system/
sudo install -o root -g root -m 0644 \
  /opt/openloop/ops/systemd/openloop.target \
  /etc/systemd/system/
sudo systemctl daemon-reload
```

Then restart the deployment:

```bash
sudo systemctl restart openloop.target
```

Never add `--build` or a pull policy to a systemd unit. Image acquisition is a
deployment action, not a boot action.

## Recover from a failed activation

Inspect the first failed unit and its container logs:

```bash
sudo systemctl --failed
sudo journalctl -u openloop-postgres.service \
  -u openloop-docker-socket-adapter.service \
  -u openloop-broker-init.service \
  -u openloop-broker.service \
  -u openloop-runtime.service \
  --since boot
```

Correct the configuration, host path, permissions, image, or service health
problem. Then clear the failed state and activate the whole dependency chain:

```bash
sudo systemctl reset-failed \
  openloop-postgres.service \
  openloop-docker-socket-adapter.service \
  openloop-broker-init.service \
  openloop-broker.service \
  openloop-runtime.service
sudo systemctl restart openloop.target
```

Do not use `docker compose down`, `docker compose down --volumes`, or manual
volume removal as routine recovery.

## Supervision boundary

Each long-running unit uses `Type=oneshot` and `RemainAfterExit=yes` because
`docker compose up --wait` switches to detached mode. An active systemd unit
therefore means its activation succeeded; it is not a continuous health
assertion.

Docker applies the Compose `restart: unless-stopped` policies after activation.
Use `docker compose ps` and `docker compose logs` to inspect current container
health and restart behavior.
