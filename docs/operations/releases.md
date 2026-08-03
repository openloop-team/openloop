# Releases and configuration versioning

A deployment resolves to four independently chosen versions:

| Element | What it pins | Changed by |
| --- | --- | --- |
| Image | one `<repository>@sha256:<digest>` artifact | building and pushing |
| Deploy bundle | Compose files, systemd units, adapter config | editing those files |
| Runtime-config bundle | `agents/*.yaml`, `configs/<env>/*.env` | editing those files |
| Doppler environments | which project/config injects each secret | choosing another config |

A **release record** is the tuple written down. It names revisions, never
values: bundles appear as file digests and Doppler environments as
project/config names, so a record is safe to keep, copy, and diff.

Records live on the host under `/var/lib/openloop/releases/`, deliberately
outside the bundles — a record kept inside the deploy bundle would change the
revision it was recording. The selected release lives at
`/etc/openloop/release.env`, which every systemd unit loads.

Rolling back one element does not touch the others: the same image digest can
be re-recorded against a newer runtime-config revision, and vice versa.

## Cut a release

Build and push from a workstation or CI — not from the deployment host, whose
job is to select an already-published artifact:

```bash
docker compose -f docker-compose.build.yml build runtime
docker tag openloop:local ghcr.io/openloop-team/openloop:2026-08-03
docker push ghcr.io/openloop-team/openloop:2026-08-03
docker buildx imagetools inspect ghcr.io/openloop-team/openloop:2026-08-03 \
  --format '{{println .Manifest.Digest}}'
```

Record the tuple from the checkout the deployment will run. `--source-commit`
is a hint stored for later rollback; the bundle revisions are the identity:

```bash
openloop release record \
  --image ghcr.io/openloop-team/openloop@sha256:<digest> \
  --doppler openloop-deploy=prd \
  --doppler openloop-runtime=prd \
  --doppler openloop-broker=prd \
  --config-env prd \
  --source-commit "$(git rev-parse HEAD)" \
  --output /var/lib/openloop/releases/"$(date -u +%Y%m%dT%H%M%SZ)".json
```

The command refuses a tag, a mutable reference such as `openloop:local`, a
missing Doppler environment, and anything that looks like a token rather than a
config name. Recording the same tuple twice keeps the first record.

On the deployment host the CLI ships inside the image, so no host Python is
needed:

```bash
sudo docker run --rm \
  --volume /opt/openloop:/opt/openloop:ro \
  --volume /var/lib/openloop/releases:/var/lib/openloop/releases \
  --workdir /opt/openloop \
  ghcr.io/openloop-team/openloop@sha256:<digest> \
  openloop release record --image ghcr.io/openloop-team/openloop@sha256:<digest> ...
```

## Select a release

Selection verifies before it writes: the checkout must hold the recorded bundle
content, or the command refuses and names each file that differs.

```bash
sudo install -d -o root -g root -m 0755 /etc/openloop
openloop release select /var/lib/openloop/releases/<record>.json \
  --root /opt/openloop \
  --output /etc/openloop/release.env
sudo systemctl restart openloop.target
```

`/etc/openloop/release.env` carries the two values Compose interpolates —
`OPENLOOP_IMAGE` and `OPENLOOP_RELEASE_ID` — plus the rest of the tuple as
comments. Each unit loads it through `EnvironmentFile=` and refuses to start
unless `OPENLOOP_IMAGE` is digest-pinned, so a host without a selected release
does not come up on whatever image happens to be local.

Pull the selected image before restarting; the units never pull at boot:

```bash
sudo docker pull ghcr.io/openloop-team/openloop@sha256:<digest>
```

## Verify what is running

```bash
openloop release verify /var/lib/openloop/releases/<record>.json \
  --root /opt/openloop \
  --image "$(docker inspect --format '{{.Config.Image}}' openloop-runtime-1)"
curl -s localhost:8000/healthz | jq .release   # HTTP surfaces only
openloop release show /var/lib/openloop/releases/<record>.json
```

`verify` exits nonzero and prints one `drift:` line per file that no longer
matches. Socket Mode deployments have no HTTP surface; there the release id in
`/etc/openloop/release.env` and the container's image digest are the check.

## Roll back

Rollback is selection, not rebuilding.

1. Pick the record to return to (`openloop release show` prints its tuple).
2. Restore the bundle content it names — `git checkout <source_commit>` in
   `/opt/openloop` when the whole tuple is going back, or restore only the
   files of the bundle being reverted.
3. `openloop release select <record> --root /opt/openloop --output
   /etc/openloop/release.env`.
4. `sudo docker pull <recorded image>` if it is no longer on the host.
5. `sudo systemctl restart openloop.target`.

Reverting one element only — an image regression, say — is a new release
recorded from the current checkout with the older digest:

```bash
openloop release record --image <previous digest> --doppler ... \
  --output /var/lib/openloop/releases/<new>.json
```

That keeps the record honest: it describes what actually ran, rather than
implying that the configuration went back too.

## What is not in the record

- Secret values. Doppler holds them; the record names the config that supplied
  them, so rotating a secret does not change any revision.
- `/opt/openloop/.env`. Compose interpolation is host-local, and the release
  overrides the parts of it that identify the deployment.
- Anything untracked. `openloop release record` refuses to digest local
  credential files if a bundle pattern ever reaches one.
