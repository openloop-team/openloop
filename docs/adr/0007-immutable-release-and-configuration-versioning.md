# 0007 — Address a deployment as an immutable release tuple

- **Status:** Accepted
- **Date:** 2026-08-03

## Context

A deployment currently resolves to whatever the host last produced. Compose
defaults the shared image to the mutable tag `openloop:local`, the deploy files
carry a `build:` stanza, and the documented deploy step builds that tag on the
server, so two hosts can run different code under one name and no host can say
which code it runs. The image also carries agent definitions, which makes an
agent edit either a rebuild or a divergence between the mounted files and the
baked ones.

Configuration is separated from credentials but not versioned. Compose
interpolation, the systemd units, the tracked runtime and broker settings, and
the agent definitions all move together as one checkout, so changing a model
setting and changing the container composition are the same event. Secrets come
from three Doppler projects whose selected configs are chosen at invocation time
and recorded nowhere.

Nothing therefore identifies a running deployment, and rollback means finding a
commit and rebuilding — which reproduces the code but not necessarily the
artifact, and never the configuration that ran beside it.

## Decision

A deployment is an immutable release tuple: one application image, one deploy
bundle revision, one runtime-configuration bundle revision, and the
secret-manager environments that inject its credentials. Each element is chosen
independently and changing one does not require changing another.

A production release selects the image by registry digest. A tag never
identifies a release, and the deploy bundle cannot build an image: image
production is a separate action whose output is a digest.

Deploy configuration and runtime configuration are distinct bundles, each
addressed by the content of its own files. Agent definitions and runtime
settings belong to the runtime-configuration bundle and are absent from the
image, so the same image digest serves every configuration revision.

A release record names the selected revisions, the configuration environment,
and the secret-manager project and config names. It never contains secret
values, and it is host state stored outside the bundles it describes. A record
is immutable: its identity is derived from the tuple, so an altered record is
refused rather than selected.

Rollback selects a previously recorded tuple. Selection fails closed unless the
checkout holds the recorded bundle content and the selected image is
digest-pinned, and it never rebuilds.

A deployed process reports the release that selected it.

## Consequences

- Releasing requires a registry: an image must be pushed and its digest
  resolved before a deployment can select it. Building on the deployment host
  no longer produces something a release can name.
- An agent or settings change ships as a new runtime-configuration revision
  against the unchanged image digest, and rolls back the same way.
- Rollback requires the recorded bundle content to be present; a record whose
  files were never committed cannot be re-selected. Records carry the source
  commit as a hint so an operator knows what to check out.
- Two hosts that select the same release run the same image and the same
  configuration, and a live process can be matched to its record.
- Every deployment surface gains a precondition: a host without a selected
  release does not start, which is a louder failure than starting on whatever
  image happened to be local.
- Local development is unaffected. Mutable tags remain the point there, and the
  build path stays available outside the deploy bundle.
- The record is an audit artifact, not an authorization one: it proves which
  configuration was selected, never that the values behind it were correct.
