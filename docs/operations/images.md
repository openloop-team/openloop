# Runtime images

CI publishes `ghcr.io/openloop-team/openloop:main-<sha>` for every commit on
`main`, and a `v*` Git tag promotes that commit's already-published digest to
`:<v*>` without rebuilding.

Deploy by digest, never by tag:

    OPENLOOP_IMAGE=ghcr.io/openloop-team/openloop@sha256:<digest>

Read the digest a tag currently resolves to:

    docker buildx imagetools inspect ghcr.io/openloop-team/openloop:<v*> \
      --format '{{println .Manifest.Digest}}'

Verify where an image came from before deploying it:

    gh attestation verify oci://ghcr.io/openloop-team/openloop@sha256:<digest> \
      --repo openloop-team/openloop \
      --signer-workflow openloop-team/openloop/.github/workflows/ci.yml

Tags are convenience aliases. A registry tag can be repointed by anyone holding
package write access, so what a deployment records and pulls is the digest.
Reproducibility covers the base images, the uv.lock dependency graph, and the
Claude CLI version; OS packages float within the base image's Debian suite, so
rebuilds are not bit-identical.
