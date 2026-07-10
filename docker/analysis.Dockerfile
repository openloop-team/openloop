# Analysis sandbox runtime image (sealed analysis worker — Phase 0).
#
# The sandbox runs `--network none`, so every dependency the model-authored
# code may import must be BAKED IN — there is no runtime `pip install`
# (docs/sealed-analysis-worker.md §4.1, locked decision §7: pandas, numpy,
# matplotlib with the Agg backend only).
#
# PRODUCTION: pin the base by digest, e.g.
#   FROM python:3.12-slim@sha256:<digest>
# and pin the built image's digest in ANALYSIS_WORKER_SANDBOX_IMAGE.
#
# `timeout` (GNU coreutils, present in slim) is the in-container deadline —
# the runner execs it as PID 1 (never pass --init; see Phase 0 lock 2).
FROM python:3.12-slim

RUN pip install --no-cache-dir pandas numpy matplotlib

# Agg = render-to-file, no display server; HOME on the tmpfs so incidental
# cache writes (matplotlib font cache) survive a read-only rootfs.
ENV MPLBACKEND=Agg \
    HOME=/tmp
