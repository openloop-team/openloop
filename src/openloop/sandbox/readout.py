"""Host-side read-out containment (sealed analysis worker, Phase 0 lock 4).

After a sealed run, ``outputs/`` is a directory **hostile code controlled**,
and the orchestrator reads it on the host — so reading is the exfiltration
gate, the file-shaped instance of the broker doc's "reads are the exfiltration
path" rule. The classic kill is a symlink planted at ``outputs/report.md``
pointing at an absolute host path: a naive ``open()`` follows it out of the
sandbox boundary and the content lands in a surface post.

Containment is enforced **at open time**, never by a path pre-check — a
realpath check races (the code can swap a checked intermediate directory for a
symlink before the open; ``O_NOFOLLOW`` guards only the final component). The
recipe, per the lock:

- open the root itself as a dirfd (``O_DIRECTORY | O_NOFOLLOW``);
- open the entry *via that dirfd* with ``O_NOFOLLOW | O_NONBLOCK`` —
  ``O_NONBLOCK`` because ``O_NOFOLLOW`` does NOT stop a FIFO open from
  blocking until a writer appears;
- ``fstat`` **the fd**: require a regular file and ``st_nlink == 1`` (a
  hardlink planted in ``outputs/`` can alias ``inputs/`` — within the
  provisioned blast radius, but refused as defense-in-depth);
- bounded read from that same fd. No stat-then-open TOCTOU window anywhere.

Phase 1 reads a single component (``report.md``), so the dirfd form suffices;
nested paths (a later bundle read-out) require
``openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`` or per-component dirfd
traversal — Python's stdlib lacks ``openat2``, which is fine until then.
Filenames are treated as hostile: only bare single components are accepted.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

__all__ = ["ReadOutViolation", "read_contained"]

_CHUNK = 65_536


class ReadOutViolation(RuntimeError):
    """The read-out target failed containment — hostile until proven benign."""


def read_contained(root: Path, name: str, *, max_bytes: int) -> tuple[bytes, bool]:
    """Read ``root/name`` with open-time containment; returns ``(data, truncated)``.

    ``name`` must be a bare single path component. Raises
    :class:`ReadOutViolation` for anything that isn't a plain, single-linked
    regular file directly under ``root``; missing files raise the usual
    :class:`FileNotFoundError` (absence is a job outcome, not an attack).
    """
    if (
        not name
        or name != os.path.basename(name)
        or name in (".", "..")
        or "\x00" in name
    ):
        raise ReadOutViolation(f"refusing read-out name {name!r}: not a bare filename")

    # The root itself must be a real directory, not a symlinked stand-in.
    dir_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=dir_fd,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ReadOutViolation(
                    f"read-out target {name!r} is a symlink — refused"
                ) from exc
            raise
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ReadOutViolation(
                    f"read-out target {name!r} is not a regular file — refused"
                )
            if st.st_nlink > 1:
                raise ReadOutViolation(
                    f"read-out target {name!r} has {st.st_nlink} links — refused"
                )
            data = bytearray()
            # Read max_bytes + 1 so truncation is detected without ever
            # buffering an attacker-sized file.
            while len(data) <= max_bytes:
                chunk = os.read(fd, min(_CHUNK, max_bytes + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            truncated = len(data) > max_bytes
            return bytes(data[:max_bytes]), truncated
        finally:
            os.close(fd)
    finally:
        os.close(dir_fd)
