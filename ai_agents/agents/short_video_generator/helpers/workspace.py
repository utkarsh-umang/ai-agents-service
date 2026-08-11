"""Per-job scratch space.

Every job gets its own private directory that is removed when the job ends,
success or failure. Nothing here is shared between jobs and nothing is ever
wiped on entry.

The original prototype used three fixed, CWD-relative directories
(``videos/input``, ``videos/output``, ``videos/captions``) and deleted their
contents at the start of every step. That is fine for one person in a notebook
and destroys data the moment two jobs run concurrently: job B's download would
unlink job A's source out from under a running ffmpeg. Passing an explicit
directory into each helper is what makes the pipeline safe to run in a Celery
worker with more than one slot.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: str, *, limit: int = 40) -> str:
    """Make ``value`` safe for use inside a directory name."""
    return _UNSAFE.sub("-", str(value)).strip("-")[:limit] or "job"


@contextmanager
def job_workspace(
    job_id: str,
    *,
    parent: Path | str | None = None,
    keep: bool = False,
) -> Iterator[Path]:
    """Yield a private scratch directory for one job.

    Args:
        job_id: Used only to make the directory identifiable while it exists.
        parent: Where to create it. Defaults to the system temp dir. Point this
            at a large volume if the system temp dir is small — a 20-minute
            720p source is ~120 MB and the clips add to that.
        keep: Leave the directory behind instead of removing it. For debugging
            a failed job; never enable it in production, it leaks disk.

    The directory is removed on exit even if the body raises, so a failed job
    cannot leak a partial download.
    """
    root = Path(
        tempfile.mkdtemp(
            prefix=f"svg-{_slug(job_id)}-",
            dir=str(parent) if parent is not None else None,
        )
    )
    try:
        yield root
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
