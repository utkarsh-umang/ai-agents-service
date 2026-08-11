"""Object-key scheme for short-video artefacts.

Pure string builders, deliberately free of any SDK import: the key layout is a
contract shared between whoever produces the clips and whoever serves them, and
both sides should be able to agree on it without depending on boto3.

Two rules encoded here:

* **Deterministic.** Keys derive only from ``(folder_id, job_id, index)``, so a
  Celery retry overwrites its own object. The prototype used
  ``{uuid4}_{filename}``, which left an orphan behind on every retry with no way
  to tell which copy the job actually referenced. This matches the idempotency
  the thumbnail tool already relies on.
* **Split by lifetime.** Clips persist; the downloaded source is transient and
  lives under its own prefix so a bucket lifecycle rule can expire it (~7 days)
  without touching the clips.

The hierarchy mirrors Studio's media keys
(``clients/{client}/batches/{batch}/...``).
"""

from __future__ import annotations

CLIPS_PREFIX = "short-video"
SOURCE_PREFIX = "short-video-inputs"


def short_video_clip_key(folder_id: str, job_id: str, index: int) -> str:
    """Key for one produced clip: ``short-video/{folder}/{job}/clips/{index}.mp4``."""
    return f"{CLIPS_PREFIX}/{folder_id}/{job_id}/clips/{index}.mp4"


def short_video_source_key(job_id: str, suffix: str = ".mp4") -> str:
    """Key for the transient downloaded source, under the expiring prefix."""
    return f"{SOURCE_PREFIX}/{job_id}/source{suffix}"
