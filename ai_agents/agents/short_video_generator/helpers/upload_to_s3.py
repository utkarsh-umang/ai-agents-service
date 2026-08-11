"""Upload a produced clip to S3 under a caller-supplied key."""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Re-exported for convenience; the scheme itself lives in storage_keys so it can
# be imported without pulling in boto3.
from ai_agents.agents.short_video_generator.helpers.storage_keys import (
    short_video_clip_key,
    short_video_source_key,
)

__all__ = ["upload_to_s3", "short_video_clip_key", "short_video_source_key"]

PRESIGN_EXPIRES_SECONDS = 86_400  # 24h


def upload_to_s3(
    file_path: Path | str,
    bucket_name: str,
    object_key: str,
    *,
    presign: bool = True,
    expires_in: int = PRESIGN_EXPIRES_SECONDS,
    content_type: str | None = None,
) -> str:
    """Upload one file and return a presigned GET URL (or the key).

    Args:
        file_path: Local file to upload.
        bucket_name: Target bucket.
        object_key: Full key. Build it with ``short_video_clip_key`` so retries
            are idempotent — this function will not invent one.
        presign: Return a time-limited GET URL. Set False to get the key back
            and let the caller serve it however it likes.
        expires_in: Presigned URL lifetime in seconds.
        content_type: Stored as the object's Content-Type when given.
    """
    source = Path(file_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not bucket_name:
        raise ValueError("bucket_name is required")
    if not object_key:
        raise ValueError("object_key is required")

    s3 = boto3.client("s3")

    extra_args = {"ContentType": content_type} if content_type else None

    try:
        s3.upload_file(
            Filename=str(source),
            Bucket=bucket_name,
            Key=object_key,
            ExtraArgs=extra_args,
        )
    except ClientError as exc:
        raise RuntimeError(f"upload of {object_key} failed: {exc}") from exc

    if not presign:
        return object_key

    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": object_key},
        ExpiresIn=expires_in,
    )
