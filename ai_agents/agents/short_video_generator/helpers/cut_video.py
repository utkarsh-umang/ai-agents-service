"""Cut a source video into clips using ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ai_agents.agents.short_video_generator.helpers.ffmpeg import resolve_ffmpeg


def create_clips(
    video_path: Path | str,
    analysis: Any,
    out_dir: Path | str,
    *,
    ffmpeg_bin: Path | str | None = None,
) -> list[dict]:
    """Cut one file per clip in ``analysis`` into ``out_dir``.

    Args:
        video_path: The downloaded source.
        analysis: A ``VideoAnalysis`` (anything with ``.clips``).
        out_dir: Directory to write clips into — pass a per-job workspace from
            ``helpers.workspace.job_workspace``. Created if absent. Existing
            files are left alone; this function never deletes anything.
        ffmpeg_bin: Override the ffmpeg binary. Defaults to ``resolve_ffmpeg()``.

    Returns:
        One dict per clip, carrying the LLM's reasoning through alongside the
        file. ``index`` is 0-based and stable, so callers can build a
        deterministic, retry-safe storage key from it.

    Note:
        ``-c copy`` is a stream copy: near-instant and lossless, but cuts snap
        to the nearest keyframe, so a clip can start a second or two off its
        requested timestamp. Re-encoding would be frame-accurate at real CPU
        cost. Stream copy is the deliberate v1 choice.
    """
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"source video not found: {source}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    binary = str(ffmpeg_bin) if ffmpeg_bin else str(resolve_ffmpeg())
    suffix = source.suffix or ".mp4"

    results: list[dict] = []

    for index, clip in enumerate(analysis.clips):
        output_file = out / f"clip_{index}{suffix}"

        command = [
            binary,
            "-y",
            "-ss", clip.start_timestamp,
            "-to", clip.end_timestamp,
            "-i", str(source),
            "-c", "copy",
            str(output_file),
        ]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            # ffmpeg puts the actual reason on stderr; swallowing it (the
            # prototype sent both streams to DEVNULL) makes failures unreadable.
            tail = (completed.stderr or "").strip().splitlines()[-5:]
            raise RuntimeError(
                f"ffmpeg failed on clip {index} "
                f"({clip.start_timestamp}-{clip.end_timestamp}): "
                + " | ".join(tail)
            )

        results.append(
            {
                "index": index,
                "title": clip.title,
                "topic": clip.topic,
                "why_it_works": clip.why_it_works,
                "start_timestamp": clip.start_timestamp,
                "end_timestamp": clip.end_timestamp,
                "duration_seconds": clip.duration_seconds,
                "video_path": str(output_file),
            }
        )

    return results
