"""Download a source video into a caller-supplied directory."""

from __future__ import annotations

from pathlib import Path

import yt_dlp

from ai_agents.agents.short_video_generator.helpers.ffmpeg import ffmpeg_dir

# Default quality ceiling. Vertical short-form never needs more than 720p, and
# capping here cuts roughly 2.5x off the bytes we download, store and re-upload
# (~120 MB for a 20-minute source instead of ~1 GB at 1080p60).
DEFAULT_MAX_HEIGHT = 720


def download_youtube_video(
    url: str,
    dest_dir: Path | str,
    *,
    max_height: int = DEFAULT_MAX_HEIGHT,
    stem: str = "source",
) -> Path:
    """Download ``url`` into ``dest_dir`` and return the resulting file.

    Args:
        url: A YouTube watch URL. Validate it first with
            ``helpers.validate_links.is_valid_youtube_url``.
        dest_dir: Directory to download into — pass a per-job workspace from
            ``helpers.workspace.job_workspace``. Created if absent. Existing
            files are left alone; this function never deletes anything.
        max_height: Quality ceiling in pixels.
        stem: Base filename; the container extension is chosen by yt-dlp.

    Raises:
        FileNotFoundError: if yt-dlp reported success but produced no file.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        # Best video at or below the ceiling, plus best audio; fall back to a
        # single pre-muxed stream when no such pair exists.
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest / f"{stem}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    # Only set ffmpeg_location when we actually found ffmpeg; passing None lets
    # yt-dlp fall back to PATH rather than erroring on an empty override.
    location = ffmpeg_dir()
    if location:
        ydl_opts["ffmpeg_location"] = location

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Match on the stem we asked for rather than "whatever is in the directory",
    # so a workspace holding other artefacts can't confuse the result.
    produced = sorted(p for p in dest.glob(f"{stem}.*") if p.is_file())
    if not produced:
        raise FileNotFoundError(
            f"yt-dlp reported success but produced no {stem}.* file in {dest}"
        )

    # With a merge, yt-dlp leaves the muxed .mp4 alongside nothing else; without
    # one it may leave separate streams. Prefer the merged container.
    for candidate in produced:
        if candidate.suffix == ".mp4":
            return candidate
    return produced[0]
