"""Locating the ffmpeg binary.

The prototype hardcoded ``os.path.join(FFMPEG_PATH, "ffmpeg.exe")``, which only
works on Windows, and read ``FFMPEG_PATH`` at import time with no None check —
so importing the module crashed outright when the variable was unset. Both are
fixed here: PATH is the normal case, the env var is an optional override, and
the error only fires when ffmpeg is actually needed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class FFmpegNotFound(RuntimeError):
    """Raised when no usable ffmpeg binary can be located."""


def resolve_ffmpeg() -> Path:
    """Return the path to the ffmpeg binary.

    Resolution order:
      1. ``FFMPEG_PATH`` — accepted either as the binary itself or as the
         directory containing it, since the prototype used the directory form.
      2. ``ffmpeg`` on PATH, which is the normal case in a container.

    Raises:
        FFmpegNotFound: with an actionable message rather than a bare
            ``FileNotFoundError`` from deep inside subprocess.
    """
    override = os.getenv("FFMPEG_PATH")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            for name in ("ffmpeg", "ffmpeg.exe"):
                binary = candidate / name
                if binary.is_file():
                    return binary
        raise FFmpegNotFound(
            f"FFMPEG_PATH is set to {override!r} but no ffmpeg binary was found "
            "there. Point it at the binary or its directory, or unset it to use PATH."
        )

    found = shutil.which("ffmpeg")
    if found:
        return Path(found)

    raise FFmpegNotFound(
        "ffmpeg not found on PATH and FFMPEG_PATH is not set. Install ffmpeg "
        "(the worker image must ship it) or set FFMPEG_PATH."
    )


def ffmpeg_dir() -> str | None:
    """Directory holding ffmpeg, for yt-dlp's ``ffmpeg_location``.

    Returns None when ffmpeg cannot be found, so a caller that only needs a
    single pre-muxed stream still works instead of failing at download time.
    yt-dlp accepts ``ffmpeg_location=None`` and falls back to PATH itself.
    """
    try:
        return str(resolve_ffmpeg().parent)
    except FFmpegNotFound:
        return None
