"""Short video generator — long-form video to short-form clip candidates.

Public API::

    from ai_agents.agents.short_video_generator import run_short_video_agent

    analysis = run_short_video_agent("https://www.youtube.com/watch?v=...")
    for clip in analysis.clips:
        ...  # clip.start_timestamp, clip.title, clip.why_it_works

**The agent returns a decision, never an artefact.** It reads the transcript,
picks the windows worth clipping, and hands back timestamps and reasoning. It
never downloads a video, runs ffmpeg, or touches storage — that half lives in
the consuming product (``app/services/short_video/`` in ``automation-tools``),
exactly as ``run_thumbnail_agent`` returns PNG bytes and lets the Tools backend
own S3.

Not re-exported from ``ai_agents`` itself: importing the top-level package must
not drag yt-dlp and the LangGraph stack in for callers that only want thumbnails.
Requires the ``shortvideo`` extra.
"""

from ai_agents.agents.short_video_generator.contracts import (
    ChannelVideos,
    Clip,
    VideoAnalysis,
    VideoMetadata,
)
from ai_agents.agents.short_video_generator.helpers.validate_links import (
    is_valid_youtube_url,
)
from ai_agents.agents.short_video_generator.nodes.get_short_clips import (
    run_short_video_agent,
)

__all__ = [
    "run_short_video_agent",
    "VideoAnalysis",
    "Clip",
    "VideoMetadata",
    "ChannelVideos",
    "is_valid_youtube_url",
]
