"""Short video generator — long-form video to short-form clip candidates.

Public API::

    from ai_agents.agents.short_video_generator import run_short_video_agent

    analysis = run_short_video_agent("https://www.youtube.com/watch?v=...")
    for clip in analysis.clips:
        ...  # clip.start_timestamp, clip.title, clip.why_it_works

**The agent returns a decision, never an artefact.** It produces timestamps and
reasoning; downloading the source and cutting the clips are the caller's job.
This mirrors the thumbnail agent, where ``run_thumbnail_agent`` hands back bytes
and the tool owns storage.

The media-side helpers live in ``helpers`` and are exported here for the
consumer that owns that half of the pipeline:

    from ai_agents.agents.short_video_generator import (
        job_workspace, download_youtube_video, create_clips, short_video_clip_key,
    )

Every one of them takes an explicit directory. Nothing is shared between jobs
and nothing is deleted that the caller did not create — see ``helpers.workspace``
for why that matters.

Not re-exported from ``ai_agents`` itself: importing the top-level package must
not drag yt-dlp and the LangGraph stack in for callers that only want thumbnails.
"""

from ai_agents.agents.short_video_generator.contracts import (
    ChannelVideos,
    Clip,
    VideoAnalysis,
    VideoMetadata,
)
from ai_agents.agents.short_video_generator.helpers.cut_video import create_clips
from ai_agents.agents.short_video_generator.helpers.download import (
    download_youtube_video,
)
from ai_agents.agents.short_video_generator.helpers.storage_keys import (
    short_video_clip_key,
    short_video_source_key,
)
from ai_agents.agents.short_video_generator.helpers.validate_links import (
    is_valid_youtube_url,
)
from ai_agents.agents.short_video_generator.helpers.workspace import job_workspace
from ai_agents.agents.short_video_generator.nodes.get_short_clips import (
    run_short_video_agent,
)

__all__ = [
    # the agent
    "run_short_video_agent",
    # contracts
    "VideoAnalysis",
    "Clip",
    "VideoMetadata",
    "ChannelVideos",
    # media pipeline
    "job_workspace",
    "download_youtube_video",
    "create_clips",
    "short_video_clip_key",
    "short_video_source_key",
    "is_valid_youtube_url",
]
