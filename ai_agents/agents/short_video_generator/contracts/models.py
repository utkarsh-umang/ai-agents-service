from pydantic import BaseModel, HttpUrl, Field
from typing import Optional
from datetime import datetime

# final AI output format
class Clip(BaseModel):
    title: str
    start_timestamp: str
    end_timestamp: str
    duration_seconds: int
    topic: str
    why_it_works: str


class VideoAnalysis(BaseModel):
    video_url: HttpUrl
    clips: list[Clip]


# used in get short clips
class VideoMetadata(BaseModel):
    title: str
    description: str

    video_id: str
    url: HttpUrl

    channel_title: str
    channel_id: str

    published_at: datetime

    duration_iso: str
    duration_seconds: int

    views: int
    likes: int
    comments: int

    category_id: Optional[str] = None

    thumbnail: HttpUrl

    tags: list[str] = Field(default_factory=list)

    default_language: Optional[str] = None
    default_audio_language: Optional[str] = None

    licensed_content: bool
    definition: Optional[str] = None
    dimension: Optional[str] = None
    caption: Optional[str] = None
    projection: Optional[str] = None


class ChannelVideos(BaseModel):
    videos: list[VideoMetadata]