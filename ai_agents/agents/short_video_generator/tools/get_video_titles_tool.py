import os
from dotenv import load_dotenv
from datetime import datetime
from googleapiclient.discovery import build

import requests
from isodate import parse_duration
from urllib.parse import urlparse

from langchain.tools import tool

from contracts import *

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY
)



def get_channel_id(channel_url: str) -> str:
    """
    Get YouTube channel ID from:
    - https://www.youtube.com/@handle
    - https://www.youtube.com/channel/UCxxxx
    """

    parsed = urlparse(channel_url)

    path = parsed.path.strip("/")

    # CASE 1:
    # Already a channel ID URL
    # /channel/UCxxxx

    if path.startswith("channel/"):

        channel_id = path.split("/")[-1]

        return channel_id

    # CASE 2:
    # Handle URL
    # /@username

    if path.startswith("@"):

        handle = path.replace("@", "")

        url = "https://www.googleapis.com/youtube/v3/search"

        params = {
            "part": "snippet",
            "q": handle,
            "type": "channel",
            "maxResults": 1,
            "key": YOUTUBE_API_KEY,
        }

        response = requests.get(url, params=params)

        data = response.json()

        if "items" not in data or len(data["items"]) == 0:
            raise Exception("Channel not found")

        return data["items"][0]["snippet"]["channelId"]

    raise ValueError("Unsupported YouTube channel URL")

@tool
def get_video_titles(channel_url: str, limit: int = 5) -> ChannelVideos:
    """
    Fetch latest long-form YouTube videos
    from a YouTube channel URL.
    Only returns videos longer than 3 minutes.
    """

    channel_id = get_channel_id(channel_url)

    channel_response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    uploads_playlist_id = (
        channel_response["items"][0]
        ["contentDetails"]
        ["relatedPlaylists"]
        ["uploads"]
    )

    playlist_response = youtube.playlistItems().list(
        part="snippet",
        playlistId=uploads_playlist_id,
        maxResults=100
    ).execute()

    video_ids = []

    for item in playlist_response["items"]:
        video_id = item["snippet"]["resourceId"]["videoId"]
        video_ids.append(video_id)

    videos: list[VideoMetadata] = []

    for i in range(0, len(video_ids), 50):

        batch_ids = video_ids[i:i + 50]

        videos_response = youtube.videos().list(
            part="snippet,contentDetails,statistics,status",
            id=",".join(batch_ids)
        ).execute()

        for item in videos_response["items"]:

            duration = item["contentDetails"]["duration"]
            seconds = int(parse_duration(duration).total_seconds())

            if seconds < 180:
                continue

            snippet = item["snippet"]
            statistics = item.get("statistics", {})
            content = item["contentDetails"]

            published = snippet["publishedAt"]

            videos.append(
                VideoMetadata(
                    title=snippet["title"],
                    description=snippet["description"],

                    video_id=item["id"],
                    url=f"https://www.youtube.com/watch?v={item['id']}",

                    channel_title=snippet["channelTitle"],
                    channel_id=snippet["channelId"],

                    published_at=datetime.fromisoformat(
                        published.replace("Z", "+00:00")
                    ),

                    duration_iso=duration,
                    duration_seconds=seconds,

                    views=int(statistics.get("viewCount", 0)),
                    likes=int(statistics.get("likeCount", 0)),
                    comments=int(statistics.get("commentCount", 0)),

                    category_id=snippet.get("categoryId"),

                    thumbnail=snippet["thumbnails"]["high"]["url"],

                    tags=snippet.get("tags", []),

                    default_language=snippet.get("defaultLanguage"),
                    default_audio_language=snippet.get("defaultAudioLanguage"),

                    licensed_content=content.get("licensedContent", False),

                    definition=content.get("definition"),
                    dimension=content.get("dimension"),
                    caption=content.get("caption"),
                    projection=content.get("projection"),
                )
            )
            
            if len(videos) >= limit:
                return ChannelVideos(videos=videos)
            
    print(videos)

    return ChannelVideos(videos=videos)