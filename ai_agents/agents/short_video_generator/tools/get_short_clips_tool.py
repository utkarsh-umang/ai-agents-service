import json
import logging
import re
import tempfile
from datetime import timedelta
from pathlib import Path

import yt_dlp
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi

from ai_agents.agents.short_video_generator.contracts import VideoAnalysis

logger = logging.getLogger(__name__)

CLIP_MODEL = "gpt-4o-mini"
# Structured extraction against a fixed schema — creativity here buys nothing and
# costs schema adherence. The prototype ran this at 0.7.
CLIP_TEMPERATURE = 0.2

_llm = None


def _get_llm() -> ChatOpenAI:
    """The clip-selection model, constructed on first use and cached.

    Constructed lazily so importing this module does not require OPENAI_API_KEY
    or open a client; that made the agent unimportable wherever the key was
    absent, and paid the cost even for callers that never selected clips.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=CLIP_MODEL, temperature=CLIP_TEMPERATURE)
    return _llm

SHORTS_PROMPT = """

# ROLE

You are an expert short-form content editor and viral content strategist.

Your expertise is identifying high-retention moments from long-form content and converting them into engaging YouTube Shorts, Instagram Reels, and TikTok clips.

You understand:

* Viewer retention
* Watch time optimization
* Viral hooks
* Storytelling
* Emotional engagement
* Educational content
* Social sharing behavior

# TASK

Analyze the provided YouTube transcript and identify the BEST clips that can be repurposed into short-form video content.

Your goal is to maximize:

* Retention
* Watch time
* Shares
* Comments
* Replays
* Audience engagement

# INSTRUCTIONS

For each selected clip:

1. Select a clip between 60 and 80 seconds.
2. Use timestamps that exist within the transcript.
3. Create a short, attention-grabbing title.
4. Explain what the clip discusses.
5. Explain why the clip is likely to perform well as short-form content.

Prioritize clips that contain:

* Strong hooks
* Surprising facts
* Emotional moments
* Controversial opinions
* Valuable insights
* Storytelling moments
* Funny moments
* Actionable advice
* Contrarian viewpoints
* Unexpected outcomes
* Personal experiences with a payoff

Return 5-10 clips if enough high-quality clips exist.

Be selective.

Quality is more important than quantity.

Clips should not significantly overlap with each other.

# GUARDRAILS

Do NOT select:

* Introductions
* Greetings
* Sponsorship segments
* Promotional content
* Housekeeping announcements
* Outros
* Repetitive sections
* Low-energy sections
* Context-heavy sections that do not work independently

Do NOT invent timestamps.

Do NOT create clips longer than 80 seconds.

Do NOT create clips shorter than 60 seconds.

Do NOT return explanations outside the requested JSON.

Return ONLY valid JSON.

# EXAMPLE

Example Output:

{{
"clips": [
{{
"start_timestamp": "00:01:15",
"end_timestamp": "00:02:25",
"duration_seconds": 70,
"title": "The Biggest Mistake Most People Make",
"topic": "A common mistake that prevents people from improving.",
"why_it_works": "Strong hook, practical insight, and broad audience appeal."
}}
]
}}

# OUTPUT FORMAT

Return ONLY JSON using the following schema:

{{
"clips": [
{{
"start_timestamp": "HH:MM:SS",
"end_timestamp": "HH:MM:SS",
"duration_seconds": <integer_between_60_and_80>,
"title": "Short catchy title",
"topic": "What this clip talks about",
"why_it_works": "Why this segment would perform well as a short-form video"
}}
]
}}

# TRANSCRIPT

{transcript}
"""



def extract_video_id(url: str) -> str:
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    raise ValueError("Could not extract video ID")


def get_transcript_from_captions(video_url: str) -> str:
    """
    Downloads YouTube captions using yt-dlp and returns the transcript
    in the same format as youtube-transcript-api.

    The .vtt lands in a private temp directory that is removed on return, so
    concurrent jobs can never read or delete each other's captions. The
    prototype used a fixed ``videos/captions/`` and emptied it on entry.
    """

    with tempfile.TemporaryDirectory(prefix="svg-captions-") as tmp:
        caption_dir = Path(tmp)

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            "outtmpl": str(caption_dir / "captions"),
            "overwrites": True,
            "noplaylist": True,
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        return _parse_vtt_dir(caption_dir)


def _parse_vtt_dir(caption_dir: Path) -> str:
    """Parse the first .vtt in ``caption_dir`` into the shared transcript format."""
    vtt_files = list(caption_dir.glob("*.vtt"))

    if not vtt_files:
        raise Exception("No captions available.")

    transcript_parts = []

    with open(vtt_files[0], "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if "-->" in line:

            start, end = line.split("-->")

            start = start.strip().replace(".", ",")
            end = end.strip().split()[0].replace(".", ",")

            text_lines = []
            i += 1

            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            text = " ".join(text_lines)

            # Remove duplicated words that appear in some auto captions
            words = text.split()
            cleaned = []

            for word in words:
                if not cleaned or cleaned[-1] != word:
                    cleaned.append(word)

            text = " ".join(cleaned)

            chunk = (
                f"[{start} --> {end}]\n"
                f"{text}\n\n"
            )

            transcript_parts.append(chunk)

        i += 1

    return "".join(transcript_parts)

def seconds_to_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))


def get_transcript(video_url: str) -> str:
    """
    First tries YouTubeTranscriptApi.
    If that fails for any reason, falls back to downloading captions via yt-dlp.

    Returns the transcript; writes nothing to disk. The prototype also wrote
    ``transcript.txt`` into the current working directory, which two concurrent
    jobs would overwrite for each other — and which silently littered whatever
    directory the worker happened to start in.
    """

    video_id = extract_video_id(video_url)
    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(video_id, languages=["en"])

        transcript_parts = []
        for item in transcript:
            start = seconds_to_timestamp(item.start)
            end = seconds_to_timestamp(item.start + item.duration)
            transcript_parts.append(f"[{start} --> {end}]\n{item.text}\n\n")

        logger.info("transcript source=youtube_transcript_api video_id=%s", video_id)
        return "".join(transcript_parts)

    except Exception as exc:
        logger.info(
            "transcript source=captions_fallback video_id=%s api_error=%s",
            video_id,
            exc,
        )
        return get_transcript_from_captions(video_url)

def extract_json_from_response(response):

    result = response.content

    if isinstance(result, list):
        result = result[0]["text"]

    result = result.replace("```json", "")
    result = result.replace("```", "")
    result = result.strip()

    return json.loads(result)


@tool
def get_shorts_clips(video_url: str):
    """
    Analyze a YouTube video and return
    the best clips for YouTube Shorts,
    Instagram Reels and TikTok.
    """

    transcript_text = get_transcript(video_url)

    query = SHORTS_PROMPT.format(
        transcript=transcript_text
    )

    response = _get_llm().invoke([
        HumanMessage(content=query)
    ])

    data = extract_json_from_response(response)

    data["video_url"] = video_url
    
    analysis = VideoAnalysis.model_validate(data)

    return analysis.model_dump(mode="json")