import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

from contracts import VideoAnalysis

load_dotenv()

FFMPEG_PATH = os.path.normpath(os.getenv("FFMPEG_PATH"))
FFMPEG_EXE = os.path.join(FFMPEG_PATH, "ffmpeg.exe")

OUTPUT_DIR = Path("videos/output")


def create_clips(video_path: str, analysis) -> list[str]:
    """
    Create clips from a downloaded video using VideoAnalysis.

    Args:
        video_path: Path to downloaded video.
        analysis: VideoAnalysis object.

    Returns:
        List of created clip paths.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old clips
    for file in OUTPUT_DIR.glob("*"):
        if file.is_file():
            file.unlink()

    video_path = Path(video_path)
    ext = video_path.suffix

    results = []

    print(f"Using video: {video_path.name}")

    for i, clip in enumerate(analysis.clips, start=1):

        output_file = OUTPUT_DIR / f"clip_{i}{ext}"

        command = [
            FFMPEG_EXE,
            "-y",
            "-ss", clip.start_timestamp,
            "-to", clip.end_timestamp,
            "-i", str(video_path),
            "-c", "copy",
            str(output_file),
        ]

        print(f"Creating clip {i}")

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        results.append({
            "title": clip.title,
            "topic": clip.topic,
            "why_it_works": clip.why_it_works,
            "start_timestamp": clip.start_timestamp,
            "end_timestamp": clip.end_timestamp,
            "duration_seconds": clip.duration_seconds,
            "video_path": str(output_file),
        })

    return results


if __name__ == "__main__":

    video_path = "videos/input/yt_video.mp4"

    analysis = VideoAnalysis.model_validate(
        {
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "clips": [
                {
                    "title": "Clip 1",
                    "start_timestamp": "00:00:30",
                    "end_timestamp": "00:00:56",
                    "duration_seconds": 26,
                    "topic": "Test",
                    "why_it_works": "Testing",
                },
                {
                    "title": "Clip 2",
                    "start_timestamp": "00:03:00",
                    "end_timestamp": "00:04:05",
                    "duration_seconds": 65,
                    "topic": "Test",
                    "why_it_works": "Testing",
                },
            ],
        }
    )

    try:
        paths = create_clips(video_path, analysis)

        print("\nCreated clips:")
        for path in paths:
            print(path)

    except Exception as e:
        print(f"Error: {e}")