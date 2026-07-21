import os
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv

load_dotenv()

FFMPEG_PATH = os.path.normpath(os.getenv("FFMPEG_PATH"))

INPUT_DIR = Path("videos") / "input"


def download_youtube_video(url: str) -> str:
    """
    Downloads a YouTube video into videos/input/.

    Removes any existing videos in the folder before downloading.

    Returns:
        Path to the downloaded video.
    """

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old videos
    for file in INPUT_DIR.iterdir():
        if file.is_file():
            file.unlink()

    output_template = str(INPUT_DIR / "yt_video.%(ext)s")

    ydl_opts = {
        "ffmpeg_location": FFMPEG_PATH,

        # Best video + best audio
        "format": "bv*+ba/b",

        # Save as videos/input/yt_video.<ext>
        "outtmpl": output_template,

        # Always output mp4
        "merge_output_format": "mp4",

        "noplaylist": True,
        "continuedl": True,
        "overwrites": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file
    for file in INPUT_DIR.iterdir():
        if file.is_file():
            return str(file)

    raise FileNotFoundError("Video download failed.")


if __name__ == "__main__":
    url = input("Enter YouTube URL: ").strip()

    try:
        video_path = download_youtube_video(url)

        print("\nDownload successful!")
        print(video_path)

    except Exception as e:
        print(f"\nError: {e}")