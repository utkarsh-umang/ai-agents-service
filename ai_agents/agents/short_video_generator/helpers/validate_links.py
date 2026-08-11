import re

YOUTUBE_URL_REGEX = re.compile(
    r"""
    ^
    (?:https?://)?                              # optional http/https
    (?:
        (?:www|m|music)\.                       # optional subdomain
    )?
    (?:
        youtube\.com|
        youtu\.be
    )
    /
    (?:
        # Short URL
        [A-Za-z0-9_-]{11}

        |

        # watch?v=VIDEO_ID
        watch\?(?:.*&)?v=[A-Za-z0-9_-]{11}(?:[&#?].*)?

        |

        # embed/VIDEO_ID
        embed/[A-Za-z0-9_-]{11}(?:[/?#].*)?

        |

        # v/VIDEO_ID
        v/[A-Za-z0-9_-]{11}(?:[/?#].*)?

        |

        # shorts/VIDEO_ID
        shorts/[A-Za-z0-9_-]{11}(?:[/?#].*)?

        |

        # live/VIDEO_ID
        live/[A-Za-z0-9_-]{11}(?:[/?#].*)?

        |

        # channel/CHANNEL_ID
        channel/UC[A-Za-z0-9_-]{22}(?:[/?#].*)?

        |

        # @handle
        @[A-Za-z0-9._-]+(?:[/?#].*)?

        |

        # c/customname
        c/[A-Za-z0-9._-]+(?:[/?#].*)?

        |

        # user/username
        user/[A-Za-z0-9._-]+(?:[/?#].*)?
    )
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def is_valid_youtube_url(url: str) -> bool:
    """Returns True if the URL is a valid YouTube video or channel URL."""
    return bool(YOUTUBE_URL_REGEX.match(url.strip()))


if __name__ == "__main__":
    test_urls = [
        # Valid video URLs
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=30",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/v/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",

        # Valid channel URLs
        "https://www.youtube.com/@MrBeast",
        "https://youtube.com/@OpenAI",
        "https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA",
        "https://www.youtube.com/c/GoogleDevelopers",
        "https://www.youtube.com/user/PewDiePie",

        # Invalid URLs
        "https://google.com",
        "https://youtube.com/",
        "https://youtube.com/playlist?list=PL123456",
        "https://youtube.com/results?search_query=test",
        "hello world",
        "https://youtu.be/abcd",
        "https://www.youtube.com/watch?v=123",
    ]

    for url in test_urls:
        print(f"{is_valid_youtube_url(url):5}  {url}")