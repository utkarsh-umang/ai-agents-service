"""Gemini (Nano Banana) thumbnail generation from remote image URLs.

URLs must be reachable by this process (public HTTP(S), presigned S3, etc.).
Private S3 objects without presigning will fail on fetch.
"""

from __future__ import annotations

import os


from google import genai
from google.genai import types

from ai_agents.agents.thumbnail_generator.prompts import (
    build_thumbnail_system_prompt,
    build_user_instruction,
)
from ai_agents.agents.thumbnail_generator.utils import fetch_and_encode


def _gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise OSError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini image generation."
        )
    return key

def generate_with_nanobanana(
    reference_image_url: str,
    base_image_urls: list[str],
    title: str,
    include_title: bool,
    creative_comments: str,
    shorts_or_reels: bool = False,
    model_variant: str = "gemini-3-pro-image-preview",
) -> bytes:
    """
    Generate a thumbnail via Gemini image output. Returns raw image bytes.
    """
    # Explicit timeout: without one, a stalled connection is invisible to this
    # call and the only thing that would ever notice is the caller's own
    # task-level time limit, which depends on signal-delivery timing rather
    # than a normal, reliably-caught exception. 90s (90_000ms — this SDK's
    # HttpOptions.timeout is in milliseconds, not seconds) leaves headroom
    # under the 150s Celery soft limit for the image downloads/uploads around
    # this call.
    client = genai.Client(
        api_key=_gemini_api_key(),
        http_options=types.HttpOptions(timeout=90_000),
    )

    system_text = build_thumbnail_system_prompt(shorts_or_reels)
    contents: list[str | types.Part] = [system_text]

    ref_bytes, ref_mime = fetch_and_encode(reference_image_url)
    contents.append(types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime))

    for url in base_image_urls:
        img_bytes, img_mime = fetch_and_encode(url)
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=img_mime))

    contents.append(
        build_user_instruction(
            title, include_title, creative_comments, shorts_or_reels
        )
    )

    aspect_ratio = "9:16" if shorts_or_reels else "16:9"

    response = client.models.generate_content(
        model=model_variant,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size="2K",
            ),
        ),
    )

    for part in response.parts or []:
        if part.inline_data is not None and part.inline_data.data is not None:
            return part.inline_data.data

    raise ValueError("Nano Banana returned no image in response")
