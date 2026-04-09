"""Gemini (Nano Banana) thumbnail generation from remote image URLs.

URLs must be reachable by this process (public HTTP(S), presigned S3, etc.).
Private S3 objects without presigning will fail on fetch.
"""

from __future__ import annotations

import os


from google import genai
from google.genai import types

from ai_agents.agents.thumbnail_generator.prompts import (
    THUMBNAIL_SYSTEM_PROMPT,
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
    model_variant: str = "gemini-3.1-flash-image-preview",
) -> bytes:
    """
    Generate a thumbnail via Gemini image output. Returns raw image bytes.
    """
    client = genai.Client(api_key=_gemini_api_key())

    contents: list[str | types.Part] = [THUMBNAIL_SYSTEM_PROMPT]

    ref_bytes, ref_mime = fetch_and_encode(reference_image_url)
    contents.append(types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime))

    for url in base_image_urls:
        img_bytes, img_mime = fetch_and_encode(url)
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=img_mime))

    contents.append(
        build_user_instruction(title, include_title, creative_comments)
    )

    response = client.models.generate_content(
        model=model_variant,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                image_size="2K",
            ),
        ),
    )

    for part in response.parts or []:
        if part.inline_data is not None and part.inline_data.data is not None:
            return part.inline_data.data

    raise ValueError("Nano Banana returned no image in response")
