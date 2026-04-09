"""OpenAI GPT Image (image edit) — reference + base images in, transformed image out."""

from __future__ import annotations

import base64
from typing import Literal

from openai import OpenAI

from ai_agents.agents.thumbnail_generator.nanobanana_image_generator import fetch_and_encode
from ai_agents.agents.thumbnail_generator.prompts import (
    THUMBNAIL_SYSTEM_PROMPT,
    build_user_instruction,
)


def _filename_for_mime(mime: str, index: int, role: str) -> str:
    ext = ".jpg"
    if "png" in mime.lower():
        ext = ".png"
    elif "webp" in mime.lower():
        ext = ".webp"
    return f"{role}_{index}{ext}"


def build_gpt_image_prompt(
    title: str,
    include_title: bool,
    creative_comments: str,
) -> str:
    """Full text prompt: Nano-style system rules + per-request instructions."""
    user_part = build_user_instruction(title, include_title, creative_comments)
    return (
        f"{THUMBNAIL_SYSTEM_PROMPT}\n\n"
        "Image order: the FIRST file is the REFERENCE thumbnail (match its style). "
        "The NEXT file(s) are BASE subject photo(s) to keep as the main subject.\n\n"
        f"{user_part}"
    )


def generate_with_gpt_image(
    reference_image_url: str,
    base_image_urls: list[str],
    title: str,
    include_title: bool,
    creative_comments: str,
    model: str = "gpt-image-1",
    input_fidelity: Literal["high", "low"] = "high",
) -> tuple[bytes, str]:
    """
    True image-conditioned generation via ``images.edit``.

    Returns ``(image_bytes, prompt_used)``. GPT Image models return base64, not URLs.
    """
    prompt = build_gpt_image_prompt(title, include_title, creative_comments)

    ref_raw, ref_mime = fetch_and_encode(reference_image_url)
    files: list[tuple[str | None, bytes, str | None]] = [
        (_filename_for_mime(ref_mime, 0, "reference"), ref_raw, ref_mime),
    ]
    for i, url in enumerate(base_image_urls):
        raw, mime = fetch_and_encode(url)
        files.append(
            (_filename_for_mime(mime, i + 1, "base"), raw, mime),
        )

    client = OpenAI()
    response = client.images.edit(
        model=model,
        image=files,
        prompt=prompt,
        size="1536x1024",
        input_fidelity=input_fidelity,
        quality="high",
        output_format="png",
        n=1,
    )

    if not response.data or not response.data[0].b64_json:
        raise ValueError("GPT Image returned no image (expected b64_json)")
    return base64.b64decode(response.data[0].b64_json), prompt
