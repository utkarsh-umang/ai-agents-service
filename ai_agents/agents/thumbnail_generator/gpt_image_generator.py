"""OpenAI GPT Image (image edit) — reference + base images in, transformed image out."""

from __future__ import annotations

import base64
from typing import Literal

from openai import OpenAI

from ai_agents.agents.thumbnail_generator.prompts import (
    build_thumbnail_system_prompt,
    build_user_instruction,
)
from ai_agents.agents.thumbnail_generator.utils import fetch_and_encode


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
    shorts_or_reels: bool = False,
) -> str:
    """Full text prompt: shared system rules + GPT image-order hint + user brief."""
    system = build_thumbnail_system_prompt(shorts_or_reels)
    user_part = build_user_instruction(
        title, include_title, creative_comments, shorts_or_reels
    )
    return (
        f"{system}\n\n"
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
    shorts_or_reels: bool = False,
    model: str = "gpt-image-2",
    # gpt-image-1 supports `input_fidelity`; gpt-image-2 rejects it outright
    # (HTTP 400 invalid_input_fidelity_model) — only pass it for gpt-image-1.
    input_fidelity: Literal["high", "low"] | None = "high",
) -> tuple[bytes, str]:
    """
    True image-conditioned generation via ``images.edit``.

    Returns ``(image_bytes, prompt_used)``. GPT Image models return base64, not URLs.
    """
    prompt = build_gpt_image_prompt(
        title, include_title, creative_comments, shorts_or_reels
    )
    size = "1024x1536" if shorts_or_reels else "1536x1024"

    ref_raw, ref_mime = fetch_and_encode(reference_image_url)
    files: list[tuple[str | None, bytes, str | None]] = [
        (_filename_for_mime(ref_mime, 0, "reference"), ref_raw, ref_mime),
    ]
    for i, url in enumerate(base_image_urls):
        raw, mime = fetch_and_encode(url)
        files.append(
            (_filename_for_mime(mime, i + 1, "base"), raw, mime),
        )

    edit_kwargs: dict[str, object] = dict(
        model=model,
        image=files,
        prompt=prompt,
        size=size,
        quality="high",
        output_format="png",
        n=1,
    )
    if not model.startswith("gpt-image-2") and input_fidelity is not None:
        edit_kwargs["input_fidelity"] = input_fidelity

    client = OpenAI()
    response = client.images.edit(**edit_kwargs)

    if not response.data or not response.data[0].b64_json:
        raise ValueError("GPT Image returned no image (expected b64_json)")
    return base64.b64decode(response.data[0].b64_json), prompt
