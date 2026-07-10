"""Black Forest Labs Flux Kontext Pro — image-conditioned edit (style ref + subject).

Submit -> poll -> download flow (BFL has no official Python SDK; this is the
pattern their own docs recommend). ``input_image`` carries the subject/base
photo (the identity to preserve); ``input_image_2`` (Kontext's multiref slot,
flagged experimental by BFL) carries the style reference when a subject photo
is present. Only the first base image is used — Kontext takes discrete named
image slots, not a list, so multiple base images aren't supported here.
"""

from __future__ import annotations

import base64
import os
import time

import httpx

from ai_agents.agents.thumbnail_generator.prompts import (
    build_thumbnail_system_prompt,
    build_user_instruction,
)
from ai_agents.agents.thumbnail_generator.utils import fetch_and_encode

_SUBMIT_URL = "https://api.bfl.ai/v1/flux-kontext-pro"
_POLL_INTERVAL_SECONDS = 1.0
_POLL_TIMEOUT_SECONDS = 75.0
_TERMINAL_FAILURE_STATUSES = {"Error", "Failed", "Request Moderated", "Content Moderated"}


def _bfl_api_key() -> str:
    key = os.environ.get("BFL_API_KEY")
    if not key:
        raise OSError("Set BFL_API_KEY for Flux Kontext image generation.")
    return key


def _build_prompt(
    title: str,
    include_title: bool,
    creative_comments: str,
    shorts_or_reels: bool,
    has_base_image: bool,
) -> str:
    system = build_thumbnail_system_prompt(shorts_or_reels)
    user_part = build_user_instruction(title, include_title, creative_comments, shorts_or_reels)
    if has_base_image:
        image_note = (
            "You are editing IMAGE 1 (the subject) to match the visual style of "
            "IMAGE 2 (the reference thumbnail)."
        )
    else:
        image_note = (
            "IMAGE 1 is the reference thumbnail. Invent a subject matching the "
            "creative direction below, in the reference's exact style."
        )
    return f"{system}\n\n{image_note}\n\n{user_part}"


def generate_with_flux_kontext(
    reference_image_url: str,
    base_image_urls: list[str],
    title: str,
    include_title: bool,
    creative_comments: str,
    shorts_or_reels: bool = False,
) -> tuple[bytes, str]:
    """Submit a Flux Kontext Pro edit, poll until ready, download the result.

    Returns ``(image_bytes, prompt_used)``.
    """
    has_base_image = bool(base_image_urls)
    prompt = _build_prompt(title, include_title, creative_comments, shorts_or_reels, has_base_image)

    payload: dict[str, object] = {
        "prompt": prompt,
        "aspect_ratio": "9:16" if shorts_or_reels else "16:9",
        "output_format": "png",
        "safety_tolerance": 2,
    }

    ref_bytes, _ = fetch_and_encode(reference_image_url)
    ref_b64 = base64.b64encode(ref_bytes).decode()

    if has_base_image:
        base_bytes, _ = fetch_and_encode(base_image_urls[0])
        payload["input_image"] = base64.b64encode(base_bytes).decode()
        payload["input_image_2"] = ref_b64
    else:
        payload["input_image"] = ref_b64

    headers = {"x-key": _bfl_api_key(), "Content-Type": "application/json"}

    with httpx.Client(timeout=30.0) as client:
        submit_resp = client.post(_SUBMIT_URL, json=payload, headers=headers)
        submit_resp.raise_for_status()
        submitted = submit_resp.json()
        request_id = submitted["id"]
        polling_url = submitted["polling_url"]

        deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
        while True:
            poll_resp = client.get(
                polling_url,
                headers={"accept": "application/json", "x-key": _bfl_api_key()},
                params={"id": request_id},
            )
            poll_resp.raise_for_status()
            result = poll_resp.json()
            status = result.get("status")

            if status == "Ready":
                sample_url = result["result"]["sample"]
                image_bytes, _ = fetch_and_encode(sample_url)
                return image_bytes, prompt

            if status in _TERMINAL_FAILURE_STATUSES:
                raise ValueError(
                    f"Flux Kontext generation failed: status={status} details={result.get('details')}"
                )

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Flux Kontext polling timed out after {_POLL_TIMEOUT_SECONDS}s "
                    f"(last status={status})"
                )

            time.sleep(_POLL_INTERVAL_SECONDS)
