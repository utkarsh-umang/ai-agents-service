"""Thumbnail generation orchestration."""

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from ai_agents.agents.thumbnail_generator.flux_kontext_generator import generate_with_flux_kontext
from ai_agents.agents.thumbnail_generator.gpt_image_generator import generate_with_gpt_image
from ai_agents.agents.thumbnail_generator.nanobanana_image_generator import generate_with_nanobanana
from ai_agents.agents.thumbnail_generator.prompts import (
    build_thumbnail_system_prompt,
    build_user_instruction,
)

ThumbnailModel = Literal["gptimage", "nanobanana", "fluxkontext"]


def run_thumbnail_agent(
    model: ThumbnailModel,
    reference_image_url: str,
    base_image_urls: list[str],
    title: str,
    include_title: bool,
    creative_comments: str,
    shorts_or_reels: bool = False,
    num_candidates: int = 1,
) -> dict[str, object]:
    """
    Route to OpenAI GPT Image, Gemini (Nano Banana), or Flux Kontext.

    Generates ``num_candidates`` images concurrently — all three provider
    calls are blocking HTTP, so candidates run in a thread pool rather than
    sequentially (keeps wall-clock roughly flat as num_candidates grows,
    which matters for the Celery task's time limit).

    Returns ``images`` (``list[bytes]``, length ``num_candidates``) and
    ``prompt_used`` (``str``, the instruction text sent on the first candidate).
    """

    def _one_gptimage() -> tuple[bytes, str]:
        return generate_with_gpt_image(
            reference_image_url=reference_image_url,
            base_image_urls=base_image_urls,
            title=title,
            include_title=include_title,
            creative_comments=creative_comments,
            shorts_or_reels=shorts_or_reels,
        )

    def _one_nanobanana() -> tuple[bytes, str]:
        image_bytes = generate_with_nanobanana(
            reference_image_url=reference_image_url,
            base_image_urls=base_image_urls,
            title=title,
            include_title=include_title,
            creative_comments=creative_comments,
            shorts_or_reels=shorts_or_reels,
        )
        prompt_used = (
            f"{build_thumbnail_system_prompt(shorts_or_reels)}\n\n"
            f"{build_user_instruction(title, include_title, creative_comments, shorts_or_reels)}"
        )
        return image_bytes, prompt_used

    def _one_fluxkontext() -> tuple[bytes, str]:
        return generate_with_flux_kontext(
            reference_image_url=reference_image_url,
            base_image_urls=base_image_urls,
            title=title,
            include_title=include_title,
            creative_comments=creative_comments,
            shorts_or_reels=shorts_or_reels,
        )

    generators = {
        "gptimage": _one_gptimage,
        "nanobanana": _one_nanobanana,
        "fluxkontext": _one_fluxkontext,
    }
    generate_one = generators[model]

    count = max(1, num_candidates)
    with ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(lambda _: generate_one(), range(count)))

    images = [image_bytes for image_bytes, _ in results]
    prompt_used = results[0][1] if results else ""

    preview = prompt_used if len(prompt_used) <= 2000 else prompt_used[:2000] + "…"
    print(f"[{model}] generated {len(images)} candidate(s); prompt ({len(prompt_used)} chars):\n{preview}\n")

    return {"images": images, "prompt_used": prompt_used}
