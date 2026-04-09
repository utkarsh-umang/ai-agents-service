"""Thumbnail generation orchestration."""

from typing import Literal

from ai_agents.agents.thumbnail_generator.nanobanana_image_generator import generate_with_nanobanana
from ai_agents.agents.thumbnail_generator.gpt_image_generator import generate_with_gpt_image
from ai_agents.agents.thumbnail_generator.prompts import build_user_instruction


def run_thumbnail_agent(
    model: Literal["gptimage", "nanobanana"],
    reference_image_url: str,
    base_image_urls: list[str],
    title: str,
    include_title: bool,
    creative_comments: str,
) -> dict[str, object]:
    """
    Route to OpenAI GPT Image (true image edit) or Gemini (Nano Banana).

    Both return ``image_bytes`` and ``prompt_used`` (instruction text for tracing).
    """
    if model == "nanobanana":
        image_bytes = generate_with_nanobanana(
            reference_image_url=reference_image_url,
            base_image_urls=base_image_urls,
            title=title,
            include_title=include_title,
            creative_comments=creative_comments,
        )
        prompt_used = build_user_instruction(
            title, include_title, creative_comments
        )
        print(f"[nanobanana] instruction:\n{prompt_used}\n")
        return {"image_bytes": image_bytes, "prompt_used": prompt_used}

    return run_thumbnail_agent_gpt_image(
        reference_image_url=reference_image_url,
        base_image_urls=base_image_urls,
        title=title,
        include_title=include_title,
        creative_comments=creative_comments,
    )


def run_thumbnail_agent_gpt_image(
    reference_image_url: str,
    base_image_urls: list[str],
    title: str,
    include_title: bool,
    creative_comments: str,
    model: str = "gpt-image-1",
) -> dict[str, object]:
    """
    OpenAI GPT Image path: multimodal edit (reference + base images + prompt).

    Returns ``image_bytes`` (PNG) and ``prompt_used``. GPT Image returns
    base64, not a hosted URL.
    """
    image_bytes, prompt_used = generate_with_gpt_image(
        reference_image_url=reference_image_url,
        base_image_urls=base_image_urls,
        title=title,
        include_title=include_title,
        creative_comments=creative_comments,
        model=model,
    )

    preview = prompt_used if len(prompt_used) <= 2000 else prompt_used[:2000] + "…"
    print(f"[gpt-image] prompt ({len(prompt_used)} chars):\n{preview}\n")

    return {
        "image_bytes": image_bytes,
        "prompt_used": prompt_used,
    }
