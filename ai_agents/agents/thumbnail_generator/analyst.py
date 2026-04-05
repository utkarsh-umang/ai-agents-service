"""GPT-4o vision analyst: images in → single DALL-E 3–ready text prompt out."""

from __future__ import annotations

import base64

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from ai_agents.agents.thumbnail_generator.generator import fetch_and_encode

ANALYST_SYSTEM_PROMPT = """You are a senior graphic designer writing a \
**single paragraph** of prose for DALL-E 3.

The client uploads (1) a reference thumbnail for **style only** and (2) \
reference photos for **layout inspiration**. Your paragraph must describe \
**original illustrated artwork** in the loud, graphic YouTube-thumbnail \
tradition—bold color blocks, clear focal subject, readable title treatment. \
Treat any people as **generic stylized illustrated characters** (similar \
pose, clothing colors, and energy)—not a photorealistic portrait of a \
specific real individual.

In your paragraph, weave in: palette and saturation from the reference, \
composition (placement, negative space), lighting mood, background \
treatment, and on-screen text style if the brief asks for text.

End the paragraph with exactly this closing: \
"16:9 aspect ratio, polished illustrated YouTube thumbnail, vibrant color, \
crisp graphic design, HDR pop."

Rules: one flowing paragraph only—no bullet lists, no preamble, no \
meta-commentary, no apologies."""


def analyst_output_looks_like_refusal(text: str) -> bool:
    """True if the vision model declined instead of emitting a DALL-E prompt."""
    t = text.strip().lower()
    if len(t) < 400 and any(
        phrase in t
        for phrase in (
            "i'm sorry",
            "i am sorry",
            "i cannot",
            "i can't",
            "can't assist",
            "cannot assist",
            "can't help",
            "cannot help",
            "unable to",
            "not able to",
            "can't comply",
            "cannot comply",
            "as an ai",
            "i apologize",
        )
    ):
        return True
    return False


def _bytes_to_data_url(raw: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def build_analyst_message(
    reference_image_url: str,
    base_image_urls: list[str],
    title: str,
    include_title: bool,
    creative_comments: str,
) -> list[BaseMessage]:
    system = SystemMessage(content=ANALYST_SYSTEM_PROMPT)

    content: list[str | dict] = []

    ref_raw, ref_mime = fetch_and_encode(reference_image_url)
    content.append(
        {"type": "text", "text": "REFERENCE THUMBNAIL (extract style from this):"}
    )
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": _bytes_to_data_url(ref_raw, ref_mime),
                "detail": "low",
            },
        }
    )

    content.append(
        {
            "type": "text",
            "text": (
                f"CLIENT REFERENCE PHOTOS ({len(base_image_urls)}) for pose/outfit "
                "and energy—translate into stylized illustrated characters:"
            ),
        }
    )
    for url in base_image_urls:
        raw, mime = fetch_and_encode(url)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _bytes_to_data_url(raw, mime),
                    "detail": "low",
                },
            }
        )

    instruction_parts: list[str] = []
    if include_title and title:
        instruction_parts.append(
            f'Include this title text in the thumbnail: "{title}". '
            f"Bold, high-contrast (yellow/white/red), thick font, "
            f"placed without covering the subject's face."
        )
    else:
        instruction_parts.append("Do not include any text in the thumbnail.")

    if creative_comments:
        instruction_parts.append(f"Additional direction: {creative_comments}")

    content.append({"type": "text", "text": "\n".join(instruction_parts)})

    human = HumanMessage(content=content)
    return [system, human]


def run_analyst(
    reference_image_url: str,
    base_image_urls: list[str],
    title: str,
    include_title: bool,
    creative_comments: str,
) -> str:
    """Step 1: GPT-4o vision → single DALL-E 3 prompt string."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.35)
    parser = StrOutputParser()
    messages = build_analyst_message(
        reference_image_url,
        base_image_urls,
        title,
        include_title,
        creative_comments,
    )
    chain = llm | parser
    return str(chain.invoke(messages))
