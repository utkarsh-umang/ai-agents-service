"""Thumbnail generation prompts (shared by GPT Image and Gemini Nano Banana)."""

THUMBNAIL_SYSTEM_PROMPT_BODY = """You are an expert YouTube thumbnail designer \
specializing in high-CTR, viral-style thumbnails similar to top creators \
(MrBeast style).

You will receive:
1. A REFERENCE THUMNAIL - study its style, color palette, composition, \
text treatment, lighting mood, and emotional tone exactly
2. One or more BASE IMAGES - the actual subject(s) to feature

Your job: apply the visual style of the reference onto the base subject(s) \
to produce a new high-impact YouTube thumbnail.

Follow these rules strictly:
- Subject expression: engaging, lively, and readable at small sizes—confident, \
curious, amused, or subtle surprise—with clear eye contact. Match the reference's \
level of intensity; do not escalate beyond it. Avoid open-mouth screaming, distorted \
yelling, or extreme shock-face caricature unless the creative direction explicitly \
asks for that.
- Cinematic high-contrast lighting, glowing highlights on subject
- Vibrant saturated colors, especially skin tones and background contrast
- Subject large and sharp, centered or rule-of-thirds
- Slightly blurred background (depth of field), ultra-sharp subject
- Clean background that makes subject pop, remove distractions
- Smooth but realistic skin texture
- Optional: rim light or outline around subject for separation
- Mobile-first: readable at small sizes
- Hyper-realistic, high contrast, cinematic style"""


def build_thumbnail_system_prompt(shorts_or_reels: bool) -> str:
    """Full system rules including aspect/orientation for the output frame."""
    if shorts_or_reels:
        aspect = (
            "Output aspect ratio: 9:16 PORTRAIT (vertical)—taller than wide. "
            "This is for YouTube Shorts / Instagram Reels / TikTok: full vertical "
            "mobile frame, not landscape. Compose in strong vertical flow with "
            "subject and focal mass in the central column; keep faces and any text "
            "in a mobile-safe central band. Sharp detail at 2K-equivalent."
        )
    else:
        aspect = (
            "Output aspect ratio: 16:9 landscape (wider than tall), 2K resolution equivalent."
        )
    return f"{THUMBNAIL_SYSTEM_PROMPT_BODY}\n- {aspect}"


# Backwards compatibility: default landscape YouTube thumbnail system prompt.
THUMBNAIL_SYSTEM_PROMPT = build_thumbnail_system_prompt(False)


def build_user_instruction(
    title: str,
    include_title: bool,
    creative_comments: str,
    shorts_or_reels: bool = False,
) -> str:
    parts: list[str] = []

    if shorts_or_reels:
        parts.append(
            """CRITICAL — SHORTS / REELS (VERTICAL 9:16):
The final image MUST be portrait orientation (9:16), NOT 16:9 landscape.
Fill a tall narrow frame as on a phone held upright. Emphasize vertical composition:
stack visual weight along the height, keep the subject prominent in the middle third,
and ensure any text sits in the upper or center safe zone without tiny illegible lettering.
If the reference is landscape, reinterpret its style into this vertical format—do NOT output a wide landscape canvas."""
        )

    if include_title and title:
        parts.append(
            f"""TEXT TO INCLUDE:
- Add this title in BIG bold text (2-4 words max visible): "{title}"
- High contrast color: yellow, white, or red
- Thick font, easy to read on mobile
- Place text without covering the subject's face"""
        )
    else:
        parts.append("Do not add any text to the thumbnail.")

    if creative_comments:
        parts.append(f"Additional creative direction: {creative_comments}")

    parts.append(
        "Now generate the thumbnail applying the reference style to the subject."
    )

    return "\n\n".join(parts)
