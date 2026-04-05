THUMBNAIL_SYSTEM_PROMPT = """You are an expert YouTube thumbnail designer \
specializing in high-CTR, viral-style thumbnails similar to top creators \
(MrBeast style).

You will receive:
1. A REFERENCE THUMBNAIL - study its style, color palette, composition, \
text treatment, lighting mood, and emotional tone exactly
2. One or more BASE IMAGES - the actual subject(s) to feature

Your job: apply the visual style of the reference onto the base subject(s) \
to produce a new high-impact YouTube thumbnail.

Follow these rules strictly:
- Enhance facial expressions: more exaggerated, emotional (shock/excitement/intensity)
- Cinematic high-contrast lighting, glowing highlights on subject
- Vibrant saturated colors, especially skin tones and background contrast
- Subject large and sharp, centered or rule-of-thirds
- Slightly blurred background (depth of field), ultra-sharp subject
- Clean background that makes subject pop, remove distractions
- Smooth but realistic skin texture
- Optional: rim light or outline around subject for separation
- Mobile-first: readable at small sizes
- Hyper-realistic, high contrast, cinematic style
- Output aspect ratio: 16:9, 2K resolution"""


def build_user_instruction(
    title: str,
    include_title: bool,
    creative_comments: str,
) -> str:
    parts: list[str] = []

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
