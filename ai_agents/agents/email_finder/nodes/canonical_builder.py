from __future__ import annotations

import re
from typing import Any
import litellm
from ai_agents.core.llm import langfuse, get_prompt
from ai_agents.agents.email_finder.state import (
    EmailFinderState,
    CanonicalLead,
    Identity,
    SocialLinks,
    SourceType,
    LeadStatus,
)

# Known social platform domains for URL classification
SOCIAL_PLATFORM_DOMAINS = {
    "facebook.com": "facebook",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "instagram.com": "instagram",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "linkedin.com": "linkedin",
}

FALLBACK_PROMPT = """
You are a data normalization assistant.

You will receive a raw row from a CSV file and a source type.
Your job is to extract and classify the fields into a structured format.

Source type: {source_type}
Raw row: {raw_row}

Extract the following and return as JSON only, no explanation:
{{
    "host_name": "string or null",
    "podcast_name": "string or null",
    "channel_name": "string or null",
    "brand_name": "string or null",
    "website": "string or null — only if it is a standalone domain, not a social platform",
    "existing_email": "string or null",
    "social_links": {{
        "facebook": "string or null",
        "twitter": "string or null",
        "instagram": "string or null",
        "youtube": "string or null",
        "linkedin": "string or null"
    }}
}}

Rules:
- Any URL belonging to a social platform must go into social_links, never into website
- website is only a standalone domain like https://example.com
- If a field is not present or not inferrable, set it to null
- Do not invent or guess values
"""


def _detect_social_platform(url: str) -> str | None:
    """Check if a URL belongs to a known social platform."""
    if not url:
        return None
    for domain, platform in SOCIAL_PLATFORM_DOMAINS.items():
        if domain in url.lower():
            return platform
    return None


def _is_valid_email(value: str) -> bool:
    """Basic email format check."""
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, value.strip()))


def _extract_via_llm(raw_row: dict, source_type: SourceType) -> dict:
    """Use LiteLLM + OpenAI to classify raw row fields."""
    prompt_text = get_prompt(
        prompt_name="canonical_builder",
        fallback=FALLBACK_PROMPT,
    )

    filled_prompt = prompt_text.format(
        source_type=source_type.value,
        raw_row=str(raw_row),
    )

    response = litellm.completion(
        model="gpt-4o-mini",        # cheap, fast, more than enough for classification
        messages=[{"role": "user", "content": filled_prompt}],
        response_format={"type": "json_object"},   # enforces JSON output, no need to strip fences
        metadata={
            "langfuse_session_id": "canonical_builder",
        },
    )

    import json
    content = response.choices[0].message.content.strip()
    return json.loads(content)


def _build_from_llm_output(
    llm_output: dict,
    source_type: SourceType,
    raw_row: dict,
) -> CanonicalLead:
    """Construct CanonicalLead from LLM classified output."""

    # Sanitize website — reject if it's actually a social platform URL
    website = llm_output.get("website")
    if website and _detect_social_platform(website):
        website = None

    # Sanitize existing email
    existing_email = llm_output.get("existing_email")
    if existing_email and not _is_valid_email(existing_email):
        existing_email = None

    # Build social links — also catch any social URLs that slipped into website
    raw_social = llm_output.get("social_links", {})
    social_links = SocialLinks(
        facebook=raw_social.get("facebook"),
        twitter=raw_social.get("twitter"),
        instagram=raw_social.get("instagram"),
        youtube=raw_social.get("youtube"),
        linkedin=raw_social.get("linkedin"),
    )

    identity = Identity(
        host_name=llm_output.get("host_name"),
        podcast_name=llm_output.get("podcast_name"),
        channel_name=llm_output.get("channel_name"),
        brand_name=llm_output.get("brand_name"),
    )

    return CanonicalLead(
        identity=identity,
        website=website,
        existing_email=existing_email,
        social_links=social_links,
        source_type=source_type,
        raw=raw_row,
    )


def canonical_builder_node(
    raw_row: dict[str, Any],
    source_type: SourceType,
) -> EmailFinderState:
    """
    Entry point node.
    Takes a raw CSV row + source type.
    Returns a fully initialized EmailFinderState.
    """
    trace = langfuse.trace(name="canonical_builder")

    try:
        span = trace.span(name="llm_classification")
        llm_output = _extract_via_llm(raw_row, source_type)
        span.end()

        lead = _build_from_llm_output(llm_output, source_type, raw_row)

        trace.event(
            name="canonical_lead_built",
            metadata={"source_type": source_type.value},
        )

        return EmailFinderState(
            lead=lead,
            status=LeadStatus.PENDING,
            nodes_executed=["canonical_builder"],
        )

    except Exception as e:
        trace.event(name="canonical_builder_failed", metadata={"error": str(e)})
        raise