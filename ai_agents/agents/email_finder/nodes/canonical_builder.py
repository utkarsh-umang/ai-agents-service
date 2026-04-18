from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urlencode, parse_qsl
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


# Query params that add no value for discovery (ad/analytics tracking)
_TRACKING_PARAMS = {
    "fbclid", "gclid", "msclkid", "utm_source", "utm_medium",
    "utm_campaign", "utm_term", "utm_content", "ref", "igshid",
}


def _strip_tracking_params(url: str) -> str:
    """Remove ad/analytics tracking query params from a URL."""
    parsed = urlparse(url)
    clean_qs = urlencode(
        [(k, v) for k, v in parse_qsl(parsed.query) if k not in _TRACKING_PARAMS]
    )
    return parsed._replace(query=clean_qs).geturl()


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


def _extract_via_llm(raw_row: dict, source_type: SourceType, trace_id: str) -> dict:
    """Use LiteLLM + OpenAI to classify raw row fields."""
    filled_prompt = get_prompt(
        "canonical_builder",
        source_type=source_type.value,
        raw_row=str(raw_row),
    )

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": filled_prompt}],
        response_format={"type": "json_object"},
        metadata={
            "langfuse_trace_id": trace_id,
            "langfuse_session_id": "email_finder",
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

    # Collect discovery URLs (linktr.ee, beacons.ai, carrd.co, etc.)
    # Strip tracking params and reject any that are social platform URLs
    raw_discovery = llm_output.get("discovery_urls") or []
    discovery_urls = [
        _strip_tracking_params(url)
        for url in raw_discovery
        if url and not _detect_social_platform(url)
    ]

    # Build social links
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
        discovery_urls=discovery_urls,
        existing_email=existing_email,
        social_links=social_links,
        source_type=source_type,
        raw=raw_row,
    )


def canonical_builder_to_graph_dict(
    raw_row: dict[str, Any],
    source_type: SourceType,
) -> dict[str, Any]:
    """
    LangGraph entry node: returns partial state including trace_id and empty reducer lists.
    """
    trace = langfuse.trace(name="canonical_builder", session_id="email_finder")

    try:
        span = trace.span(name="llm_classification")
        llm_output = _extract_via_llm(raw_row, source_type, trace_id=trace.id)
        span.end()

        lead = _build_from_llm_output(llm_output, source_type, raw_row)

        trace.event(
            name="canonical_lead_built",
            metadata={
                "source_type": source_type.value,
                "has_website": bool(lead.website),
                "discovery_urls_count": len(lead.discovery_urls),
            },
        )

        return {
            "lead": lead.model_dump(mode="json"),
            "status": LeadStatus.PENDING.value,
            "email_candidates": [],
            "errors": [],
            "nodes_executed": ["canonical_builder"],
            "trace_id": trace.id,
            "website_scrape_candidates": [],
            "scrape_plan": [],
        }

    except Exception as e:
        trace.event(name="canonical_builder_failed", metadata={"error": str(e)})
        raise


def canonical_builder_node(
    raw_row: dict[str, Any],
    source_type: SourceType,
) -> EmailFinderState:
    """
    Entry point node.
    Takes a raw CSV row + source type.
    Returns a fully initialized EmailFinderState.
    """
    payload = canonical_builder_to_graph_dict(raw_row, source_type)
    lead = CanonicalLead.model_validate(payload["lead"])
    return EmailFinderState(
        lead=lead,
        status=LeadStatus.PENDING,
        nodes_executed=["canonical_builder"],
    )
