from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urlencode, parse_qsl
import litellm
from ai_agents.agents.email_finder.nodes.cost_utils import llm_call_cost
from ai_agents.core.llm import langfuse, get_prompt
from ai_agents.agents.email_finder.state import (
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


# Public mailbox providers — their domain is NOT a personal/company website, so
# we never derive a website from an email hosted at one of these.
_FREE_EMAIL_DOMAINS = frozenset(
    {
        "gmail.com", "googlemail.com",
        "yahoo.com", "yahoo.co.uk", "yahoo.co.in", "ymail.com", "rocketmail.com",
        "hotmail.com", "hotmail.co.uk", "outlook.com", "live.com", "msn.com",
        "icloud.com", "me.com", "mac.com",
        "aol.com", "gmx.com", "gmx.net", "mail.com", "zoho.com",
        "proton.me", "protonmail.com", "pm.me",
        "yandex.com", "fastmail.com", "hey.com", "hushmail.com", "tutanota.com",
    }
)


def _website_from_email(email: str | None) -> str | None:
    """
    Derive a likely website from an email's domain when no website was given
    (jane@acme.com -> https://acme.com). Returns None for free/public mailbox
    providers (gmail, outlook, ...) whose domain is not the person's site, and
    for malformed input.
    """
    if not email or "@" not in email:
        return None
    domain = email.rsplit("@", 1)[-1].strip().lower().strip(".")
    if not domain or "." not in domain:
        return None
    if domain in _FREE_EMAIL_DOMAINS:
        return None
    return f"https://{domain}"


def _extract_via_llm(raw_row: dict, source_type: SourceType, trace_id: str) -> tuple[dict, float]:
    """Use LiteLLM + OpenAI to classify raw row fields. Returns (output, cost_usd)."""
    filled_prompt = get_prompt(
        "canonical_builder",
        source_type=source_type.value,
        raw_row=str(raw_row),
    )

    response = litellm.completion(
        model="gpt-4o-mini",
        # 60s cap — without it the OpenAI SDK default (600s × retries) can hang
        # an executor thread ~20min and jam the whole page (the Joseph llinas RCA).
        timeout=60,
        messages=[{"role": "user", "content": filled_prompt}],
        response_format={"type": "json_object"},
        metadata={
            "langfuse_trace_id": trace_id,
            "langfuse_session_id": "email_finder",
        },
    )

    import json
    content = response.choices[0].message.content.strip()
    return json.loads(content), llm_call_cost(response)


def _structured_llm_output(raw_row: dict) -> dict:
    """Map an already-canonical row (source='lms') into the shape
    `_build_from_llm_output` expects, so the classifier LLM can be skipped.

    Safe ONLY because an LMS queue row is already canonical — website is in the
    `website` column, each social in its own column — so there is nothing to
    *classify*, only to reshape. All downstream sanitization (social-platform
    rejection, tracking-param strip, email-derived website) still runs via
    `_build_from_llm_output`, so the output is identical to the LLM path minus
    the round-trip. Do not use for free-shape rows where columns are ambiguous.
    """
    def pick(*keys: str) -> str | None:
        for k in keys:
            v = raw_row.get(k)
            if v and str(v).strip():
                return str(v).strip()
        return None

    return {
        "website": pick("website", "Website"),
        "existing_email": pick("email", "Email", "existing_email"),
        "host_name": pick("host_name", "Host Name"),
        "podcast_name": pick("podcast_name", "Podcast Name", "company_name"),
        "channel_name": pick("channel_name", "Channel Name"),
        "brand_name": pick("brand_name"),
        "discovery_urls": [],
        "social_links": {
            "facebook": pick("facebook_url", "facebook"),
            "twitter": pick("twitter_url", "twitter"),
            "instagram": pick("instagram_url", "instagram"),
            "youtube": pick("youtube_url", "channel_url", "youtube"),
            "linkedin": pick("linkedin_url", "linkedin"),
        },
    }


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
    if existing_email:
        # Handle comma-separated multiple emails — take first valid one
        for candidate in re.split(r"[,;]\s*", existing_email):
            if _is_valid_email(candidate.strip()):
                existing_email = candidate.strip()
                break
        else:
            existing_email = None

    # If no website was given but we have a usable (non-free-provider) email,
    # derive the website from its domain so the free crawl path has a domain to
    # work with. Tagged in raw for provenance; routing treats an email-derived
    # website as a fallback — validation of the existing email still runs first
    # (see route_after_canonical), so a good existing email is never bypassed.
    if not website and existing_email:
        derived = _website_from_email(existing_email)
        if derived:
            website = derived
            raw_row = {**raw_row, "_website_source": "email_domain"}

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
    source: str | None = None,
) -> dict[str, Any]:
    """
    LangGraph entry node: returns partial state including trace_id and empty reducer lists.

    When `source == "lms"` the row is already canonical (one field per column),
    so we skip the classifier LLM entirely and reshape deterministically — a
    per-lead latency + token-cost win with no recall change (the LMS is where
    every lead comes from in production).
    """
    trace = langfuse.trace(name="email_finder", session_id="email_finder")

    try:
        if source == "lms":
            span = trace.span(name="structured_classification")
            llm_output = _structured_llm_output(raw_row)
            llm_cost = 0.0
            span.end()
        else:
            span = trace.span(name="llm_classification")
            llm_output, llm_cost = _extract_via_llm(raw_row, source_type, trace_id=trace.id)
            span.end()

        lead = _build_from_llm_output(llm_output, source_type, raw_row)

        trace.event(
            name="canonical_lead_built",
            metadata={
                "source_type": source_type.value,
                "classifier": "structured" if source == "lms" else "llm",
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
            "cost_usd": llm_cost,
        }

    except Exception as e:
        trace.event(name="canonical_builder_failed", metadata={"error": str(e)})
        raise


