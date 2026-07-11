"""Map LangGraph dict state to/from Pydantic node contracts."""
from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from ai_agents.agents.email_finder.io.contract_models import (
    CrawlPageInput,
    DiscoveryInput,
    FBCrawlerInput,
    PerplexityInput,
    ResolveInput,
    ValidateEmailInput,
    WebsiteGuessInput,
    YouTubeEnrichInput,
)
from ai_agents.agents.email_finder.state import CanonicalLead, EmailCandidate, SourceType


def parse_canonical_lead(lead_dict: dict[str, Any]) -> CanonicalLead:
    return CanonicalLead.model_validate(lead_dict)


def lead_to_dict(lead: CanonicalLead) -> dict[str, Any]:
    return lead.model_dump(mode="json")


def parse_email_candidates(items: list[dict[str, Any]] | None) -> list[EmailCandidate]:
    if not items:
        return []
    return TypeAdapter(list[EmailCandidate]).validate_python(items)


def candidates_to_dicts(candidates: list[EmailCandidate]) -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in candidates]


def discovery_input_from_state(state: dict[str, Any]) -> DiscoveryInput:
    return DiscoveryInput(
        lead=parse_canonical_lead(state["lead"]),
        max_urls=state.get("max_scrape_urls", 5),
        trace_id=state.get("trace_id", ""),
    )


def crawl_input_from_worker_state(worker: dict[str, Any]) -> CrawlPageInput:
    return CrawlPageInput(
        url=worker["url"],
        lead=parse_canonical_lead(worker["lead"]),
        trace_id=worker.get("trace_id", ""),
        page_timeout_ms=worker.get("page_timeout_ms", 60000),
    )


def resolve_input_from_state(state: dict[str, Any]) -> ResolveInput:
    wc = state.get("website_scrape_candidates") or []
    return ResolveInput(
        lead=parse_canonical_lead(state["lead"]),
        website_candidates=parse_email_candidates(wc),
        trace_id=state.get("trace_id", ""),
    )


def perplexity_input_from_state(state: dict[str, Any]) -> PerplexityInput:
    prior = state.get("email_candidates") or []
    return PerplexityInput(
        lead=parse_canonical_lead(state["lead"]),
        trace_id=state.get("trace_id", ""),
        prior_email_candidates=parse_email_candidates(prior),
        source=state.get("source"),
    )


def youtube_enrich_input_from_state(state: dict[str, Any]) -> YouTubeEnrichInput:
    return YouTubeEnrichInput(
        lead=parse_canonical_lead(state["lead"]),
        trace_id=state.get("trace_id", ""),
    )


def website_guess_input_from_state(state: dict[str, Any]) -> WebsiteGuessInput:
    return WebsiteGuessInput(
        lead=parse_canonical_lead(state["lead"]),
        trace_id=state.get("trace_id", ""),
        source=state.get("source"),
    )


def validate_email_input_from_state(state: dict[str, Any]) -> ValidateEmailInput:
    return ValidateEmailInput(
        lead=parse_canonical_lead(state["lead"]),
        trace_id=state.get("trace_id", ""),
    )


def fb_link_from_state(state: dict[str, Any]) -> str | None:
    """Return FB link from structured data, or first scraped FB link."""
    lead = state.get("lead") or {}
    social = lead.get("social_links") or {}
    fb_lead = social.get("facebook") or ""
    if fb_lead.strip():
        return fb_lead.strip()
    scraped = state.get("scraped_fb_links") or []
    return scraped[0] if scraped else None


def fb_crawler_input_from_state(state: dict[str, Any]) -> FBCrawlerInput:
    """Build FBCrawlerInput using best available FB link from state."""
    lead = parse_canonical_lead(state["lead"])
    fb_link = fb_link_from_state(state) or ""
    return FBCrawlerInput(
        lead=lead,
        fb_link=fb_link,
        trace_id=state.get("trace_id", ""),
    )


def merge_candidate_dicts(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by lowercased email."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for d in a + b:
        e = (d.get("email") or "").lower().strip()
        if e and e not in seen:
            seen.add(e)
            out.append(d)
    return out


def source_type_from_state(state: dict[str, Any]) -> SourceType:
    raw = state.get("source_type")
    if isinstance(raw, SourceType):
        return raw
    return SourceType(raw)