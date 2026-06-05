"""FB page crawler node — Apify igview-owner~facebook-page-details-scraper."""
from __future__ import annotations

import asyncio
import os

import httpx

from ai_agents.agents.email_finder.io.contract_models import FBCrawlerInput, FBCrawlerOutput
from ai_agents.agents.email_finder.nodes.email_utils import confidence_for_email, enrich_confidence, is_plus_addressed
from ai_agents.agents.email_finder.state import EmailCandidate, LeadStatus
from ai_agents.core.llm import langfuse

_APIFY_ACTOR = "igview-owner~facebook-page-details-scraper"
_APIFY_URL = f"https://api.apify.com/v2/actors/{_APIFY_ACTOR}/run-sync-get-dataset-items"


def _fb_crawl(inp: FBCrawlerInput) -> FBCrawlerOutput:
    trace = langfuse.trace(
        id=inp.trace_id,
        name="email_finder",
        session_id="email_finder",
    )
    span = trace.span(name="fb_crawler", metadata={"fb_link": inp.fb_link})

    api_token = os.environ.get("APIFY_API_TOKEN")
    if not api_token:
        span.end()
        return FBCrawlerOutput(errors=["APIFY_API_TOKEN not set"])

    if not inp.fb_link:
        span.end()
        return FBCrawlerOutput(errors=["no_fb_link_available"])

    payload = {
        "pageUrls": [inp.fb_link],
        "showVerifiedBadge": True,
    }
    headers = {
        "Content-Type": "application/json",
    }
    params = {
        "token": api_token,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                _APIFY_URL,
                json=payload,
                headers=headers,
                params=params,
            )
    except Exception as exc:
        span.end()
        trace.event(name="fb_crawler_error", metadata={"error": str(exc)})
        return FBCrawlerOutput(errors=[f"fb_crawler request failed: {exc}"])

    if response.status_code >= 400:
        span.end()
        trace.event(
            name="fb_crawler_http_error",
            metadata={"status_code": response.status_code},
        )
        return FBCrawlerOutput(
            errors=[f"fb_crawler HTTP {response.status_code}"]
        )

    try:
        data = response.json()
    except Exception as exc:
        span.end()
        return FBCrawlerOutput(errors=[f"fb_crawler JSON parse error: {exc}"])

    if not isinstance(data, list) or len(data) == 0:
        span.end()
        trace.event(name="fb_crawler_empty", metadata={"fb_link": inp.fb_link})
        return FBCrawlerOutput()

    page = data[0]
    email = (page.get("email") or "").strip()
    if not email:
        span.end()
        trace.event(name="fb_crawler_no_email", metadata={"fb_link": inp.fb_link})
        return FBCrawlerOutput()

    # Apify returns description/bio — use them for richer confidence scoring
    page_blob = " ".join(filter(None, [
        page.get("description") or "",
        page.get("bio") or "",
        page.get("title") or "",
    ]))

    if is_plus_addressed(email):
        span.end()
        trace.event(name="fb_crawler_no_email", metadata={"fb_link": inp.fb_link, "reason": "plus-addressed email skipped"})
        return FBCrawlerOutput()

    base_confidence = confidence_for_email(email)
    enriched_confidence, note = enrich_confidence(
        email,
        inp.fb_link,
        page_blob,
        base_confidence,
    )

    best_email = EmailCandidate(
        email=email,
        source="fb_crawler — Apify",
        confidence=enriched_confidence,
        note=note,
    )

    status = LeadStatus.EMAIL_FOUND if enriched_confidence > 0.75 else LeadStatus.EMAIL_NOT_FOUND

    span.end()
    trace.event(
        name="fb_crawler_ok",
        metadata={"fb_link": inp.fb_link, "email": email, "confidence": enriched_confidence},
    )
    return FBCrawlerOutput(
        best_email=best_email,
        status=status,
    )


async def fb_crawler_node_async(state: dict) -> dict:
    from ai_agents.agents.email_finder.adapters import fb_crawler_input_from_state, parse_canonical_lead

    # Collect all available FB links — structured first, then scraped
    lead_dict = state.get("lead") or {}
    social = lead_dict.get("social_links") or {}
    structured_fb = (social.get("facebook") or "").strip()
    scraped_fb: list[str] = state.get("scraped_fb_links") or []

    all_fb_links: list[str] = []
    if structured_fb:
        all_fb_links.append(structured_fb)
    for link in scraped_fb:
        if link not in all_fb_links:
            all_fb_links.append(link)

    if not all_fb_links:
        inp = fb_crawler_input_from_state(state)
        out = await asyncio.to_thread(_fb_crawl, inp)
    else:
        lead = parse_canonical_lead(state["lead"])
        trace_id = state.get("trace_id", "")
        out = FBCrawlerOutput(errors=["no_fb_link_available"])
        for fb_link in all_fb_links:
            candidate_inp = FBCrawlerInput(lead=lead, fb_link=fb_link, trace_id=trace_id)
            out = await asyncio.to_thread(_fb_crawl, candidate_inp)
            if out.best_email:
                break

    return {
        "best_email": out.best_email.model_dump(mode="json") if out.best_email else None,
        "status": out.status.value,
        "errors": out.errors,
        "nodes_executed": out.nodes_executed_delta,
    }