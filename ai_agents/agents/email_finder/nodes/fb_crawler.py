"""FB page crawler node — Apify igview-owner~facebook-page-details-scraper."""
from __future__ import annotations

import asyncio
import os

import httpx

from ai_agents.agents.email_finder.io.contract_models import FBCrawlerInput, FBCrawlerOutput
from ai_agents.agents.email_finder.state import EmailCandidate, LeadStatus
from ai_agents.core.llm import langfuse

_APIFY_ACTOR = "igview-owner~facebook-page-details-scraper"
_APIFY_URL = f"https://api.apify.com/v2/actors/{_APIFY_ACTOR}/run-sync-get-dataset-items"

_GENERIC = frozenset(
    {"info", "contact", "hello", "support", "admin", "team", "mail", "enquiries", "enquiry", "noreply"}
)


def _confidence_for_email(email: str) -> float:
    local = email.split("@")[0].lower()
    if local in _GENERIC:
        return 0.55
    return 0.75


def _enrich_confidence(
    email: str,
    url: str,
    blob: str,
    base_confidence: float,
) -> tuple[float, str]:
    local_part = email.split("@")[0].lower().rstrip("0123456789")

    if len(local_part) < 4:
        return base_confidence, "extracted from page content"

    confidence = base_confidence
    notes: list[str] = []

    if local_part in url.lower():
        confidence += 0.15
        notes.append(f"local part '{local_part}' matched in URL")

    if local_part in blob.lower():
        confidence += 0.10
        notes.append(f"local part '{local_part}' matched in page text")

    confidence = min(confidence, 0.95)

    note = "; ".join(notes) if notes else "extracted from page content"

    return confidence, note


def _fb_crawl(inp: FBCrawlerInput) -> FBCrawlerOutput:
    trace = langfuse.trace(
        id=inp.trace_id,
        name="email_finder",
        session_id="email_finder",
    )
    span = trace.span(name="fb_crawler", metadata={"fb_link": inp.fb_link})
    print(f"[fb_crawler] fb_link received: {inp.fb_link!r}")  # ← add this


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

    base_confidence = _confidence_for_email(email)
    enriched_confidence, note = _enrich_confidence(
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

    span.end()
    trace.event(
        name="fb_crawler_ok",
        metadata={"fb_link": inp.fb_link, "email": email},
    )
    return FBCrawlerOutput(
        best_email=best_email,
        status=LeadStatus.EMAIL_FOUND,
    )


async def fb_crawler_node_async(state: dict) -> dict:
    from ai_agents.agents.email_finder.adapters import fb_crawler_input_from_state

    inp = fb_crawler_input_from_state(state)
    out = await asyncio.to_thread(_fb_crawl, inp)
    return {
        "best_email": out.best_email.model_dump(mode="json") if out.best_email else None,
        "status": out.status.value,
        "errors": out.errors,
        "nodes_executed": out.nodes_executed_delta,
    }