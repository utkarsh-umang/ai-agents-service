"""
resolve_bio_links node — Locators subsystem.

Runs the free link-in-bio / shortener hop-resolver over a lead's `discovery_urls`
when we have no real website yet. Emits, back onto the graph:
  - a resolved website on the lead (tagged `_website_source="bio_link"`), so the
    existing discover_urls → crawl_page path can scrape it, and/or
  - any emails printed on the aggregator page itself, pushed into the
    website_scrape_candidates reducer so resolve_best_email can score them.

It runs BEFORE any paid node, so a lead that would otherwise fall through to
Perplexity gets a free resolution attempt first.
"""
from __future__ import annotations

import httpx

from ai_agents.agents.email_finder.adapters import parse_canonical_lead
from ai_agents.agents.email_finder.nodes.email_utils import confidence_for_email
from ai_agents.agents.email_finder.nodes.link_resolver import expand_candidates
from ai_agents.agents.email_finder.state import EmailCandidate
from ai_agents.core.llm import langfuse

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EmailFinderBot/1.0)"}
_TIMEOUT = 15.0


async def resolve_bio_links_node_async(state: dict) -> dict:
    """LangGraph async node."""
    trace_id = state.get("trace_id", "")
    trace = langfuse.trace(id=trace_id, name="email_finder", session_id="email_finder")
    span = trace.span(name="resolve_bio_links")

    lead = parse_canonical_lead(state["lead"])
    name = lead.identity.best_name() or ""
    seed = list(lead.discovery_urls or [])

    targets: list[str] = []
    agg_emails: list[str] = []
    errors: list[str] = []

    if seed:
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT, follow_redirects=True, headers=_HEADERS
            ) as client:
                targets, agg_emails = await expand_candidates(seed, client, name)
        except Exception as e:  # graceful — never break the run on a bad bio page
            errors.append(f"resolve_bio_links: {str(e)[:120]}")

    update: dict = {
        "status": state.get("status", "pending"),
        "errors": errors,
        "nodes_executed": ["resolve_bio_links"],
    }

    if targets:
        lead.website = targets[0]
        lead.raw = {**(lead.raw or {}), "_website_source": "bio_link"}
        update["lead"] = lead.model_dump(mode="json")

    if agg_emails:
        candidates = [
            EmailCandidate(
                email=e,
                source="bio_link aggregator",
                confidence=confidence_for_email(e),
                note="found on link-in-bio page",
            ).model_dump(mode="json")
            for e in agg_emails
        ]
        # website_scrape_candidates is a reducer (operator.add) — appends.
        update["website_scrape_candidates"] = candidates

    span.end()
    trace.event(
        name="resolve_bio_links_complete",
        metadata={
            "seed": len(seed),
            "targets": len(targets),
            "agg_emails": len(agg_emails),
        },
    )
    return update
