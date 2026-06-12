from __future__ import annotations

import asyncio
import nest_asyncio
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from ai_agents.agents.email_finder.adapters import source_type_from_state
from ai_agents.agents.email_finder.graph_state import EmailFinderGraphState
from ai_agents.agents.email_finder.nodes.canonical_builder import (
    canonical_builder_to_graph_dict,
)
from ai_agents.agents.email_finder.nodes.crawl_page import crawl_page_node_async
from ai_agents.agents.email_finder.nodes.fb_crawler import fb_crawler_node_async
from ai_agents.agents.email_finder.nodes.perplexity_discovery import (
    perplexity_discovery_node_async,
)
from ai_agents.agents.email_finder.nodes.resolve_best_email import (
    resolve_best_email_node_async,
)
from ai_agents.agents.email_finder.nodes.url_discovery import discover_urls_node_async
from ai_agents.agents.email_finder.nodes.validate_existing_email import (
    validate_existing_email_node_async,
)
from ai_agents.agents.email_finder.nodes.youtube_about_enricher import (
    youtube_about_enricher_node_async,
)
from ai_agents.agents.email_finder.state import LeadStatus, SourceType


# ── Node wrappers ────────────────────────────────────────────────────────────


def run_canonical_builder(state: EmailFinderGraphState) -> dict[str, Any]:
    st = source_type_from_state(state)
    return canonical_builder_to_graph_dict(state["raw_row"], st)


def _youtube_enabled(state: EmailFinderGraphState) -> bool:
    """
    True when this run is flagged youtube_list AND the lead actually has a
    YouTube URL — the gate for the ScrapingBee youtube_about_enricher node.
    """
    if not state.get("youtube_list"):
        return False
    lead = state.get("lead") or {}
    social = lead.get("social_links") or {}
    if (social.get("youtube") or "").strip():
        return True
    raw = lead.get("raw") or {}
    for key in ("Channel URL", "channel_url", "YouTube", "youtube"):
        val = (raw.get(key) or "").strip()
        if "youtube.com" in val.lower() or "youtu.be" in val.lower():
            return True
    return False


def route_after_canonical(state: EmailFinderGraphState) -> str:
    lead = state.get("lead") or {}
    existing_email = lead.get("existing_email")
    website = lead.get("website")

    # Only validate-and-exit if we have an email AND no website to improve on
    if existing_email and str(existing_email).strip() and not (website and str(website).strip()):
        return "validate_existing_email"
    elif website and str(website).strip():
        # Website crawl takes priority — user-chosen ordering.
        return "discover_urls"
    elif _youtube_enabled(state):
        # No website yet: enrich from the YouTube About page to discover one
        # (and socials), then the enricher routes into the normal flow.
        return "youtube_about_enricher"
    else:
        return "perplexity_discovery"


def route_after_validate(state: EmailFinderGraphState) -> str:
    if state.get("status") == LeadStatus.EMAIL_FOUND.value:
        return END
    w = (state.get("lead") or {}).get("website")
    if w and str(w).strip():
        return "discover_urls"
    if _youtube_enabled(state):
        return "youtube_about_enricher"
    return "perplexity_discovery"


def route_after_discover(state: EmailFinderGraphState) -> str | list[Send]:
    plan = state.get("scrape_plan") or []
    if not plan:
        return "resolve_best_email"
    tid = state.get("trace_id") or ""
    lead = state["lead"]
    return [
        Send(
            "crawl_page",
            {
                "url": u,
                "lead": lead,
                "trace_id": tid,
                "page_timeout_ms": 60000,
            },
        )
        for u in plan
    ]


def route_after_resolve(state: EmailFinderGraphState) -> str:
    if state.get("status") == LeadStatus.EMAIL_FOUND.value:
        return END

    # Only invoke the slow FB crawler if scraping produced zero candidates
    candidates = state.get("email_candidates") or []
    if not candidates:
        lead = state.get("lead") or {}
        social = lead.get("social_links") or {}
        fb_from_lead = social.get("facebook") or ""
        fb_from_scrape = state.get("scraped_fb_links") or []
        has_fb = bool(fb_from_lead.strip()) or bool(fb_from_scrape)
        if has_fb:
            return "fb_crawler"

    return "perplexity_discovery"


def route_after_fb_crawler(state: EmailFinderGraphState) -> str:
    if state.get("status") == LeadStatus.EMAIL_FOUND.value:
        return END
    return "perplexity_discovery"


def route_after_youtube_enrich(state: EmailFinderGraphState) -> str:
    # A plaintext email was on the About page → done.
    if state.get("status") == LeadStatus.EMAIL_FOUND.value:
        return END
    # Enrichment may have discovered the creator's website → crawl it.
    w = (state.get("lead") or {}).get("website")
    if w and str(w).strip():
        return "discover_urls"
    # Otherwise hand off to Perplexity with the enriched socials/discovery_urls.
    return "perplexity_discovery"


# ── Graph definition ─────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    graph = StateGraph(EmailFinderGraphState)

    graph.add_node("canonical_builder", run_canonical_builder)
    graph.add_node("validate_existing_email", validate_existing_email_node_async)
    graph.add_node("discover_urls", discover_urls_node_async)
    graph.add_node("crawl_page", crawl_page_node_async)
    graph.add_node("resolve_best_email", resolve_best_email_node_async)
    graph.add_node("perplexity_discovery", perplexity_discovery_node_async)
    graph.add_node("fb_crawler", fb_crawler_node_async)
    graph.add_node("youtube_about_enricher", youtube_about_enricher_node_async)

    graph.set_entry_point("canonical_builder")

    graph.add_conditional_edges(
        "canonical_builder",
        route_after_canonical,
        {
            "validate_existing_email": "validate_existing_email",
            "discover_urls": "discover_urls",
            "youtube_about_enricher": "youtube_about_enricher",
            "perplexity_discovery": "perplexity_discovery",
        },
    )

    graph.add_conditional_edges(
        "validate_existing_email",
        route_after_validate,
        {
            END: END,
            "discover_urls": "discover_urls",
            "youtube_about_enricher": "youtube_about_enricher",
            "perplexity_discovery": "perplexity_discovery",
        },
    )

    graph.add_conditional_edges(
        "discover_urls",
        route_after_discover,
        {
            "crawl_page": "crawl_page",
            "resolve_best_email": "resolve_best_email",
        },
    )

    graph.add_edge("crawl_page", "resolve_best_email")

    graph.add_conditional_edges(
        "resolve_best_email",
        route_after_resolve,
        {
            END: END,
            "fb_crawler": "fb_crawler",
            "perplexity_discovery": "perplexity_discovery",
        },
    )

    graph.add_conditional_edges(
        "fb_crawler",
        route_after_fb_crawler,
        {
            END: END,
            "perplexity_discovery": "perplexity_discovery",
        },
    )

    graph.add_conditional_edges(
        "youtube_about_enricher",
        route_after_youtube_enrich,
        {
            END: END,
            "discover_urls": "discover_urls",
            "perplexity_discovery": "perplexity_discovery",
        },
    )

    graph.add_edge("perplexity_discovery", END)

    return graph.compile()


# ── Single lead runner ────────────────────────────────────────────────────────


def run_single(
    raw_row: dict[str, Any],
    source_type: SourceType,
    youtube_list: bool = False,
) -> dict[str, Any]:
    """
    Run the graph for a single lead.
    Returns the final state as a dict.

    youtube_list: enable the ScrapingBee youtube_about_enricher node (for lists
        known to be YouTube channels). Only fires when the lead has a YouTube URL.
    """
    graph = build_graph()

    async def _run() -> dict[str, Any]:
        return await graph.ainvoke(
            {
                "raw_row": raw_row,
                "source_type": source_type.value
                if isinstance(source_type, SourceType)
                else source_type,
                "youtube_list": youtube_list,
            }
        )

    try:
        return asyncio.run(_run())
    except RuntimeError:
        # Nested loop (e.g. Jupyter)
        nest_asyncio.apply()
        return asyncio.run(_run())


# ── Batch runner ──────────────────────────────────────────────────────────────


async def run_single_async(
    graph: Any,
    raw_row: dict[str, Any],
    source_type: SourceType,
    semaphore: asyncio.Semaphore,
    index: int = 0,
    total: int = 0,
    youtube_list: bool = False,
) -> dict[str, Any]:
    """Run a single lead through the graph with semaphore rate limiting."""
    label = raw_row.get("Guest Name") or raw_row.get("Host Name") or raw_row.get("Podcast Name") or raw_row.get("Channel Name") or f"row-{index}"
    async with semaphore:
        print(f"  [{index}/{total}] Starting: {label}")
        st_val = source_type.value if isinstance(source_type, SourceType) else source_type
        result = await graph.ainvoke(
            {"raw_row": raw_row, "source_type": st_val, "youtube_list": youtube_list}
        )
        status = result.get("status", "?")
        nodes = result.get("nodes_executed") or []
        email = (result.get("best_email") or {}).get("email", "")
        summary = f"status={status}"
        if email:
            summary += f" email={email}"
        summary += f" path={' → '.join(nodes)}"
        print(f"  [{index}/{total}] Done: {label} — {summary}")
        return result


async def run_batch_async(
    rows: list[dict[str, Any]],
    source_type: SourceType,
    concurrency: int = 3,
    youtube_list: bool = False,
) -> list[dict[str, Any]]:
    """
    Run multiple leads concurrently.
    Lower default concurrency due to Playwright + Perplexity load.
    """
    graph = build_graph()
    semaphore = asyncio.Semaphore(concurrency)
    total = len(rows)

    tasks = [
        run_single_async(graph, row, source_type, semaphore, i + 1, total, youtube_list)
        for i, row in enumerate(rows)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  [{i+1}/{total}] FAILED: {str(result)[:120]}")
            output.append({
                "status": LeadStatus.FAILED.value,
                "errors": [str(result)],
                "raw_row": rows[i],
            })
        else:
            output.append(result)

    return output


def run_batch(
    rows: list[dict[str, Any]],
    source_type: SourceType,
    concurrency: int = 3,
    youtube_list: bool = False,
) -> list[dict[str, Any]]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            nest_asyncio.apply()
        return asyncio.run(run_batch_async(rows, source_type, concurrency, youtube_list))
    except RuntimeError:
        return asyncio.get_event_loop().run_until_complete(
            run_batch_async(rows, source_type, concurrency, youtube_list)
        )