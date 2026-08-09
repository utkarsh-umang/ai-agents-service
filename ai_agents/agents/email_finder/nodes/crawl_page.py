"""Single-URL crawl worker (LangGraph Send target)."""
from __future__ import annotations

import asyncio
from crawl4ai import CacheMode, CrawlerRunConfig

# Absolute wall-clock cap per crawl. crawl4ai's own `page_timeout` (30s) covers
# normal slow pages; this asyncio backstop only fires if the browser/connection
# hangs past that. Kept at ~3x page_timeout so a genuinely stuck navigation is
# abandoned well inside the lead's PER_LEAD_BUDGET_S instead of eating it.
_CRAWL_HARD_TIMEOUT_S = 90

from ai_agents.agents.email_finder.nodes import browser_pool
from ai_agents.agents.email_finder.nodes.page_extract import build_page_candidates
from ai_agents.agents.email_finder.adapters import crawl_input_from_worker_state, candidates_to_dicts
from ai_agents.agents.email_finder.io.contract_models import CrawlPageInput, CrawlPageOutput
from ai_agents.core.llm import langfuse


async def _crawl_once(run_conf, url: str):
    # Runs on the process-wide shared browser (browser_pool) — no per-URL
    # Chromium launch, and nothing to orphan when a crawl is cancelled at the
    # lead's budget deadline.
    return await browser_pool.crawl(url, run_conf)


async def _crawl_async(inp: CrawlPageInput) -> CrawlPageOutput:
    trace = langfuse.trace(
        id=inp.trace_id,
        name="email_finder",
        session_id="email_finder",
    )
    span = trace.span(name="crawl_page", metadata={"crawl_url": inp.url})

    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=inp.page_timeout_ms,
    )

    result = None
    last_error: str = ""
    # Single attempt. A failed crawl here is almost always a dead/blocked domain
    # that an identical retry can't fix (crawl4ai 0.8.6 exposes no HTTP status to
    # tell transient from deterministic), and the lead's fan-out already crawls
    # several pages on the domain — so a transient blip on any one page is
    # covered without retrying it. Retrying doubled the wasted time on dead sites
    # and pushed leads past PER_LEAD_BUDGET_S into false timeout-fails.
    try:
        result = await asyncio.wait_for(
            _crawl_once(run_conf, inp.url),
            timeout=_CRAWL_HARD_TIMEOUT_S,
        )
        if not result.success:
            last_error = result.error_message or "unknown crawl failure"
            trace.event(name="crawl_failed", metadata={"url": inp.url, "error": last_error})
    except asyncio.TimeoutError:
        # Hung site — abandon it so the lead can move on.
        last_error = f"hard timeout after {_CRAWL_HARD_TIMEOUT_S}s"
        trace.event(name="crawl_timeout", metadata={"url": inp.url})
    except Exception as e:
        # Truncate internal crawl4ai stack paths — keep only the first sentence
        raw = str(e)
        last_error = raw.split("\n")[0][:200]
        trace.event(name="crawl_error", metadata={"url": inp.url, "error": last_error})

    if result is None or not result.success:
        span.end()
        return CrawlPageOutput(
            candidates=[],
            errors=[f"crawl_page failed: {last_error}"],
            page_url=inp.url,
        )

    html = result.html or ""
    md = ""
    if result.markdown:
        md = getattr(result.markdown, "raw_markdown", None) or str(result.markdown)

    candidates, fb_links = build_page_candidates(
        html, md, inp.url, bool(inp.lead.social_links.facebook)
    )

    span.end()
    trace.event(
        name="crawl_ok",
        metadata={"url": inp.url, "candidate_count": len(candidates)},
    )
    return CrawlPageOutput(
        candidates=candidates,
        errors=[],
        page_url=inp.url,
        fb_links=fb_links,
    )


async def crawl_page_node_async(worker_state: dict) -> dict:
    """Partial state update for website_scrape_candidates reducer."""
    inp = crawl_input_from_worker_state(worker_state)
    out = await _crawl_async(inp)
    return {
        "website_scrape_candidates": candidates_to_dicts(out.candidates),
        "errors": out.errors,
        "scraped_fb_links": out.fb_links,
    }