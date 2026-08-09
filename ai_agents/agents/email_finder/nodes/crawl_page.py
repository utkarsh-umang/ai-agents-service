"""Single-URL crawl worker (LangGraph Send target)."""
from __future__ import annotations

import asyncio
import re
from crawl4ai import CacheMode, CrawlerRunConfig

# Absolute wall-clock cap per crawl attempt. crawl4ai's own `page_timeout` covers
# normal slow pages, but a hung browser/connection can blow past it — this asyncio
# backstop guarantees we abandon a site after ~3 min instead of stalling the run.
_CRAWL_HARD_TIMEOUT_S = 180

from ai_agents.agents.email_finder.nodes import browser_pool
from ai_agents.agents.email_finder.adapters import crawl_input_from_worker_state, candidates_to_dicts
from ai_agents.agents.email_finder.io.contract_models import CrawlPageInput, CrawlPageOutput
from ai_agents.agents.email_finder.nodes.email_utils import confidence_for_email, enrich_confidence, filter_emails
from ai_agents.agents.email_finder.state import EmailCandidate
from ai_agents.core.llm import langfuse

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.MULTILINE,
)

_FB_HREF_RE = re.compile(
    r'href=["\']([^"\']*facebook\.com/[^"\']+)["\']',
    re.IGNORECASE,
)

_FB_GENERIC_FRAGMENTS = frozenset(
    {
        "facebook.com/sharer",
        "facebook.com/share",
        "facebook.com/dialog",
        "facebook.com/plugins",
        "facebook.com/tr?",
    }
)


def _extract_emails_from_text(text: str) -> list[str]:
    if not text:
        return []
    found = set()
    for m in _EMAIL_RE.findall(text):
        e = m.strip().rstrip(".,);]")
        if "@" in e and "." in e.split("@")[-1]:
            found.add(e.lower())
    return filter_emails(list(found))


def _mailto_from_html(html: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r'mailto:([^"\'>\s?]+)', html or "", re.I):
        addr = m.group(1).split("?")[0].strip()
        if "@" in addr:
            out.append(addr)
    return filter_emails(out)


def _extract_fb_links(html: str) -> list[str]:
    if not html:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _FB_HREF_RE.finditer(html):
        url = m.group(1)
        url_lower = url.lower()
        if any(fragment in url_lower for fragment in _FB_GENERIC_FRAGMENTS):
            continue
        key = url_lower
        if key not in seen:
            seen.add(key)
            out.append(url)
    return out


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
    for attempt in range(1, 3):  # 2 attempts
        try:
            result = await asyncio.wait_for(
                _crawl_once(run_conf, inp.url),
                timeout=_CRAWL_HARD_TIMEOUT_S,
            )
            if result.success:
                break
            last_error = result.error_message or "unknown crawl failure"
            trace.event(name="crawl_failed", metadata={"url": inp.url, "attempt": attempt, "error": last_error})
        except asyncio.TimeoutError:
            # Hung site — abandon it (no retry) so the lead can move on.
            last_error = f"hard timeout after {_CRAWL_HARD_TIMEOUT_S}s"
            trace.event(name="crawl_timeout", metadata={"url": inp.url, "attempt": attempt})
            break
        except Exception as e:
            # Truncate internal crawl4ai stack paths — keep only the first sentence
            raw = str(e)
            last_error = raw.split("\n")[0][:200]
            trace.event(name="crawl_error", metadata={"url": inp.url, "attempt": attempt, "error": last_error})

    if result is None or not result.success:
        span.end()
        return CrawlPageOutput(
            candidates=[],
            errors=[f"crawl_page failed after retries: {last_error}"],
            page_url=inp.url,
        )

    html = result.html or ""
    md = ""
    if result.markdown:
        md = getattr(result.markdown, "raw_markdown", None) or str(result.markdown)
    blob = f"{html}\n{md}"

    emails = set(_extract_emails_from_text(blob))

    for m in _mailto_from_html(html):
        emails.add(m.lower())

    candidates: list[EmailCandidate] = []
    for e in sorted(emails):
        base_confidence = confidence_for_email(e)

        enriched_confidence, note = enrich_confidence(
            e,
            inp.url,
            blob,
            base_confidence,
        )

        candidates.append(
            EmailCandidate(
                email=e,
                source=f"website_scraper — {inp.url}",
                confidence=enriched_confidence,
                note=note,
            )
        )

    # FB link extraction — skip if structured data already has a FB link
    if inp.lead.social_links.facebook:
        fb_links: list[str] = []
    else:
        fb_links = _extract_fb_links(html)

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