"""Single-URL crawl worker (LangGraph Send target)."""
from __future__ import annotations

import re
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from ai_agents.agents.email_finder.adapters import crawl_input_from_worker_state, candidates_to_dicts
from ai_agents.agents.email_finder.io.contract_models import CrawlPageInput, CrawlPageOutput
from ai_agents.agents.email_finder.state import EmailCandidate
from ai_agents.core.llm import langfuse

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.MULTILINE,
)

_GENERIC = frozenset(
    {"info", "contact", "hello", "support", "admin", "team", "mail", "enquiries", "enquiry", "noreply"}
)

# Form placeholders and template junk that appear verbatim in page markup
# ("your@email.com" in a newsletter input, name@example.com in docs). These
# are never a real contact — caught in production: a lead landed in the
# master table with email "your@email.com".
_PLACEHOLDER_DOMAINS = frozenset(
    {"example.com", "example.org", "example.net", "domain.com", "yourdomain.com",
     "yourcompany.com", "yoursite.com", "mysite.com", "website.com", "test.com",
     "sample.com", "company.com", "sentry.io", "sentry.wixpress.com"}
)
_PLACEHOLDER_LOCALS = frozenset(
    {"your", "youremail", "yourname", "name", "firstname", "lastname",
     "firstname.lastname", "john.doe", "jane.doe", "user", "username",
     "test", "example", "sample", "someone", "somebody", "email"}
)


def _is_placeholder_email(email: str) -> bool:
    local, _, domain = email.lower().partition("@")
    return domain in _PLACEHOLDER_DOMAINS or local in _PLACEHOLDER_LOCALS


def _extract_emails_from_text(text: str) -> list[str]:
    if not text:
        return []
    found = set()
    for m in _EMAIL_RE.findall(text):
        e = m.strip().rstrip(".,);]").lower()
        if "@" in e and "." in e.split("@")[-1] and not _is_placeholder_email(e):
            found.add(e)
    return list(found)


def _mailto_from_html(html: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r'mailto:([^"\'>\s?]+)', html or "", re.I):
        addr = m.group(1).split("?")[0].strip()
        if "@" in addr and not _is_placeholder_email(addr):
            out.append(addr)
    return out


def _confidence_for_email(email: str) -> float:
    local = email.split("@")[0].lower()
    if local in _GENERIC:
        return 0.55
    return 0.75


async def _crawl_async(inp: CrawlPageInput) -> CrawlPageOutput:
    trace = langfuse.trace(
        id=inp.trace_id,
        name="email_finder",
        session_id="email_finder",
    )
    span = trace.span(name="crawl_page", metadata={"crawl_url": inp.url})

    browser_conf = BrowserConfig(headless=True)
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=inp.page_timeout_ms,
    )
    try:
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=inp.url, config=run_conf)
    except Exception as e:
        span.end()
        trace.event(name="crawl_timeout", metadata={"url": inp.url, "error": str(e)})
        return CrawlPageOutput(
            candidates=[],
            errors=[f"crawl_page {inp.url}: {e}"],
            page_url=inp.url,
        )

    if not result.success:
        err = result.error_message or "unknown"
        span.end()
        trace.event(name="crawl_failed", metadata={"url": inp.url, "error": err})
        return CrawlPageOutput(candidates=[], errors=[f"{inp.url}: {err}"], page_url=inp.url)

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
        candidates.append(
            EmailCandidate(
                email=e,
                source=f"website_scraper — {inp.url}",
                confidence=_confidence_for_email(e),
                note="extracted from page content",
            )
        )

    span.end()
    trace.event(
        name="crawl_ok",
        metadata={"url": inp.url, "candidate_count": len(candidates)},
    )
    return CrawlPageOutput(candidates=candidates, errors=[], page_url=inp.url)


async def crawl_page_node_async(worker_state: dict) -> dict:
    """Partial state update for website_scrape_candidates reducer."""
    inp = crawl_input_from_worker_state(worker_state)
    out = await _crawl_async(inp)
    return {
        "website_scrape_candidates": candidates_to_dicts(out.candidates),
        "errors": out.errors,
    }
