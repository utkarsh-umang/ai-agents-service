"""Parallel sitemap + homepage link discovery for scrape_plan."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from ai_agents.agents.email_finder.io.contract_models import DiscoveryInput, DiscoveryMeta, DiscoveryOutput
from ai_agents.core.llm import langfuse

_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

_KEYWORDS = (
    "contact",
    "about",
    "team",
    "company",
    "imprint",
    "kontakt",
    "support",
    "get-in-touch",
    "reach",
    "hello",
    "press",
    "media",
    "advertise",
    "sponsor",
    "partner",
)

def _normalize_origin(website: str) -> str | None:
    if not website or not website.strip():
        return None
    u = website.strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    p = urlparse(u)
    if not p.netloc:
        return None
    scheme = p.scheme or "https"
    netloc = p.netloc.lower()
    path = p.path or "/"
    return urlunparse((scheme, netloc, path if path.endswith("/") else path + "/", "", "", ""))


def _same_origin(origin: str, candidate: str) -> bool:
    try:
        o, c = urlparse(origin), urlparse(candidate)
        return o.netloc.lower() == c.netloc.lower()
    except Exception:
        return False


def _score_url(url: str) -> float:
    low = url.lower()
    score = 0.0
    for kw in _KEYWORDS:
        if kw in low:
            score += 10.0
    if urlparse(url).path in ("/", ""):
        score += 2.0
    return score


def _fetch_text(client: httpx.Client, url: str, timeout: float = 15.0) -> str | None:
    try:
        r = client.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "EmailFinderBot/1.0"})
        if r.status_code >= 400:
            return None
        return r.text
    except Exception:
        return None


def _robots_sitemap_urls(client: httpx.Client, base_origin: str) -> list[str]:
    parsed = urlparse(base_origin)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    text = _fetch_text(client, robots_url, timeout=8.0)
    if not text:
        return []
    urls: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.lower().startswith("sitemap:"):
            continue
        part = line.split(":", 1)[1].strip()
        if part:
            urls.append(part)
    return urls


def _parse_sitemap_locs(xml_bytes: str) -> tuple[list[str], list[str]]:
    """Return (locs from urlset, child sitemap URLs from sitemap index)."""
    locs: list[str] = []
    child_sitemaps: list[str] = []
    try:
        root = ET.fromstring(xml_bytes)
        tag = root.tag.lower()
        if tag.endswith("sitemapindex"):
            for sm in root.findall("sm:sitemap", _NS) + root.findall("{*}sitemap"):
                loc = sm.find("sm:loc", _NS)
                if loc is None:
                    loc = sm.find("{*}loc")
                if loc is not None and loc.text:
                    child_sitemaps.append(loc.text.strip())
        elif tag.endswith("urlset"):
            for url_el in root.findall("sm:url", _NS) + root.findall("{*}url"):
                loc = url_el.find("sm:loc", _NS)
                if loc is None:
                    loc = url_el.find("{*}loc")
                if loc is not None and loc.text:
                    locs.append(loc.text.strip())
    except ET.ParseError:
        pass
    return locs, child_sitemaps


def _collect_sitemap_urls(client: httpx.Client, entry_urls: list[str], origin: str, max_locs: int = 500) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    stack = list(entry_urls)
    while stack and len(out) < max_locs:
        sm_url = stack.pop()
        if sm_url in seen:
            continue
        seen.add(sm_url)
        body = _fetch_text(client, sm_url, timeout=15.0)
        if not body:
            continue
        locs, children = _parse_sitemap_locs(body)
        stack.extend(children)
        for loc in locs:
            if _same_origin(origin, loc):
                out.append(loc)
                if len(out) >= max_locs:
                    break
    return out


def _filter_scored_sitemap(urls: list[str], origin: str) -> list[str]:
    ranked = [(u, _score_url(u)) for u in urls if _same_origin(origin, u)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in ranked]


async def _homepage_same_origin_links(homepage: str, _trace_id: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    hrefs: list[str] = []
    browser_conf = BrowserConfig(headless=True)
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=60000)
    try:
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=homepage, config=run_conf)
    except Exception as e:
        errors.append(f"homepage_crawl: {e}")
        return [], errors

    if not result.success:
        errors.append(result.error_message or "homepage_crawl_failed")
        return [], errors

    md_extra = ""
    if result.markdown:
        md_extra = getattr(result.markdown, "raw_markdown", None) or str(result.markdown)
    html = (result.html or "") + "\n" + (md_extra or "")
    for m in re.finditer(r'href\s*=\s*"([^"]+)"', html, re.I):
        hrefs.append(m.group(1))
    for m in re.finditer(r"href\s*=\s*'([^']+)'", html, re.I):
        hrefs.append(m.group(1))

    origin_p = urlparse(homepage)
    origin_root = f"{origin_p.scheme}://{origin_p.netloc}"
    resolved: list[str] = []
    for h in hrefs:
        if not h or h.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        abs_u = urljoin(homepage, h.split("#")[0])
        if any(abs_u.lower().split("?")[0].endswith(x) for x in (".jpg", ".png", ".gif", ".css", ".js", ".ico")):
            continue
        if _same_origin(homepage, abs_u):
            clean = abs_u.split("?")[0]
            if not any(clean.lower().split("?")[0].endswith(p) for p in (".jpg", ".png", ".gif")):
                resolved.append(clean)

    # score and dedupe preserve order
    seen = set()
    scored: list[tuple[str, float]] = []
    for u in resolved:
        if u in seen:
            continue
        seen.add(u)
        scored.append((u, _score_url(u)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in scored], errors


def _build_scrape_plan(homepage: str, sitemap_ranked: list[str], home_ranked: list[str], max_urls: int) -> list[str]:
    """Homepage first; merge unique by score heuristic."""
    out: list[str] = []
    seen: set[str] = set()

    def push(u: str) -> None:
        if u not in seen:
            seen.add(u)
            out.append(u)

    push(homepage.rstrip("/") + "/" if not homepage.endswith("/") else homepage)

    # interleave high-value from both sources
    i, j = 0, 0
    while len(out) < max_urls and (i < len(sitemap_ranked) or j < len(home_ranked)):
        if i < len(sitemap_ranked):
            push(sitemap_ranked[i].rstrip("/"))
            i += 1
        if len(out) >= max_urls:
            break
        if j < len(home_ranked):
            push(home_ranked[j].rstrip("/"))
            j += 1

    # fill remainder
    for u in sitemap_ranked[i:] + home_ranked[j:]:
        if len(out) >= max_urls:
            break
        push(u.rstrip("/"))

    return out[:max_urls]


async def _run_discovery_async(inp: DiscoveryInput, parent_trace) -> DiscoveryOutput:
    meta = DiscoveryMeta()
    errors: list[str] = []
    website = inp.lead.website
    if not website:
        return DiscoveryOutput(scrape_plan=[], discovery_meta=meta, errors=["no_website"])

    origin = _normalize_origin(website)
    if not origin:
        return DiscoveryOutput(scrape_plan=[], discovery_meta=meta, errors=["invalid_website"])

    homepage = origin if origin.endswith("/") else origin + "/"

    with httpx.Client() as client:
        span_sm = parent_trace.span(name="sitemap_harvest")
        entry = _robots_sitemap_urls(client, homepage)
        if not entry:
            candidate = f"{urlparse(homepage).scheme}://{urlparse(homepage).netloc}/sitemap.xml"
            entry = [candidate]
        raw_locs = _collect_sitemap_urls(client, entry, homepage, max_locs=400)
        meta.sitemap_urls_found = len(raw_locs)
        meta.strategies.append("sitemap")
        filtered = _filter_scored_sitemap(raw_locs, homepage)
        span_sm.end()

    span_home = parent_trace.span(name="homepage_links")
    home_links, h_err = await _homepage_same_origin_links(homepage, inp.trace_id)
    span_home.end()
    meta.homepage_links_found = len(home_links)
    meta.strategies.append("homepage_links")
    errors.extend(h_err)

    ranked_sm = filtered
    ranked_home = [(u, _score_url(u)) for u in home_links]
    ranked_home.sort(key=lambda x: x[1], reverse=True)
    home_urls = [u for u, _ in ranked_home]

    plan = _build_scrape_plan(homepage, ranked_sm, home_urls, inp.max_urls)
    meta.errors = errors

    parent_trace.event(
        name="discovery_complete",
        metadata={
            "scrape_plan_len": len(plan),
            "sitemap_urls_found": meta.sitemap_urls_found,
            "homepage_links_found": meta.homepage_links_found,
        },
    )

    return DiscoveryOutput(scrape_plan=plan, discovery_meta=meta, errors=errors)


async def discover_urls_node_async(state: dict) -> dict:
    """Graph node: populate scrape_plan + discovery_meta."""
    from ai_agents.agents.email_finder.adapters import discovery_input_from_state

    inp = discovery_input_from_state(state)
    parent = langfuse.trace(id=inp.trace_id, name="email_finder", session_id="email_finder")
    span = parent.span(name="url_discovery")
    try:
        out = await _run_discovery_async(inp, parent)
    finally:
        span.end()
    return {
        "scrape_plan": out.scrape_plan,
        "discovery_meta": out.discovery_meta.model_dump(mode="json"),
        "errors": out.errors,
        "nodes_executed": out.nodes_executed_delta,
        "status": state.get("status", "pending"),
    }
