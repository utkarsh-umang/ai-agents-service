"""ICP / industry classifier core — shared by the CLI script and the worker.

Crawl a company's homepage + best services/about page (Serper fallback when the
site won't load), then ask gpt-4o-mini for its industry (controlled taxonomy;
the model may propose a new label when none fit) plus a 0-100
paid-advertising-agency score. `drain_batch` runs the LMS pending -> classify ->
results loop for one list.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from urllib.parse import urljoin, urlparse

import httpx
import litellm
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

MODEL = "gpt-4o-mini"
MAX_CHARS = 12000
PER_LEAD_S = 90
PAGE_TIMEOUT_MS = 20000
ICP_ACCEPT_THRESHOLD = 60

# Canonical industry taxonomy. The model prefers these; when none genuinely fit
# it may propose a new concise Title-Case label (governance: review off-taxonomy
# labels after a run and fold frequent ones back into this list).
TAXONOMY = [
    "Paid Advertising Agency",
    "SEO Agency",
    "Full-Service / Digital Marketing Agency",
    "Social / Content / PR Agency",
    "Branding / Creative / Design",
    "Web Design / Development",
    "Software / SaaS / Tech Platform",
    "E-commerce / Retail Brand",
    "Real Estate",
    "Recruitment / Staffing",
    "Consulting / Professional Services",
    "Media / Publisher",
    "Education / Coaching",
    "Healthcare / Pharma",
    "Hospitality / Travel / Events",
    "Nonprofit / Government",
    "Other / Unclear",
]

_LINK_KW = ("service", "about", "ppc", "paid", "advertis", "media", "marketing",
            "what-we-do", "solution", "capabilit", "expertise")

PROMPT = """You are an expert B2B company classification AI. You are given the textual content of a company's website (homepage + a services/about page). Infer the company's PRIMARY BUSINESS MODEL from the whole site — never from isolated keywords.

Do TWO things:

1) classified_industry — the company's primary industry. PREFER exactly one of these canonical buckets:
{taxonomy}
If NONE of them genuinely fits, output a concise NEW industry label in Title Case instead of forcing a poor fit. Do not invent a new label when a canonical one fits.

2) paid_ads_confidence — an integer 0-100 for how strongly the site shows Paid Advertising / Paid Media / PPC / Performance Marketing is a CORE, revenue-generating business. This is a one-directional "paid-media-agency-ness" score, NOT how certain you are. A company that clearly is NOT a paid-ads agency must score LOW (0-25) even if you are 100% sure.
- 90-100: entire business is paid media / performance marketing.
- 80-89: paid media is one of the largest services.
- 70-79: clearly a major, core service (PPC/paid media).
- 60-69: genuinely offers paid ads as a real service, one of several, not dominant.
- 40-59: mentions paid ads but minor/incidental, OR full-service where paid media is unclear.
- 0-39: not a paid advertising agency at all.
The core test: would a business owner hire THIS company as a SERVICE to run and manage their paid campaigns (they log into ad accounts and manage spend on the client's behalf)? Only that is a paid advertising agency.
CRITICAL — companies that are ABOUT paid media but are NOT agencies must score LOW (0-25), even if their entire site is about ads: courses/education/training that TEACH ads; software/SaaS/tools/platforms for running ads; consultants/coaches/advisors who advise but don't run; influencer/creator/affiliate marketplaces; lead-gen or growth "agencies" whose real product is coaching/software/consulting. A single "Google Ads" mention on a web-dev/SEO/branding shop also does NOT make it a paid-ads agency.

RULE: if paid_ads_confidence >= 60, classified_industry MUST be "Paid Advertising Agency". Only give >= 60 to companies that actually RUN campaigns as a managed service.

Respond ONLY with JSON:
{{"classified_industry": "<bucket or new label>", "paid_ads_confidence": <int 0-100>, "reasoning": "<2-3 sentences>"}}""".format(
    taxonomy="\n".join(f"- {t}" for t in TAXONOMY)
)


# ── self-contained shared browser ────────────────────────────────────────────
_CRAWLER = None


async def _get_crawler():
    global _CRAWLER
    if _CRAWLER is None:
        _CRAWLER = AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False))
        await _CRAWLER.start()
    return _CRAWLER


async def _crawl(url, run_config):
    c = await _get_crawler()
    return await c.arun(url=url, config=run_config)


async def close_crawler():
    global _CRAWLER
    if _CRAWLER is not None:
        try:
            await _CRAWLER.close()
        except Exception:
            pass
        _CRAWLER = None


def _norm(url: str) -> str:
    url = (url or "").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _text(result) -> str:
    if not result or not getattr(result, "success", False):
        return ""
    md = ""
    if result.markdown:
        md = getattr(result.markdown, "raw_markdown", None) or str(result.markdown)
    return md or (result.html or "")


def _pick_second_url(homepage: str, html: str) -> str | None:
    origin = urlparse(homepage).netloc
    best, best_score = None, 0
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html or "", re.I):
        href = m.group(1).split("#")[0]
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absu = urljoin(homepage, href)
        p = urlparse(absu)
        if p.netloc != origin:
            continue
        path = p.path.lower()
        if path in ("", "/"):
            continue
        score = sum(1 for kw in _LINK_KW if kw in path)
        if re.search(r"/(about|services?)(/|$)", path):
            score += 2
        if score > best_score:
            best, best_score = absu, score
    return best if best_score > 0 else None


async def gather_text(company: str, website: str) -> tuple[str, str]:
    """(combined_text, source). source = crawl | serper | none."""
    url = _norm(website)
    rc = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=PAGE_TIMEOUT_MS)
    try:
        home = await _crawl(url, rc)
    except Exception:
        home = None
    home_txt = _text(home)
    if home_txt:
        parts = [home_txt]
        second = _pick_second_url(url, (home.html or "") if home else "")
        if second:
            try:
                t2 = _text(await _crawl(second, rc))
                if t2:
                    parts.append(t2)
            except Exception:
                pass
        return ("\n\n".join(parts)[:MAX_CHARS], "crawl")
    key = os.environ.get("SERPER_API_KEY")
    if key and company:
        try:
            r = httpx.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json={"q": f"{company} advertising agency", "num": 8}, timeout=20,
            ).json()
            txt = "\n".join(o.get("snippet", "") for o in r.get("organic", []) if o.get("snippet"))
            if txt:
                return (txt[:MAX_CHARS], "serper")
        except Exception:
            pass
    return ("", "none")


def _classify(company: str, website: str, text: str) -> dict:
    resp = litellm.completion(
        model=MODEL, timeout=60, temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"COMPANY: {company}\nWEBSITE: {website}\n\nWEBSITE CONTENT:\n{text}"},
        ],
    )
    return json.loads(resp.choices[0].message.content)


async def classify_lead(lead: dict, sem: asyncio.Semaphore) -> dict | None:
    """One lead -> a results row for POST /classification/results, or None on a
    transient error (leave it unclassified so a re-run retries)."""
    async with sem:
        company = lead.get("company_name") or ""
        website = lead.get("website") or ""
        try:
            text, src = await asyncio.wait_for(gather_text(company, website), timeout=PER_LEAD_S)
        except Exception:
            text, src = "", "none"
        if not text:
            return {"lead_id": lead["lead_id"], "classified_industry": "Other / Unclear",
                    "icp_confidence": 0, "icp_reasoning": "no website content (crawl + serper empty)",
                    "icp_source": "none"}
        try:
            c = await asyncio.to_thread(_classify, company, website, text)
        except Exception as exc:
            print(f"[icp]   classify error {company[:30]}: {exc}")
            return None
        conf = int(c.get("paid_ads_confidence", 0) or 0)
        industry = (c.get("classified_industry") or "Other / Unclear").strip()
        if conf >= ICP_ACCEPT_THRESHOLD:
            industry = "Paid Advertising Agency"
        return {"lead_id": lead["lead_id"], "classified_industry": industry,
                "icp_confidence": conf, "icp_reasoning": (c.get("reasoning") or "")[:600],
                "icp_source": src}


async def drain_batch(
    client: httpx.AsyncClient, lms_api: str, batch_id: str,
    sem: asyncio.Semaphore, limit: int = 100, max_pages: int | None = None,
    should_continue=None,
) -> tuple[int, int]:
    """Classify a list's pending leads, page by page, until empty. Returns
    (classified, accepted). `should_continue` (optional) is an async callable
    checked each page — return False to stop early (e.g. Stop was pressed)."""
    done, accepted, pages = 0, 0, 0
    while True:
        if max_pages is not None and pages >= max_pages:
            break
        if should_continue is not None and not await should_continue():
            print(f"[icp] batch {batch_id}: stop requested")
            break
        resp = await client.get(f"{lms_api}/api/v1/classification/pending",
                                params={"batch_id": batch_id, "limit": limit})
        resp.raise_for_status()
        leads = resp.json()
        if not leads:
            break
        results = [r for r in await asyncio.gather(*[classify_lead(l, sem) for l in leads]) if r]
        if results:
            pr = await client.post(f"{lms_api}/api/v1/classification/results", json={"results": results})
            pr.raise_for_status()
            done += pr.json().get("updated", 0)
            accepted += sum(1 for r in results if r["icp_confidence"] >= ICP_ACCEPT_THRESHOLD)
        pages += 1
        print(f"[icp] batch {batch_id}: +{len(results)} (total {done}, paid_ads {accepted})")
    return done, accepted
