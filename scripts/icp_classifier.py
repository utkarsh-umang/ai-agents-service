"""ICP / industry classifier runner.

For every lead in an LMS list (batch): crawl the company's homepage + best
services/about page (Serper fallback when the site won't load), then ask
gpt-4o-mini for (a) the primary industry from a controlled taxonomy — the model
may propose a new label when none fit — and (b) a 0-100 paid-advertising-agency
score. Writes the verdict back to the LMS.

Resumable by construction: it asks the LMS for leads NOT yet classified, so a
re-run just continues. Self-contained browser (one Chromium, closed at exit) so
it doesn't depend on the email-finder internals.

  python scripts/icp_classifier.py --batch-id <uuid> [--concurrency 8] \
      [--limit 100] [--max-pages N] [--once]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from urllib.parse import urljoin, urlparse

import httpx
import litellm
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

LMS_API = os.environ.get("LMS_API_URL", "http://localhost:8000").rstrip("/")
MODEL = "gpt-4o-mini"
MAX_CHARS = 12000
PER_LEAD_S = 90
PAGE_TIMEOUT_MS = 20000

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

Positive signals: sells Google/Meta/LinkedIn/TikTok/YouTube/Microsoft Ads MANAGEMENT, PPC/media buying/performance marketing as a done-for-you service, with case studies, ROAS/CPA/CAC, conversion tracking, Google/Meta Partner badges, bid/budget management, remarketing.

CRITICAL — companies that are ABOUT paid media but are NOT agencies must score LOW (0-25), even if their entire site is about ads:
- Courses / education / training that TEACH paid advertising or PPC.
- Software / SaaS / tools / platforms for running or optimizing ads.
- Consultants / coaches / advisors who advise on ads but don't run them.
- Influencer / creator / affiliate marketplaces and platforms.
- Lead-gen or growth "agencies" whose actual product is coaching, software, or consulting.
A single "Google Ads" mention on a web-dev/SEO/branding shop also does NOT make it a paid-ads agency.

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


async def _close_crawler():
    global _CRAWLER
    if _CRAWLER is not None:
        try:
            await _CRAWLER.close()
        except Exception:
            pass
        _CRAWLER = None


# ── content gathering ────────────────────────────────────────────────────────
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


async def _gather_text(company: str, website: str) -> tuple[str, str]:
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


async def _classify_lead(lead: dict, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        company = lead.get("company_name") or ""
        website = lead.get("website") or ""
        try:
            text, src = await asyncio.wait_for(_gather_text(company, website), timeout=PER_LEAD_S)
        except Exception:
            text, src = "", "none"
        if not text:
            # No content anywhere → mark as unclassifiable so it isn't re-fetched
            # forever; icp_confidence 0, industry Other/Unclear, source none.
            return {"lead_id": lead["lead_id"], "classified_industry": "Other / Unclear",
                    "icp_confidence": 0, "icp_reasoning": "no website content (crawl + serper empty)",
                    "icp_source": "none"}
        try:
            c = await asyncio.to_thread(_classify, company, website, text)
        except Exception as e:
            print(f"[icp]   classify error {company[:30]}: {e}")
            return None  # transient — leave unclassified so a re-run retries
        conf = int(c.get("paid_ads_confidence", 0) or 0)
        industry = (c.get("classified_industry") or "Other / Unclear").strip()
        if conf >= 60:
            industry = "Paid Advertising Agency"  # enforce the rule server-side too
        return {"lead_id": lead["lead_id"], "classified_industry": industry,
                "icp_confidence": conf, "icp_reasoning": (c.get("reasoning") or "")[:600],
                "icp_source": src}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=100, help="leads per page")
    ap.add_argument("--max-pages", type=int, default=None, help="stop after N pages (testing)")
    ap.add_argument("--once", action="store_true", help="one page then exit")
    args = ap.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    pages = 0
    done = 0
    accepted = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            while True:
                if args.max_pages is not None and pages >= args.max_pages:
                    print(f"[icp] page limit {args.max_pages} reached")
                    break
                resp = await client.get(f"{LMS_API}/api/v1/classification/pending",
                                        params={"batch_id": args.batch_id, "limit": args.limit})
                resp.raise_for_status()
                leads = resp.json()
                if not leads:
                    print("[icp] nothing pending — done")
                    break
                print(f"[icp] page {pages+1}: classifying {len(leads)} leads")
                results = [r for r in await asyncio.gather(*[_classify_lead(l, sem) for l in leads]) if r]
                if results:
                    pr = await client.post(f"{LMS_API}/api/v1/classification/results",
                                           json={"results": results})
                    pr.raise_for_status()
                    done += pr.json().get("updated", 0)
                    accepted += sum(1 for r in results if r["icp_confidence"] >= 60)
                pages += 1
                print(f"[icp]   committed. total classified={done}  paid_ads so far={accepted}")
                if args.once:
                    break
        finally:
            await _close_crawler()

    print(f"[icp] finished — {done} classified this run, {accepted} paid-ads agencies")


if __name__ == "__main__":
    asyncio.run(main())
