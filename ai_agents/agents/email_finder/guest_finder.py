"""Podcast-guest email finder — the low-cost, search-first pipeline.

The main email-finder graph is a *crawl-a-known-site* cascade, which works for
YouTube channels / SaaS / Apollo contacts (they have a domain) but scores ~8% on
podcast guests, who are individuals with no company domain to crawl. This module
is the guest-specific replacement, benchmarked end-to-end (25 labelled guests):

    low-cost crawl 8%  ->  raw serper 65%  ->  +filters 74%  ->  +precision 82%

Flow (all cheap, ~$2/1k, no paid Perplexity):
    1. Serper search               — find the pages that could hold the email
    2. Crawl4AI scrape (JS+md)      — free; ScrapingBee(JS) fallback on anti-bot
    3. LLM rank (gpt-4o-mini)       — pool-only, person + affiliation bound,
                                      source-trust aware (distrust data-brokers)
    4. deterministic guards         — in-pool (kill invented), reject-generic
                                      (domain-name exception), suspicious local-
                                      part, MX/SMTP verify, confidence gate

Returns a state dict shaped like the graph's output so the LMS worker's
``_result_payload`` consumes it unchanged:
    {"best_email": {"email","source","confidence","note"} | None,
     "status": LeadStatus, "cost_usd": float, "nodes_executed": [...]}

Design notes:
- Synchronous by design. Crawl4AI is driven via a private event loop
  (``asyncio.run``); call this from the worker with ``asyncio.to_thread`` so the
  worker's loop is never blocked (mirrors how the graph offloads sync I/O).
- Known-remaining misses (see the benchmark): stale addresses, entertainers with
  no findable personal email, and multi-person institutional directories (which
  need structure-aware extraction). Tracked for a later pass.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from typing import Any

import requests

from ai_agents.agents.email_finder.state import LeadStatus

logger = logging.getLogger(__name__)

_EMAIL_RX = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_IMG_EXT = (".png", ".jpg", ".gif", ".jpeg", ".webp", ".svg")

# domains that are never the lead's own page — scraping an email off one of these
# is a guessed/broker address, not a first-party contact.
_SOCIAL_DIRECTORY = ("linkedin.com", "twitter.com", "x.com", "instagram.com", "facebook.com",
    "tiktok.com", "youtube.com", "apple.com", "spotify.com", "wikipedia.org", "imdb.com",
    "amazon.com", "podchaser.com", "listennotes.com")
_AGGREGATORS = ("rocketreach", "contactout", "signalhire", "leadiq", "zoominfo", "spokeo",
    "apollo.io", "hunter.io", "anymailfinder", "clearbit", "lusha", "adapt.io", "seamless.ai",
    "usphonebook", "whitepages", "beenverified", "truepeoplesearch", "fastpeoplesearch",
    "peoplefinder", "radaris", "nuwber", "clustrmaps", "emailsherlock", "voilanorbert",
    "snov.io", "findymail", "getprospect", "contactanycelebrity", "kendo", "uplead",
    "leadfeeder", "emailhippo", "neverbounce")
# generic role local-parts. On the person's OWN-name domain they're kept (a
# personal brand's direct line, e.g. hi@janedoe.com); otherwise rejected.
_GENERIC_LP = {"info", "contact", "hello", "support", "admin", "team", "office", "sales", "help",
    "media", "press", "inquiries", "inquiry", "enquiries", "general", "mail", "booking",
    "bookings", "pr", "marketing", "newsletter", "careers", "jobs", "webmaster", "postmaster",
    "noreply", "no-reply", "hi", "hey", "podcast", "podcasts", "feedback", "hq", "studio",
    "partnerships", "partner", "collab", "collabs", "business", "biz"}
_SUSPICIOUS_LP = ("official", "real", "thereal", "fanmail", "fan", "fake")
_CATCHALL_PROVIDERS = ("gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "yahoo.com", "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com", "msn.com")
_CONF_GATE = 0.72
_MODEL = "gpt-4o-mini"


class GuestFinderConfig:
    def __init__(self) -> None:
        self.serper_key = os.environ.get("SERPER_API_KEY")
        self.scrapingbee_key = os.environ.get("SCRAPINGBEE_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.crawl_timeout_s = 35
        self.max_pages = 2


# --- helpers -----------------------------------------------------------------
def _domain(u: str | None) -> str:
    return re.sub(r"^https?://", "", (u or "").lower()).split("/")[0]

def _is_aggregator(url: str) -> bool:
    d = _domain(url)
    return any(a in d for a in _AGGREGATORS)

def _person_tokens(name: str | None) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if len(t) >= 3]

def _is_generic(email: str, ptoks: list[str]) -> bool:
    lp, _, dom = email.lower().partition("@")
    if lp in _GENERIC_LP:
        return not any(t in dom for t in ptoks)   # keep on own-name domain
    return False


def _name_bound(email: str, ptoks: list[str]) -> bool:
    """True iff the email plausibly belongs to THIS person: a name token appears
    in the local-part or the domain. This is the single strongest precision
    signal on multi-person organization pages (an Economist/Atlantic/university
    page holds dozens of staff emails on the same domain; only the one bearing
    the person's name is theirs). A live 2,245-lead run showed ~40% of raw
    'found' emails were a colleague's or a department inbox — every one failed
    this check, every correct one passed it.

    Trade-off: rejects cryptic institutional IDs that don't spell the name
    (sv2r@virginia.edu, mmain@ufl.edu). Recovering those correctly needs
    structure-aware extraction (bind the email to the name's position on the
    page) — the Tier-2 follow-up; until then precision wins over that tail."""
    if not ptoks:
        return True
    lp, _, dom = email.lower().partition("@")
    return any(t in lp for t in ptoks) or any(t in dom for t in ptoks)


def _email_contexts(text: str, window: int = 140):
    """Every email in `text` with the surrounding text window — so we can bind
    an email to the name it sits next to on a multi-person page (Tier 2)."""
    out = []
    for m in _EMAIL_RX.finditer(text or ""):
        s = max(0, m.start() - window)
        e = min(len(text), m.end() + window)
        out.append((m.group(0).lower(), text[s:e]))
    return out


def _url_has_name(url: str, ptoks: list[str]) -> bool:
    u = (url or "").lower()
    return any(t in u for t in ptoks)

def _mx_hosts(domain: str) -> list[str] | None:
    try:
        out = subprocess.run(["dig", "+short", "MX", domain],
                             capture_output=True, text=True, timeout=8).stdout.strip()
        return [l.split()[-1].rstrip(".") for l in out.splitlines() if l.split()]
    except Exception:
        return None

def _verify(email: str) -> str:
    dom = email.split("@")[-1].lower()
    mx = _mx_hosts(dom)
    if mx is None:
        return "unknown"
    if not mx:
        return "invalid_nomx"
    if any(dom == d for d in _CATCHALL_PROVIDERS):
        return "catchall_provider"
    try:
        import smtplib
        s = smtplib.SMTP(timeout=6)
        s.connect(mx[0], 25); s.helo("example.com"); s.mail("probe@example.com")
        code, _m = s.rcpt(email); s.quit()
        if code in (250, 251):
            return "valid"
        if code in (550, 551, 553, 501):
            return "invalid_mailbox"
        return "unknown"
    except Exception:
        return "unknown"   # port 25 blocked / greylisted — don't punish

def _lead_context(lead: dict) -> str:
    parts = [f"Name: {lead.get('name')}"]
    if lead.get("company"): parts.append(f"Company/Affiliation: {lead['company']}")
    if lead.get("occupation"): parts.append(f"Role: {lead['occupation']}")
    if lead.get("podcast"): parts.append(f"Was a guest on the podcast: {lead['podcast']}")
    for k in ("website", "linkedin", "twitter", "instagram"):
        if lead.get(k): parts.append(f"{k}: {lead[k]}")
    return "\n".join(parts)


def _not_found(cost: float, note: str, nodes: list[str]) -> dict:
    return {"best_email": None, "status": LeadStatus.EMAIL_NOT_FOUND.value,
            "cost_usd": round(cost, 6), "nodes_executed": nodes, "note": note}


# --- main --------------------------------------------------------------------
def find_guest_email(lead: dict, cost_mode: str = "low",
                     config: GuestFinderConfig | None = None) -> dict[str, Any]:
    """lead keys: name, company, occupation, industry, website, linkedin,
    twitter, instagram, podcast. Returns a graph-shaped state dict."""
    cfg = config or GuestFinderConfig()
    nodes: list[str] = []
    cost = 0.0
    if not cfg.serper_key:
        return {"best_email": None, "status": LeadStatus.FAILED.value, "cost_usd": 0.0,
                "nodes_executed": nodes, "errors": ["SERPER_API_KEY not set"]}
    try:
        # 1. SEARCH
        nodes.append("serper_search")
        q = f'{lead.get("name","")} {lead.get("company","")} email contact'.strip()
        organic = requests.post("https://google.serper.dev/search",
                                headers={"X-API-KEY": cfg.serper_key, "Content-Type": "application/json"},
                                json={"q": q, "num": 10}, timeout=30).json().get("organic", [])
        cost += 0.001

        ptoks = _person_tokens(lead.get("name"))
        prov: dict[str, set[str]] = {}
        ctx: dict[str, list[str]] = {}   # email -> text windows it appeared in (Tier 2 proximity)
        def add(email: str, url: str, context: str = "") -> None:
            e = email.lower()
            if e.endswith(_IMG_EXT):
                return
            prov.setdefault(e, set()).add(url or "")
            if context:
                ctx.setdefault(e, []).append(context)
        for o in organic:
            snip = o.get("snippet", "") + " " + o.get("title", "")
            for e in _EMAIL_RX.findall(snip):
                add(e, o.get("link", ""), snip)
        # Tier 2b: prefer the person's OWN page (name in the URL slug, e.g.
        # /contributor/jane-doe) over a directory — on their own page the only
        # email is theirs, so the multi-person ambiguity never arises.
        cand_urls = [o["link"] for o in organic if o.get("link")
                     and not any(d in o["link"].lower() for d in _SOCIAL_DIRECTORY)]
        cand_urls.sort(key=lambda u: not _url_has_name(u, ptoks))
        cands = cand_urls[:cfg.max_pages]

        # 2. SCRAPE — Crawl4AI primary, ScrapingBee(JS) fallback
        nodes.append("scrape")
        async def crawl_all(urls: list[str]) -> list[str]:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
            out: list[str] = []
            async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as cr:
                for u in urls:
                    try:
                        res = await asyncio.wait_for(
                            cr.arun(url=u, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=25000)),
                            timeout=cfg.crawl_timeout_s)
                        out.append(getattr(res.markdown, "raw_markdown", None) or str(res.markdown or ""))
                    except Exception:
                        out.append("")
            return out
        c4a = asyncio.run(crawl_all(cands)) if cands else []
        for u, md in zip(cands, c4a):
            for e, cx in _email_contexts(md):
                add(e, u, cx)
        for u, md in zip(cands, c4a):
            if cfg.scrapingbee_key and len(md or "") < 300:
                nodes.append("scrapingbee_fallback")
                try:
                    p = requests.get("https://app.scrapingbee.com/api/v1/",
                                     params={"api_key": cfg.scrapingbee_key, "url": u, "render_js": "true"},
                                     timeout=60)
                    cost += 0.005
                    for e, cx in _email_contexts(p.text):
                        add(e, u, cx)
                except Exception:
                    pass
        if not prov:
            return _not_found(cost, "no candidate emails scraped", nodes)

        # 3. LLM RANK — pool-only, person + affiliation bound, source-trust aware,
        #    with Tier-2 name-proximity ("does the person's name sit next to this
        #    email on the page") as the strongest single signal.
        nodes.append("llm_select")
        def _name_near(email: str) -> bool:
            return any(any(t in w.lower() for t in ptoks) for w in ctx.get(email, []))
        lines = []
        for e, urls in prov.items():
            srcs = sorted({_domain(u) for u in urls if u})
            trusted = any(not _is_aggregator(u) for u in urls)
            near = "NAME-IS-NEXT-TO-IT" if _name_near(e) else "name-not-near-it"
            lines.append(f"{e}  | sources: {', '.join(srcs) or '?'} | "
                         f"{'trusted-page' if trusted else 'DATA-BROKER-ONLY'} | {near}")
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_key)
        resp = client.chat.completions.create(model=_MODEL, temperature=0, messages=[{"role": "user", "content":
            f"{_lead_context(lead)}\n\nCandidate emails scraped for this person (with where each was found):\n"
            + "\n".join(lines) +
            "\n\nPick the ONE email that belongs to THIS SPECIFIC person. Hard rules:\n"
            "- STRONGEST signal: an email marked NAME-IS-NEXT-TO-IT sits beside this person's name on "
            "the page — strongly prefer it even if the local-part doesn't spell their name (e.g. a "
            "cryptic university ID). An email marked name-not-near-it on a shared org page is likely a "
            "COLLEAGUE's or a department inbox -> avoid unless nothing better exists.\n"
            "- The org/domain must be consistent with the person's role/company above. An email whose "
            "organization is unrelated to their known work is a DIFFERENT person with the same name -> reject.\n"
            "- Reject an email found ONLY on data-broker pages (marked DATA-BROKER-ONLY).\n"
            "- Prefer their own site / employer / institution. A lone generic role inbox is weak.\n"
            "- Only choose from the list. If none clearly belong to this person, return null.\n"
            'Return ONLY JSON {"email":"<from list or null>","confidence":<0..1>,"why":"<short>"}'}])
        cost += 0.0005
        m = re.search(r"\{.*\}", resp.choices[0].message.content, re.S)
        pick = {}
        if m:
            try:
                pick = json.loads(m.group(0))
            except Exception:
                pick = {}
        em = (pick.get("email") or "").strip().lower()

        # 4. DETERMINISTIC GUARDS
        nodes.append("guards")
        if not em or "@" not in em or em not in prov:
            return _not_found(cost, f"no valid in-pool pick ({em or 'null'})", nodes)
        lp = em.split("@")[0]
        if any(s in lp for s in _SUSPICIOUS_LP):
            return _not_found(cost, f"{em}: suspicious local-part", nodes)
        if _is_generic(em, ptoks):
            return _not_found(cost, f"{em}: generic role inbox", nodes)
        # Bind to the person: EITHER the name is in the email (Tier 1) OR the
        # name sits next to it on the page (Tier 2). This recovers cryptic-but-
        # correct institutional IDs while still rejecting a colleague's address
        # (whose neighbour on the page is THEIR name, not the target's).
        if not (_name_bound(em, ptoks) or _name_near(em)):
            return _not_found(cost, f"{em}: not bound to person (name not in it nor near it on page)", nodes)
        if not any(not _is_aggregator(u) for u in prov[em]):
            return _not_found(cost, f"{em}: data-broker-only source", nodes)
        vs = _verify(em)
        if vs in ("invalid_nomx", "invalid_mailbox"):
            return _not_found(cost, f"{em}: {vs}", nodes)

        conf = float(pick.get("confidence") or 0.5)
        if any(t in lp for t in ptoks):
            conf = min(1.0, conf + 0.08)
        conf = min(1.0, conf + 0.1) if vs == "valid" else conf * 0.9
        conf = round(conf, 2)
        if conf < _CONF_GATE:
            return _not_found(cost, f"{em}: confidence {conf} < gate {_CONF_GATE}", nodes)

        return {
            "best_email": {"email": em, "source": "; ".join(cands),
                           "confidence": conf, "note": f"verify={vs}; {pick.get('why','')[:90]}"},
            "status": LeadStatus.EMAIL_FOUND.value,
            "cost_usd": round(cost, 6),
            "nodes_executed": nodes,
        }
    except Exception as exc:
        logger.exception("guest_finder failed for %s", lead.get("name"))
        return {"best_email": None, "status": LeadStatus.FAILED.value,
                "cost_usd": round(cost, 6), "nodes_executed": nodes, "errors": [str(exc)[:200]]}
