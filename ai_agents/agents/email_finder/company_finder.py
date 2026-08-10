"""Company-as-lead contact resolver — for Clutch agency leads (the lead IS a
company, with no email, no person, and no website; only a Clutch profile URL).

The person-centric finders (crawl-graph, guest_finder) can't start here: there's
no domain to crawl and no name to bind. So this resolver first turns a company
into a single senior contact, in three stages:

  1. profile -> domain   Crawl the Clutch profile (Cloudflare -> Crawl4AI) and
                          lift the agency's real site from the "Visit Website"
                          tracking link (`provider_website=`). Fallback: Serper
                          name-search for the domain.
  2. domain -> person     Fetch the site's team/about/leadership pages, extract
                          {name,title} with gpt-4o-mini, rank by the caller's
                          seniority preference (CEO > CMO > CBO > CxO > Founder/
                          President/Owner/MD/Partner > SVP/EVP > VP > Director >
                          Head). Fallback: Serper LinkedIn snippets.
  3. person -> email       Generate address patterns + harvest on-page emails,
                          then VERIFY each candidate via Mailin (blind patterns
                          are unreliable — the CEO of a real agency verified
                          `invalid`). Keep the first `ok`. If no personal address
                          verifies, fall back to a verified generic (info@/…).

Returns ONE contact (the design choice: one usable contact per agency beats a
fan-out of unverified guesses), shaped like the other finders' state dict so the
worker maps it to an /enrichment/results payload — plus the person + email_status
so a company lead learns WHOSE address it is.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import unicodedata
from typing import Any

import httpx

from ai_agents.agents.email_finder.nodes.verify_email_mailin import verify_email_mailin
from ai_agents.agents.email_finder.state import LeadStatus

_MODEL = "gpt-4o-mini"
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/121.0 Safari/537.36")

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_GENERIC_LP = {"info", "hello", "contact", "hi", "hey", "sales", "office", "team",
               "mail", "admin", "enquiries", "enquiry", "hq", "studio", "general"}
# Preferred generic local-parts to try as a last-resort contact, best first.
_GENERIC_TRY = ["info", "hello", "contact", "hi", "team", "enquiries", "office"]
_TEAM_HINTS = ["team", "about", "about-us", "aboutus", "leadership", "our-team", "people",
               "who-we-are", "company", "our-people", "meet-the-team", "management",
               "staff", "founders", "leaders"]

# Seniority tiers — higher rank = more preferred (the caller's stated order).
_SENIORITY = [
    (re.compile(r"\b(chief executive officer|chief executive|ceo)\b", re.I), 100),
    (re.compile(r"\b(chief marketing officer|cmo)\b", re.I), 95),
    (re.compile(r"\b(chief brand officer|cbo)\b", re.I), 90),
    (re.compile(r"\b(chief (financial|operating|revenue|growth|commercial) officer|cfo|coo|cro|cco)\b", re.I), 85),
    (re.compile(r"\bchief\b", re.I), 82),  # any other C-level
    (re.compile(r"\b(co-?founder|founder|owner|president|managing director|managing partner)\b", re.I), 80),
    (re.compile(r"\b(managing partner|partner|principal)\b", re.I), 70),
    (re.compile(r"\b(executive vice president|senior vice president|evp|svp)\b", re.I), 65),
    (re.compile(r"\b(vice president|vp)\b", re.I), 58),
    (re.compile(r"\bdirector\b", re.I), 50),
    (re.compile(r"\bhead of\b", re.I), 45),
]


class CompanyFinderConfig:
    def __init__(self) -> None:
        self.serper_key = os.environ.get("SERPER_API_KEY")
        self.firecrawl_key = os.environ.get("FIRECRAWL_API_KEY")  # optional Stage-1 fallback
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.http_timeout_s = 15.0
        self.crawl_timeout_s = 40


def _seniority_rank(title: str) -> int:
    for rx, rank in _SENIORITY:
        if rx.search(title or ""):
            return rank
    return 0


def _html_text(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _domain_matches(domain: str, email_domain: str) -> bool:
    stem = domain.split(".")[0]
    return email_domain == domain or (len(stem) > 3 and stem in email_domain)


def _mx_ok(domain: str) -> bool:
    """Cheap gate before spending Mailin credits: does the domain accept mail?"""
    try:
        mx = subprocess.run(["dig", "+short", "MX", domain], capture_output=True,
                            text=True, timeout=8).stdout.strip()
        if mx:
            return True
        a = subprocess.run(["dig", "+short", "A", domain], capture_output=True,
                           text=True, timeout=8).stdout.strip()
        return bool(a)
    except Exception:
        return True  # don't punish on a dig failure — let Mailin decide


def _ascii_fold(s: str) -> str:
    """Strip accents so a name maps to the address its mailbox actually uses:
    'Lütke' -> 'lutke', not 'ltke' (the naive [^a-z] strip dropped the ü and
    silently produced tobi.ltke@ — a wrong, unverifiable guess)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))


def _patterns(full_name: str, domain: str) -> list[str]:
    parts = re.sub(r"[^a-z\s\-]", "", _ascii_fold((full_name or "").lower())).split()
    if len(parts) < 2:
        return []
    f, l = parts[0], parts[-1]
    fi = f[0]
    # Ordered by global corporate frequency so the pattern-probe learns the
    # domain's format in as few verifies as possible: first.last, flast, first,
    # firstlast, then the long tail.
    locals_ = [f"{f}.{l}", f"{fi}{l}", f"{f}", f"{f}{l}", f"{fi}.{l}", f"{f}_{l}", f"{f}-{l}", f"{l}.{f}", f"{l}{fi}"]
    seen, out = set(), []
    for lp in locals_:
        if lp and lp not in seen:
            seen.add(lp)
            out.append(f"{lp}@{domain}")
    return out


def _local_matches_name(local: str, full_name: str) -> bool:
    parts = re.sub(r"[^a-z\s\-]", "", _ascii_fold((full_name or "").lower())).split()
    if len(parts) < 2:
        return False
    f, l = parts[0], parts[-1]
    return local.lower() in {f"{f}.{l}", f"{f}{l}", f"{f[0]}{l}", f, f"{f[0]}.{l}",
                             f"{f}_{l}", f"{f}-{l}", f"{l}.{f}", f"{l}{f[0]}"}


# ---------- network ----------
async def _get(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, headers={"User-Agent": _UA}, follow_redirects=True)
        if r.status_code == 200 and "just a moment" not in r.text[:2000].lower():
            return r.text
    except Exception:
        pass
    return ""


async def _crawl(url: str, timeout_s: int) -> str:
    """Real local browser (free, clears most Cloudflare). Always closes."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
        async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as c:
            res = await c.arun(url=url, config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS, magic=True, simulate_user=True,
                page_timeout=timeout_s * 1000, wait_until="domcontentloaded",
                delay_before_return_html=3.0))
        return res.html or "" if getattr(res, "success", False) else ""
    except Exception:
        return ""


async def _serper(client: httpx.AsyncClient, key: str, q: str) -> dict:
    try:
        r = await client.post("https://google.serper.dev/search",
                              headers={"X-API-KEY": key, "Content-Type": "application/json"},
                              json={"q": q, "num": 10})
        return r.json()
    except Exception:
        return {"organic": []}


async def _gpt(client: httpx.AsyncClient, key: str, system: str, user: str) -> str:
    try:
        r = await client.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": _MODEL, "temperature": 0, "max_tokens": 900,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user[:14000]}]})
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""


# ---------- stages ----------
async def _resolve_domain(client, cfg, profile_url, company) -> tuple[str | None, str]:
    html = await _crawl(profile_url, cfg.crawl_timeout_s) if profile_url else ""
    if html:
        m = re.search(r"provider_website=([^&\"'\s]+)", html)
        if m:
            return m.group(1).strip().lower().lstrip("www."), "clutch_profile"
    # Fallback: Serper name-search (less authoritative, ~66% agreement).
    if cfg.serper_key and company:
        res = await _serper(client, cfg.serper_key, f"{company} agency official website")
        skip = ("clutch.co", "facebook.", "linkedin.", "twitter.", "instagram.", "youtube.",
                "yelp.", "glassdoor.", "crunchbase.", "goodfirms.", "designrush.", "g2.com",
                "trustpilot.", "wikipedia.", "sortlist.", "google.")
        for o in res.get("organic", []):
            dom = re.sub(r"^https?://", "", (o.get("link") or "").lower()).split("/")[0].lstrip("www.")
            if dom and not any(s in dom for s in skip):
                return dom, "serper_search"
    return None, "unresolved"


_EXTRACT_SYS = (
    "You extract senior leadership from an agency's website text. Return STRICT JSON "
    '{"people":[{"name":"Full Name","title":"Exact Title"}]}. ONLY real named individuals '
    "in a senior role: C-level (CEO/CFO/COO/CMO/CBO/CTO/CRO), Founder/Co-Founder/Owner/"
    "President/Principal/Partner/Managing Director, or VP/SVP/EVP/Head of/Director. Exclude "
    "junior staff, generic team members, testimonial authors, and clients. If none, return "
    '{"people":[]}. JSON only, no prose.'
)


def _parse_people(raw: str) -> list[dict]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out, seen = [], set()
    for p in data.get("people", []):
        name, title = (p.get("name") or "").strip(), (p.get("title") or "").strip()
        if name and title and " " in name and _seniority_rank(title) > 0 and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "title": title, "rank": _seniority_rank(title)})
    out.sort(key=lambda x: x["rank"], reverse=True)
    return out


async def _find_people(client, cfg, domain, company) -> tuple[list[dict], set[str], str]:
    """Return (ranked people, on-page agency emails, source)."""
    home = await _get(client, f"https://{domain}") or await _get(client, f"http://{domain}")
    if not home:
        home = await _crawl(f"https://{domain}", cfg.crawl_timeout_s)
    page_emails = {e.lower() for e in _EMAIL_RE.findall(home)
                   if _domain_matches(domain, e.split("@")[-1].lower())}
    texts = [_html_text(home)]
    # team/about pages from real nav links + two classic guesses
    urls, low = [], home.lower()
    for m in re.finditer(r'href="([^"#]+)"', home, re.I):
        href = m.group(1)
        if any(h in href.lower() for h in _TEAM_HINTS):
            full = href if href.startswith("http") else f"https://{domain}/{href.lstrip('/')}"
            if domain in full and full not in urls and full.count("/") < 7:
                urls.append(full)
    for p in ("about", "team"):
        u = f"https://{domain}/{p}"
        if u not in urls:
            urls.append(u)
    for u in urls[:4]:
        h = await _get(client, u)
        if h and len(h) > 500:
            texts.append(_html_text(h))
            for e in _EMAIL_RE.findall(h):
                if _domain_matches(domain, e.split("@")[-1].lower()):
                    page_emails.add(e.lower())
    people = _parse_people(await _gpt(client, cfg.openai_key, _EXTRACT_SYS, " ".join(texts)[:16000]))
    src = "site"
    # FREE pattern signal (0 Mailin credits): harvest any published personal
    # address on this domain from a search snippet, so Step 0 of _pick_contact
    # can match it to a senior person and skip the verify-probe entirely.
    if cfg.serper_key:
        res = await _serper(client, cfg.serper_key, f'"@{domain}"')
        for o in res.get("organic", [])[:8]:
            for e in _EMAIL_RE.findall(f"{o.get('title','')} {o.get('snippet','')} {o.get('link','')}"):
                if _domain_matches(domain, e.split("@")[-1].lower()) and e.split("@")[0] not in _GENERIC_LP:
                    page_emails.add(e.lower())
    if not people and cfg.serper_key:
        res = await _serper(client, cfg.serper_key,
                            f'"{company}" (CEO OR founder OR "managing director" OR CMO OR president) site:linkedin.com/in')
        snips = " ".join((o.get("title", "") + " . " + o.get("snippet", ""))
                         for o in res.get("organic", [])[:8])
        if snips:
            people = _parse_people(await _gpt(client, cfg.openai_key, _EXTRACT_SYS, snips))
            src = "serper" if people else "none"
    return people, page_emails, src


async def _pick_contact(cfg, domain, people, page_emails, max_patterns):
    """Credit-optimal: the email pattern is per-DOMAIN, not per-person, so we
    only need ONE senior person + the domain's pattern to have an address for
    them. We therefore:
      0. take a FREE win if a scraped on-page email already matches a senior
         person (their published address) — verify once to confirm;
      1. PROBE the single top person's likeliest patterns to learn the domain's
         format, stopping at the first `ok` (or accepting a catch-all as-is);
      2. EARLY-BAIL if those all verify `invalid` — every mailbox on this domain
         shares the pattern, so trying other PEOPLE is pure wasted credits;
      3. fall back to just two generics (info@, contact@), not seven.
    A `catch_all` verdict is accepted (the domain takes all mail; probing more
    can't do better without the 7-credit catch_all check). Never iterates people.
    """
    log: list[dict] = []
    credits = 0
    mx = _mx_ok(domain)

    async def verify(email):
        nonlocal credits
        if not mx:
            return {"result": "invalid_nomx", "deliverable": False, "score": None, "credits_used": 0}
        v = await verify_email_mailin(email)
        credits += int(v.get("credits_used") or 0)
        log.append({"email": email, "result": v.get("result"), "score": v.get("score")})
        return v

    def ok(v):  # deliverable, or a catch-all we accept as-is
        return v.get("deliverable") is True or v.get("result") == "catch_all"

    def hit(email, person, v):
        conf = (0.9 if v.get("score") == "good" else 0.75) if person else 0.5
        return {"email": email, "person": person, "email_status": v.get("result"),
                "confidence": conf}, credits, log

    # Step 0 — FREE pattern: the highest-ranked person who has a name-matching
    # on-page email is already published; confirm with one verify.
    for person in people[:5]:
        match = next((e for e in page_emails if e.split("@")[0] not in _GENERIC_LP
                      and _local_matches_name(e.split("@")[0], person["name"])), None)
        if match:
            v = await verify(match)
            if ok(v):
                return hit(match, person, v)
            break  # a published personal address that won't verify => domain is
                   # verification-resistant; go straight to probe/bail, don't loop.

    # Step 1 — PROBE the single top person to learn the domain pattern.
    top = people[0] if people else None
    if top:
        for email in _patterns(top["name"], domain)[:max_patterns]:
            v = await verify(email)
            if ok(v):
                return hit(email, top, v)
        # Step 2 — all patterns invalid for a real senior person => the DOMAIN
        # rejects verification. Others share the pattern, so we do NOT try them.

    # Step 3 — generic fallback: two mailboxes, not seven.
    for lp in ("info", "contact"):
        v = await verify(f"{lp}@{domain}")
        if ok(v):
            return hit(f"{lp}@{domain}", None, v)
    return None, credits, log


# ---------- entry ----------
async def _find_async(lead: dict, cost_mode: str, cfg: CompanyFinderConfig) -> dict[str, Any]:
    nodes: list[str] = []
    company = lead.get("company") or lead.get("company_name")
    profile_url = lead.get("clutch_profile_url")
    # We never iterate PEOPLE (the pattern is per-domain; one probe suffices).
    # cost_mode only controls how many pattern hypotheses the probe tries before
    # bailing — each is 1 Mailin credit.
    max_patterns = 4 if cost_mode == "high" else 3

    def result(best, cost, evidence, status):
        return {"best_email": best, "status": status.value, "cost_usd": round(cost, 6),
                "nodes_executed": nodes, "evidence": evidence}

    if not cfg.openai_key or not cfg.serper_key:
        return result(None, 0.0, {"error": "missing SERPER/OPENAI key"}, LeadStatus.FAILED)

    async with httpx.AsyncClient(timeout=cfg.http_timeout_s) as client:
        nodes.append("resolve_domain")
        domain, dsrc = await _resolve_domain(client, cfg, profile_url, company)
        if not domain:
            return result(None, 0.0, {"stage": "domain", "note": dsrc}, LeadStatus.EMAIL_NOT_FOUND)

        nodes.append("find_people")
        people, page_emails, psrc = await _find_people(client, cfg, domain, company)

        nodes.append("verify_contact")
        best, credits, vlog = await _pick_contact(cfg, domain, people, page_emails, max_patterns)

    evidence = {"domain": domain, "domain_source": dsrc, "people_source": psrc,
                "people": [{"name": p["name"], "title": p["title"]} for p in people[:8]],
                "page_emails": sorted(page_emails)[:8], "mailin_credits": credits, "verify_log": vlog}
    # rough cost: gpt-4o-mini calls (~$0.0005) + Mailin credits (~$0.0007/credit est.)
    cost = 0.001 + credits * 0.0007
    if best:
        p = best.get("person") or {}
        best_email = {
            "email": best["email"], "confidence": best["confidence"],
            "person_name": p.get("name"), "person_title": p.get("title"),
            "seniority": p.get("title"), "email_status": best.get("email_status"),
        }
        return result(best_email, cost, evidence, LeadStatus.EMAIL_FOUND)
    return result(None, cost, evidence, LeadStatus.EMAIL_NOT_FOUND)


def find_company_contact(lead: dict, cost_mode: str = "low",
                         config: CompanyFinderConfig | None = None) -> dict[str, Any]:
    """lead keys: company (or company_name), clutch_profile_url, website(optional).
    Returns a finder-shaped state dict; best_email carries the chosen person +
    email_status. Sync wrapper (drives Crawl4AI + async I/O on its own loop), so
    the worker can run it in a thread like find_guest_email."""
    return asyncio.run(_find_async(lead, cost_mode, config or CompanyFinderConfig()))
