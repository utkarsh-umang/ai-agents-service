"""
Enrich a lead from its YouTube channel "About" page.

Why this exists: YouTube gates a channel's *business email* behind a reCAPTCHA,
so it is NOT in the rendered HTML and can't be scraped directly. BUT the About
page reliably exposes the creator's external links — their own website and
social profiles — as `/redirect?...&q=<url>` links. Those are exactly what the
rest of the email-finder graph needs: a website to crawl and socials to feed
Perplexity.

So for a `youtube_list` run, when a lead has a YouTube URL but no website yet,
this node fetches the About page, extracts website + socials (and any plaintext
email that happens to be on the page), writes them onto the lead, and lets the
normal flow take over (discover_urls → crawl → resolve, then Perplexity).

Fetch strategy (see `fetch_youtube_about`): a free plain HTTP GET first — the
links live in the initial HTML, so no browser/proxy is needed from a residential
IP — and ScrapingBee (1 credit) only as a fallback when YouTube serves a
consent/bot wall (e.g. from a datacenter IP or under heavy rate-limiting). A
ScrapingBee plan is therefore optional; without a key the node simply uses the
free path and degrades to Perplexity if that's blocked. Nothing here ever raises.
"""
from __future__ import annotations

import re
from urllib.parse import unquote

import httpx

from ai_agents.agents.email_finder.io.contract_models import (
    YouTubeEnrichInput,
    YouTubeEnrichOutput,
)
from ai_agents.agents.email_finder.nodes.canonical_builder import (
    _detect_social_platform,
    _strip_tracking_params,
)
from ai_agents.agents.email_finder.nodes.cost_utils import SCRAPINGBEE_COST_USD
from ai_agents.agents.email_finder.nodes.email_utils import is_plus_addressed
from ai_agents.agents.email_finder.state import (
    CanonicalLead,
    EmailCandidate,
    LeadStatus,
    SocialLinks,
)
from ai_agents.core.config import SCRAPINGBEE_API_KEY, SCRAPINGBEE_BASE_URL
from ai_agents.core.llm import langfuse

# Confidence for a plaintext email actually printed on the channel About page.
_YT_EMAIL_CONFIDENCE = 0.9

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Redirect links on the About page look like /redirect?...&q=<url-encoded target>.
# YouTube embeds '&' as the literal escape sequence & inside its JSON blob,
# so we match `q=` anywhere and stop the capture at the next delimiter
# (&, ", ', backslash, or whitespace) — non-http matches are dropped downstream.
_REDIRECT_Q_RE = re.compile(r"q=([^&\"'\\\s]+)")

# Domains that are YouTube/Google infra, never a creator's own link.
_INFRA_DOMAINS = ("youtube.com", "youtu.be", "google.com", "gstatic.com", "ytimg.com", "schema.org")

# Browser-like headers + consent cookies so the free HTTP GET gets the real page
# (the creator's links live in the ytInitialData JSON in the initial HTML).
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_CONSENT_COOKIES = {"CONSENT": "YES+1", "SOCS": "CAI"}

# Markers that mean YouTube served a consent/bot wall instead of the real page —
# only THEN is it worth escalating to ScrapingBee's residential proxies.
_BLOCK_MARKERS = (
    "before you continue", "consent.youtube.com",
    "sign in to confirm you", "/sorry/index",
)

# Process-wide circuit breaker: once ScrapingBee reports out-of-credits / bad key,
# stop calling it for the rest of the run so we don't waste time or credits.
_scrapingbee_disabled = False


def _channel_about_url(youtube_url: str) -> str:
    """Normalise a channel URL to its About page."""
    url = youtube_url.strip().rstrip("/")
    if url.endswith("/about"):
        return url
    return f"{url}/about"


def _extract_external_links(html: str) -> list[str]:
    """Decode the creator's outbound links from About-page redirect URLs."""
    seen: set[str] = set()
    out: list[str] = []
    for enc in _REDIRECT_Q_RE.findall(html or ""):
        url = unquote(enc)
        if not url.startswith("http"):
            continue
        low = url.lower()
        if any(d in low for d in _INFRA_DOMAINS):
            continue
        clean = _strip_tracking_params(url)
        key = clean.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _extract_emails(html: str) -> list[str]:
    """Any plaintext email on the page (rare — most are reCAPTCHA-gated)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in _EMAIL_RE.findall(html or ""):
        email = raw.strip().strip(".").lower()
        if email in seen or is_plus_addressed(email):
            continue
        if any(d in email.split("@")[-1] for d in _INFRA_DOMAINS):
            continue
        seen.add(email)
        out.append(email)
    return out


def _parse_about_html(html: str) -> dict:
    return {"links": _extract_external_links(html), "emails": _extract_emails(html)}


def _looks_blocked(html: str) -> bool:
    low = (html or "").lower()
    return any(m in low for m in _BLOCK_MARKERS)


def _fetch_about_http(channel_url: str) -> dict | None:
    """
    Free path: a plain HTTP GET of the About page.

    The creator's links live in the ytInitialData JSON in the initial HTML, so no
    browser/JS is needed. Works from residential IPs. Returns the parsed dict, or
    None if the request failed or YouTube served a consent/bot wall (the only case
    where escalating to ScrapingBee's proxies is worthwhile).
    """
    about_url = _channel_about_url(channel_url)
    try:
        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            cookies=_CONSENT_COOKIES,
        ) as client:
            resp = client.get(about_url)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return None
    if _looks_blocked(html):
        return None
    return _parse_about_html(html)


def _fetch_about_scrapingbee(channel_url: str) -> dict:
    """
    Paid fallback (1 credit): ScrapingBee residential proxies, for when the free
    GET is blocked (e.g. running from a datacenter IP, or rate-limited at scale).

    render_js=false (1 credit, not 5) — the links are in the initial HTML.
    Raises httpx.HTTPStatusError on credit/auth errors so the caller can react.
    """
    about_url = _channel_about_url(channel_url)
    params = {"api_key": SCRAPINGBEE_API_KEY, "url": about_url, "render_js": "false"}
    with httpx.Client(timeout=90.0) as client:
        resp = client.get(SCRAPINGBEE_BASE_URL, params=params)
        resp.raise_for_status()
        html = resp.text
    result = _parse_about_html(html)
    result["paid"] = True  # a ScrapingBee credit was spent on this fetch
    return result


def fetch_youtube_about(channel_url: str) -> dict:
    """
    Get the channel About-page links/emails. Free path first, ScrapingBee only as
    a fallback — so a ScrapingBee plan is NOT required to use youtube_list.

      1. Plain HTTP GET (free)        — works from residential IPs.
      2. ScrapingBee (1 credit)       — ONLY if the free GET was blocked AND a key
                                        is set AND ScrapingBee hasn't already failed
                                        this run (circuit breaker on credits/auth).

    Never raises: always returns {"links": [...], "emails": [...]} (possibly empty),
    so a flagged lead always continues down the normal Perplexity/crawl path.
    """
    global _scrapingbee_disabled

    result = _fetch_about_http(channel_url)
    if result is not None:
        # Free path worked. Empty links just means the channel lists no website —
        # a legitimate result, not worth spending a ScrapingBee credit on.
        return result

    # Free GET was blocked/failed → escalate to ScrapingBee if it's available.
    if SCRAPINGBEE_API_KEY and not _scrapingbee_disabled:
        try:
            return _fetch_about_scrapingbee(channel_url)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (401, 402, 403):
                _scrapingbee_disabled = True  # out of credits / bad key → stop trying
                print(
                    f"  [youtube_about_enricher] ScrapingBee disabled for this run "
                    f"(HTTP {code} — credits/auth). Continuing on the Perplexity path."
                )
            # transient errors (429/5xx): just skip ScrapingBee for this lead
        except Exception:
            pass  # any other ScrapingBee failure → fall through gracefully

    return {"links": [], "emails": []}


def _classify_links(links: list[str]) -> tuple[str | None, SocialLinks, list[str]]:
    """Split creator links into (website, social_links, discovery_urls)."""
    website: str | None = None
    socials: dict[str, str] = {}
    discovery: list[str] = []
    for url in links:
        platform = _detect_social_platform(url)
        if platform:
            socials.setdefault(platform, url)
        elif website is None:
            website = url        # first non-social standalone URL = the website
        else:
            discovery.append(url)
    return website, SocialLinks(**socials), discovery


def _merge_socials(existing: SocialLinks, found: SocialLinks) -> SocialLinks:
    """Keep existing values; fill blanks from the About page."""
    data = existing.model_dump()
    for k, v in found.model_dump().items():
        if v and not data.get(k):
            data[k] = v
    return SocialLinks(**data)


def youtube_about_enricher_run(inp: YouTubeEnrichInput) -> YouTubeEnrichOutput:
    trace = langfuse.trace(id=inp.trace_id, name="email_finder", session_id="email_finder")
    span = trace.span(name="youtube_about_enricher")

    lead = inp.lead
    channel_url = (lead.social_links.youtube or "").strip()
    if not channel_url:
        raw = lead.raw or {}
        for key in ("Channel URL", "channel_url", "YouTube", "youtube"):
            val = (raw.get(key) or "").strip()
            if "youtube.com" in val.lower() or "youtu.be" in val.lower():
                channel_url = val
                break

    if not channel_url:
        span.end()
        trace.event(name="youtube_enrich_skipped", metadata={"reason": "no youtube url"})
        return YouTubeEnrichOutput(lead=lead, status=LeadStatus.PENDING, errors=[])

    try:
        about = fetch_youtube_about(channel_url)
        cost = SCRAPINGBEE_COST_USD if about.get("paid") else 0.0
        website, found_socials, discovery = _classify_links(about["links"])

        # Enrich the lead in place (don't clobber existing values).
        if not lead.website and website:
            lead.website = website
        lead.social_links = _merge_socials(lead.social_links, found_socials)
        if discovery:
            existing = set(lead.discovery_urls)
            lead.discovery_urls = lead.discovery_urls + [d for d in discovery if d not in existing]

        # Any plaintext email on the page is a strong candidate.
        candidates = [
            EmailCandidate(
                email=e,
                source="youtube_about — ScrapingBee",
                confidence=_YT_EMAIL_CONFIDENCE,
                note="email found on YouTube channel About page",
            )
            for e in about["emails"]
        ]

        span.end()
        trace.event(
            name="youtube_enrich_complete",
            metadata={
                "channel_url": channel_url,
                "website_found": bool(website),
                "socials_found": sum(1 for v in found_socials.model_dump().values() if v),
                "discovery_found": len(discovery),
                "emails_found": len(candidates),
            },
        )

        if candidates:
            return YouTubeEnrichOutput(
                lead=lead,
                email_candidates=candidates,
                best_email=candidates[0],
                status=LeadStatus.EMAIL_FOUND,
                errors=[],
                cost_usd=cost,
            )

        # No direct email — pass the enriched lead onward (PENDING = keep going).
        return YouTubeEnrichOutput(lead=lead, status=LeadStatus.PENDING, errors=[], cost_usd=cost)

    except Exception as e:
        error_msg = str(e)
        span.end()
        trace.event(name="youtube_enrich_failed", metadata={"error": error_msg})
        # Graceful: keep the original lead and let the graph continue to Perplexity.
        return YouTubeEnrichOutput(
            lead=lead,
            status=LeadStatus.PENDING,
            errors=[f"youtube_about_enricher: {error_msg}"],
        )


async def youtube_about_enricher_node_async(state: dict) -> dict:
    """LangGraph async node."""
    from ai_agents.agents.email_finder.adapters import youtube_enrich_input_from_state

    inp = youtube_enrich_input_from_state(state)
    out = youtube_about_enricher_run(inp)
    result: dict = {
        "lead": out.lead.model_dump(mode="json"),
        "status": out.status.value,
        "errors": out.errors,
        "nodes_executed": out.nodes_executed_delta,
        "cost_usd": out.cost_usd,
    }
    if out.email_candidates:
        result["email_candidates"] = [c.model_dump(mode="json") for c in out.email_candidates]
    if out.best_email:
        result["best_email"] = out.best_email.model_dump(mode="json")
    return result
