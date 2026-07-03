"""
Link-in-bio / shortener hop-resolution (the "free approach" ported from the
lite notebook).

Job: turn the ambiguous `discovery_urls` on a lead (linktr.ee, beacons.ai,
bit.ly, …) into a real crawlable website — plus any emails printed directly on
the aggregator page — using only free HTTP requests. This fixes the biggest miss
bucket ("no website, but a link-in-bio was present") without paying for research.

Belongs to the Locators subsystem: it *produces targets*. It never scores emails
or decides routing — it hands a website + candidate emails back to the graph.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from ai_agents.agents.email_finder.nodes.canonical_builder import _detect_social_platform

# Link-in-bio aggregators — fetch them, harvest their outbound links + page emails.
_AGG_HOSTS = {
    "linktr.ee", "beacons.ai", "lnk.bio", "linkin.bio", "bio.link", "allmylinks.com",
    "campsite.bio", "solo.to", "tap.bio", "koji.to", "withkoji.com", "msha.ke",
    "flowpage.com", "snipfeed.co", "link.space", "hoo.be", "komi.io", "shor.by",
    "liinks.co", "carrd.co", "pillar.io", "glnk.io", "znap.link",
}

# URL shorteners / affiliate redirectors — resolve to the final URL.
_SHORT_HOSTS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly", "rb.gy", "cutt.ly",
    "shorturl.at", "rebrand.ly", "is.gd", "tiny.cc", "sjv.io", "pxf.io", "go.magik.ly",
    "lnk.to",
}

# Marketplaces / landing hosts that never carry the creator's own contact.
_MARKET_HOSTS = {
    "amazon.com", "amzn.to", "apps.apple.com", "play.google.com", "etsy.com",
    "patreon.com", "geni.us", "fanlink.to", "ffm.to", "smarturl.it", "spotify.com",
    "itunes.apple.com",
}

# Third-party hosts a creator *links to* (labels, distributors, merch, sponsors,
# agencies) — their email is the platform's, never the creator's.
_THIRDPARTY_HOSTS = {
    "ncs.io", "distrokid.com", "unitedmasters.com", "tunecore.com", "cdbaby.com",
    "monstercat.com", "soundcloud.com", "bandcamp.com", "awal.com",
    "ko-fi.com", "buymeacoffee.com", "cameo.com", "teespring.com", "spring.com",
    "bonfire.com", "streamelements.com", "streamlabs.com", "payhip.com",
    "redbubble.com", "teepublic.com", "spreadshirt.com", "gumroad.com",
    "caa.com", "unitedtalent.com", "wmeagency.com",
    "surfshark.com", "nordvpn.com", "expressvpn.com", "betterhelp.com",
    "epidemicsound.com",
}

# Infra / CDN hosts — never a real destination.
_INFRA_HOSTS = (
    "ytimg.com", "googlevideo.com", "youtube.com", "youtu.be", "google.com",
    "gstatic.com", "ggpht.com", "googleusercontent.com", "googleapis.com",
    "fbcdn.net", "cloudflare.com", "gravatar.com", "w3.org", "schema.org",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_BAD_EMAIL_SUBSTR = (
    "example.com", "example.org", "sentry", "wixpress", "godaddy", "schema.org",
    "w3.org", "yourdomain", "yourname", "domain.com", "placeholder", "squarespace",
    "cloudflare", "noreply", "no-reply", "donotreply",
)
_IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
_JUNK_DOMAINS = {
    "sentry.io", "wixpress.com", "wix.com", "schema.org", "w3.org", "example.com",
    "cloudflare.com", "godaddy.com", "squarespace.com", "gstatic.com", "googleapis.com",
}

# Registered-domain suffixes that are two labels (so we keep three).
_TWO_LABEL_TLDS = {
    "co.uk", "com.au", "co.in", "co.nz", "org.uk", "ac.uk", "co.za", "com.br",
    "co.jp", "com.sg", "co.kr",
}


def _host(u: str) -> str:
    try:
        h = urlparse(u if "://" in u else "https://" + u).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def _reg_domain(u: str) -> str:
    """Registered domain, dependency-free (no tldextract)."""
    h = _host(u)
    parts = h.split(".")
    if len(parts) <= 2:
        return h
    if ".".join(parts[-2:]) in _TWO_LABEL_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _is_infra(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in _INFRA_HOSTS)


def _bucket(u: str) -> str:
    """Classify a URL so we know whether it's a real destination, an aggregator to
    mine, a shortener to resolve, or noise to drop."""
    h = _host(u)
    rd = _reg_domain(u)
    if not h:
        return "bad"
    if _detect_social_platform(u):
        return "social"
    if rd in _AGG_HOSTS or h in _AGG_HOSTS:
        return "agg"
    if rd in _SHORT_HOSTS or h in _SHORT_HOSTS:
        return "short"
    if rd in _THIRDPARTY_HOSTS or h in _THIRDPARTY_HOSTS:
        return "thirdparty"
    if rd in _MARKET_HOSTS or h in _MARKET_HOSTS:
        return "market"
    if _is_infra(h):
        return "infra"
    return "real"


def _root_url(u: str) -> str:
    """Normalize to the site root so we scrape the homepage, not a deep page."""
    try:
        p = urlparse(u if "://" in u else "https://" + u)
        return f"{p.scheme}://{p.netloc}/" if p.netloc else u
    except Exception:
        return u


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if len(t) >= 3]


def _name_related(u: str, toks: list[str]) -> bool:
    rd = _reg_domain(u)
    return any(t in rd for t in toks)


def _urls_in_html(html: str) -> list[str]:
    """All http(s) URLs in the HTML (incl. JSON blobs) — used to mine agg pages."""
    h = (html or "").replace("\\u002f", "/").replace("\\/", "/")
    for ch in ('"', "'", "<", ">", "\\", "(", ")"):
        h = h.replace(ch, " ")
    return [t.rstrip(".,;") for t in h.split() if t.startswith("http")]


def _clean_emails(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _EMAIL_RE.findall(text or ""):
        e = m.strip().strip(".").lower()
        local, _, dom = e.partition("@")
        if not dom or "." not in dom or len(local) < 2:
            continue
        if any(e.endswith(ext) for ext in _IMG_EXT):
            continue
        if any(b in e for b in _BAD_EMAIL_SUBSTR):
            continue
        if dom in _JUNK_DOMAINS or _reg_domain("https://" + dom) in _JUNK_DOMAINS:
            continue
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


async def expand_candidates(
    links: list[str],
    client: httpx.AsyncClient,
    name: str = "",
    max_shorts: int = 6,
    max_aggs: int = 3,
    max_targets: int = 4,
) -> tuple[list[str], list[str]]:
    """
    Turn raw bio/discovery links into real website targets + any emails found on
    aggregator pages. Resolves shortener redirects, mines link-in-bio pages, drops
    third-party/social/infra hosts, normalizes to root, and prioritizes domains
    that echo the lead's name.

    Returns (targets, aggregator_emails).
    """
    toks = _name_tokens(name)
    real: list[str] = []
    aggs: list[str] = []
    shorts: list[str] = []
    agg_emails: list[str] = []

    for u in links:
        b = _bucket(u)
        if b == "real":
            real.append(u)
        elif b == "agg":
            aggs.append(u)
        elif b == "short":
            shorts.append(u)

    # Resolve shortener redirects to their final URL, then re-bucket.
    for s in shorts[:max_shorts]:
        try:
            resp = await client.get(s)
            fu = str(resp.url)
            b = _bucket(fu)
            if b == "real":
                real.append(fu)
            elif b == "agg":
                aggs.append(fu)
        except Exception:
            pass

    # Mine aggregator pages for outbound real links + inline emails.
    for a in aggs[:max_aggs]:
        try:
            resp = await client.get(a)
            html = resp.text
            agg_emails += _clean_emails(html)
            real += [u for u in _urls_in_html(html) if _bucket(u) == "real"]
        except Exception:
            pass

    # Dedup targets by registered domain, root-normalize, name-match first.
    seen_dom: set[str] = set()
    targets: list[str] = []
    for u in real:
        d = _reg_domain(u)
        if d and d not in seen_dom:
            seen_dom.add(d)
            targets.append(_root_url(u))
    targets.sort(key=lambda u: 0 if _name_related(u, toks) else 1)

    # Dedup emails preserving order.
    em_seen: set[str] = set()
    emails: list[str] = []
    for e in agg_emails:
        if e not in em_seen:
            em_seen.add(e)
            emails.append(e)

    return targets[:max_targets], emails
