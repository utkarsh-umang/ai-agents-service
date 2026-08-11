"""Shared page → email/FB-link extraction.

Factored out of crawl_page so url_discovery can harvest emails from the homepage
HTML it *already* fetches for link discovery — instead of the homepage being
crawled a second time as the first fan-out page. One extraction implementation,
two call sites, identical candidate shape.
"""
from __future__ import annotations

import re

from ai_agents.agents.email_finder.nodes.email_utils import (
    confidence_for_email,
    enrich_confidence,
    filter_emails,
)
from ai_agents.agents.email_finder.state import EmailCandidate

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


def extract_emails_from_text(text: str) -> list[str]:
    if not text:
        return []
    found = set()
    for m in _EMAIL_RE.findall(text):
        e = m.strip().rstrip(".,);]")
        if "@" in e and "." in e.split("@")[-1]:
            found.add(e.lower())
    return filter_emails(list(found))


def mailto_from_html(html: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r'mailto:([^"\'>\s?]+)', html or "", re.I):
        addr = m.group(1).split("?")[0].strip()
        if "@" in addr:
            out.append(addr)
    return filter_emails(out)


def extract_fb_links(html: str) -> list[str]:
    if not html:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _FB_HREF_RE.finditer(html):
        url = m.group(1)
        url_lower = url.lower()
        if any(fragment in url_lower for fragment in _FB_GENERIC_FRAGMENTS):
            continue
        if url_lower not in seen:
            seen.add(url_lower)
            out.append(url)
    return out


def build_page_candidates(
    html: str,
    md: str,
    url: str,
    fb_already_known: bool,
) -> tuple[list[EmailCandidate], list[str]]:
    """Turn one page's HTML/markdown into email candidates + FB links.

    fb_already_known: True when the lead already carries a Facebook social link,
    so we skip harvesting FB links off the page (mirrors crawl_page's old rule).
    """
    blob = f"{html}\n{md}"

    emails = set(extract_emails_from_text(blob))
    for m in mailto_from_html(html):
        emails.add(m.lower())

    candidates: list[EmailCandidate] = []
    for e in sorted(emails):
        base_confidence = confidence_for_email(e)
        enriched_confidence, note = enrich_confidence(e, url, blob, base_confidence)
        candidates.append(
            EmailCandidate(
                email=e,
                source=f"website_scraper — {url}",
                confidence=enriched_confidence,
                note=note,
            )
        )

    fb_links = [] if fb_already_known else extract_fb_links(html)
    return candidates, fb_links
