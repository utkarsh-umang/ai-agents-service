"""Unit tests for email finder adapters and helpers (no network)."""
from __future__ import annotations

from ai_agents.agents.email_finder.adapters import merge_candidate_dicts
from ai_agents.agents.email_finder.nodes.crawl_page import (
    _extract_emails_from_text,
    _mailto_from_html,
)
from ai_agents.agents.email_finder.nodes.resolve_best_email import _parse_json_response
from ai_agents.agents.email_finder.nodes.url_discovery import (
    _build_scrape_plan,
    _normalize_origin,
    _score_url,
)


def test_merge_candidate_dicts_dedupes_by_email() -> None:
    a = [{"email": "a@x.com", "source": "s1"}]
    b = [{"email": "A@x.com", "source": "s2"}, {"email": "b@y.com", "source": "s3"}]
    m = merge_candidate_dicts(a, b)
    assert len(m) == 2
    assert {x["email"].lower() for x in m} == {"a@x.com", "b@y.com"}


def test_extract_emails_basic() -> None:
    t = "Reach us at hello@example.com or support@test.co.uk."
    assert "hello@example.com" in _extract_emails_from_text(t)
    assert "support@test.co.uk" in _extract_emails_from_text(t)


def test_mailto_from_html() -> None:
    html = '<a href="mailto:Jane@Site.com?subject=Hi">x</a>'
    assert _mailto_from_html(html) == ["Jane@Site.com"]


def test_parse_json_response_strips_fences() -> None:
    raw = '```json\n{"chosen_email": "a@b.co", "confidence": 0.9, "reason": "ok"}\n```'
    d = _parse_json_response(raw)
    assert d["chosen_email"] == "a@b.co"


def test_normalize_origin_adds_scheme() -> None:
    assert _normalize_origin("example.com") is not None
    assert _normalize_origin("example.com").startswith("https://")


def test_score_url_keywords() -> None:
    assert _score_url("https://x.com/contact") > _score_url("https://x.com/blog/foo")


def test_build_scrape_plan_home_first() -> None:
    home = "https://site.com/"
    plan = _build_scrape_plan(
        home,
        ["https://site.com/contact"],
        ["https://site.com/about"],
        max_urls=3,
    )
    assert plan[0] == home
    assert len(plan) == 3


# ── youtube_about_enricher node + routing ─────────────────────────────────────

from ai_agents.agents.email_finder.graph import (
    _youtube_enabled,
    route_after_canonical,
    route_after_youtube_enrich,
)
from ai_agents.agents.email_finder.nodes.youtube_about_enricher import (
    _channel_about_url,
    _classify_links,
    _extract_emails,
    _extract_external_links,
)
from ai_agents.agents.email_finder.state import LeadStatus
from langgraph.graph import END

_YT_LEAD = {"social_links": {"youtube": "https://youtube.com/channel/UCabc"}}
_YT_LEAD_WEB = {**_YT_LEAD, "website": "https://brand.com"}


def test_channel_about_url_appends_about() -> None:
    assert _channel_about_url("https://youtube.com/channel/UCabc") == (
        "https://youtube.com/channel/UCabc/about"
    )
    # Idempotent if already an about page
    assert _channel_about_url("https://youtube.com/@h/about/") == (
        "https://youtube.com/@h/about"
    )


def test_extract_external_links_decodes_redirect_and_drops_infra() -> None:
    # YouTube embeds '&' as the literal escape & in its JSON blob — the
    # website link below uses that form to lock in the regex fix for it.
    html = (
        r'/redirect?event=channel_header&redir_token=AB&q=https%3A%2F%2Flearnwithhasan.com%2F" '
        'href="/redirect?event=channel_header&q=https%3A%2F%2Fx.com%2Fhasan_ab_hasan" '
        'href="/redirect?q=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3Dx" '  # infra → dropped
        'src="https://i.ytimg.com/vi/x/hq720.jpg?q=tbn"'                       # infra → dropped
    )
    links = _extract_external_links(html)
    assert "https://learnwithhasan.com/" in links
    assert "https://x.com/hasan_ab_hasan" in links
    assert all("youtube.com" not in u and "ytimg" not in u for u in links)


def test_classify_links_splits_website_and_socials() -> None:
    website, socials, discovery = _classify_links(
        ["https://learnwithhasan.com/", "https://x.com/hasan_ab_hasan"]
    )
    assert website == "https://learnwithhasan.com/"
    assert socials.twitter == "https://x.com/hasan_ab_hasan"
    assert discovery == []


def test_extract_emails_filters_plus_and_infra() -> None:
    blob = "biz@channel.com info+tag@channel.com a@schema.org"
    assert _extract_emails(blob) == ["biz@channel.com"]


def test_youtube_enabled_gate() -> None:
    assert _youtube_enabled({"youtube_list": True, "lead": _YT_LEAD}) is True
    assert _youtube_enabled({"youtube_list": False, "lead": _YT_LEAD}) is False
    # Flag on but no youtube url anywhere → disabled
    assert _youtube_enabled({"youtube_list": True, "lead": {"social_links": {}, "raw": {}}}) is False


def test_routing_website_first_then_enrich() -> None:
    # Website present → crawl it first (user-chosen ordering); enricher skipped
    assert route_after_canonical({"youtube_list": True, "lead": _YT_LEAD_WEB}) == "discover_urls"
    # No website → enrich from the About page to discover one
    assert route_after_canonical({"youtube_list": True, "lead": _YT_LEAD}) == "youtube_about_enricher"
    # Flag off → never touches the enricher
    assert route_after_canonical({"youtube_list": False, "lead": _YT_LEAD}) == "perplexity_discovery"


def test_routing_after_enrich() -> None:
    # Enrichment discovered a website → crawl it
    assert route_after_youtube_enrich(
        {"status": LeadStatus.PENDING.value, "lead": {"website": "https://learnwithhasan.com/"}}
    ) == "discover_urls"
    # No website discovered → hand off to Perplexity with enriched socials
    assert route_after_youtube_enrich(
        {"status": LeadStatus.PENDING.value, "lead": {"social_links": {"twitter": "https://x.com/h"}}}
    ) == "perplexity_discovery"
    # A plaintext email was on the About page → done
    assert route_after_youtube_enrich({"status": LeadStatus.EMAIL_FOUND.value, "lead": {}}) == END


# ── fetch fallback chain: free HTTP GET first, ScrapingBee optional, never raises ──

import httpx
import ai_agents.agents.email_finder.nodes.youtube_about_enricher as yt_node

_CH = "https://youtube.com/channel/UCabc"


def test_looks_blocked() -> None:
    assert yt_node._looks_blocked("... Before you continue to YouTube ...")
    assert not yt_node._looks_blocked("<html>normal page learnwithhasan.com</html>")


def test_free_path_used_first_scrapingbee_not_called(monkeypatch) -> None:
    monkeypatch.setattr(yt_node, "_fetch_about_http", lambda u: {"links": ["https://site.com/"], "emails": []})
    called = {"sb": False}
    def sb(u):
        called["sb"] = True
        return {"links": [], "emails": []}
    monkeypatch.setattr(yt_node, "_fetch_about_scrapingbee", sb)
    out = yt_node.fetch_youtube_about(_CH)
    assert out["links"] == ["https://site.com/"]
    assert called["sb"] is False  # free path worked → no ScrapingBee credit spent


def test_falls_back_to_scrapingbee_only_when_blocked(monkeypatch) -> None:
    yt_node._scrapingbee_disabled = False
    monkeypatch.setattr(yt_node, "SCRAPINGBEE_API_KEY", "key")
    monkeypatch.setattr(yt_node, "_fetch_about_http", lambda u: None)  # blocked
    monkeypatch.setattr(yt_node, "_fetch_about_scrapingbee", lambda u: {"links": ["https://sb.com/"], "emails": []})
    assert yt_node.fetch_youtube_about(_CH)["links"] == ["https://sb.com/"]


def test_credit_error_trips_circuit_breaker_and_degrades(monkeypatch) -> None:
    yt_node._scrapingbee_disabled = False
    monkeypatch.setattr(yt_node, "SCRAPINGBEE_API_KEY", "key")
    monkeypatch.setattr(yt_node, "_fetch_about_http", lambda u: None)
    def boom(u):
        req = httpx.Request("GET", "https://app.scrapingbee.com/api/v1/")
        raise httpx.HTTPStatusError("payment required", request=req, response=httpx.Response(402, request=req))
    monkeypatch.setattr(yt_node, "_fetch_about_scrapingbee", boom)
    out = yt_node.fetch_youtube_about(_CH)
    assert out == {"links": [], "emails": []}     # graceful → empty, lead continues
    assert yt_node._scrapingbee_disabled is True  # breaker tripped → no more SB calls this run
    yt_node._scrapingbee_disabled = False         # reset for isolation


def test_no_key_uses_free_path_only(monkeypatch) -> None:
    monkeypatch.setattr(yt_node, "SCRAPINGBEE_API_KEY", "")
    monkeypatch.setattr(yt_node, "_fetch_about_http", lambda u: None)  # blocked, no key to fall back to
    assert yt_node.fetch_youtube_about(_CH) == {"links": [], "emails": []}
