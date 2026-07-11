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
    # example.com is deliberately NOT used here — it's rejected as a
    # placeholder domain now (see test_placeholder_emails_rejected).
    t = "Reach us at hello@acmecorp.io or support@test.co.uk."
    assert "hello@acmecorp.io" in _extract_emails_from_text(t)
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


# ── cost_mode routing (control plane) ────────────────────────────────────────


def test_terminal_fallback_defaults_to_paid_research() -> None:
    from langgraph.graph import END

    from ai_agents.agents.email_finder.graph import _terminal_fallback

    # Missing key and explicit "high" both preserve pre-flag behavior.
    assert _terminal_fallback({}) == "perplexity_discovery"
    assert _terminal_fallback({"cost_mode": "high"}) == "perplexity_discovery"
    assert _terminal_fallback({"cost_mode": "low"}) == END


def test_route_after_canonical_low_cost_ends_without_website() -> None:
    from langgraph.graph import END

    from ai_agents.agents.email_finder.graph import route_after_canonical

    no_website = {"lead": {"website": None}}
    assert route_after_canonical({**no_website, "cost_mode": "low"}) == END
    assert route_after_canonical({**no_website, "cost_mode": "high"}) == "perplexity_discovery"
    # A website means free methods still have work to do, regardless of mode.
    has_website = {"lead": {"website": "https://example.com"}, "cost_mode": "low"}
    assert route_after_canonical(has_website) == "discover_urls"


def test_route_after_resolve_low_cost_ends_instead_of_escalating() -> None:
    from langgraph.graph import END

    from ai_agents.agents.email_finder.graph import route_after_resolve
    from ai_agents.agents.email_finder.state import LeadStatus

    found = {"status": LeadStatus.EMAIL_FOUND.value}
    assert route_after_resolve(found) == END  # found: done in any mode
    assert route_after_resolve({"status": "pending", "cost_mode": "low"}) == END
    assert route_after_resolve({"status": "pending"}) == "perplexity_discovery"


def test_placeholder_emails_rejected() -> None:
    from ai_agents.agents.email_finder.nodes.crawl_page import _extract_emails_from_text

    text = "Subscribe: your@email.com. Docs: name@example.com. Real: tony@beastmodecamping.com"
    out = _extract_emails_from_text(text)
    assert out == ["tony@beastmodecamping.com"]
