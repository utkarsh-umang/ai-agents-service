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
