"""Unit tests for email finder adapters and helpers (no network)."""
from __future__ import annotations

from ai_agents.agents.email_finder.adapters import merge_candidate_dicts
from ai_agents.agents.email_finder.nodes.page_extract import (
    extract_emails_from_text,
    mailto_from_html,
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
    # placeholder domain (see test_placeholder_emails_rejected).
    t = "Reach us at hello@acmecorp.io or support@test.co.uk."
    assert "hello@acmecorp.io" in extract_emails_from_text(t)
    assert "support@test.co.uk" in extract_emails_from_text(t)


def test_mailto_from_html() -> None:
    html = '<a href="mailto:Jane@Site.com?subject=Hi">x</a>'
    assert mailto_from_html(html) == ["Jane@Site.com"]


def test_build_page_candidates_emails_and_fb() -> None:
    from ai_agents.agents.email_finder.nodes.page_extract import build_page_candidates

    html = (
        '<a href="mailto:hi@acmelaw.com">mail</a> also tony@beastmodecamping.com '
        '<a href="https://facebook.com/acmelaw">fb</a>'
    )
    cands, fb = build_page_candidates(html, "", "https://acmelaw.com", fb_already_known=False)
    emails = {c.email for c in cands}
    assert "hi@acmelaw.com" in emails
    assert "tony@beastmodecamping.com" in emails
    assert all(c.source == "website_scraper — https://acmelaw.com" for c in cands)
    assert fb == ["https://facebook.com/acmelaw"]


def test_build_page_candidates_skips_fb_when_known() -> None:
    from ai_agents.agents.email_finder.nodes.page_extract import build_page_candidates

    html = '<a href="https://facebook.com/acmelaw">fb</a> tony@beastmodecamping.com'
    _cands, fb = build_page_candidates(html, "", "https://acmelaw.com", fb_already_known=True)
    assert fb == []  # lead already has a FB link, don't harvest another


def test_parse_json_response_strips_fences() -> None:
    raw = '```json\n{"chosen_email": "a@b.co", "confidence": 0.9, "reason": "ok"}\n```'
    d = _parse_json_response(raw)
    assert d["chosen_email"] == "a@b.co"


def test_normalize_origin_adds_scheme() -> None:
    assert _normalize_origin("example.com") is not None
    assert _normalize_origin("example.com").startswith("https://")


def test_score_url_keywords() -> None:
    assert _score_url("https://x.com/contact") > _score_url("https://x.com/blog/foo")


def test_build_scrape_plan_excludes_homepage() -> None:
    # The homepage is harvested during discovery, so it must NOT be re-crawled
    # via the plan — even if it resurfaces as a discovered link.
    home = "https://site.com/"
    plan = _build_scrape_plan(
        home,
        ["https://site.com/contact", "https://site.com"],  # homepage in sitemap too
        ["https://site.com/about", "https://site.com/"],   # and in homepage links
        max_urls=5,
    )
    assert "https://site.com/contact" in plan
    assert "https://site.com/about" in plan
    # No trailing-slash variant of the homepage sneaks into the plan.
    assert not any(u.rstrip("/") == "https://site.com" for u in plan)


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
    # Flag off → never touches the enricher; with no website it goes to the cheap
    # website_guesser before Perplexity.
    assert route_after_canonical({"youtube_list": False, "lead": _YT_LEAD}) == "website_guesser"


def test_routing_after_enrich() -> None:
    # Enrichment discovered a website → crawl it
    assert route_after_youtube_enrich(
        {"status": LeadStatus.PENDING.value, "lead": {"website": "https://learnwithhasan.com/"}}
    ) == "discover_urls"
    # No website discovered → try the cheap website_guesser before Perplexity
    assert route_after_youtube_enrich(
        {"status": LeadStatus.PENDING.value, "lead": {"social_links": {"twitter": "https://x.com/h"}}}
    ) == "website_guesser"
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


# ── website_guesser node + routing ────────────────────────────────────────────

import ai_agents.agents.email_finder.nodes.website_guesser as wg
from ai_agents.agents.email_finder.graph import (
    _no_website_fallback,
    route_after_website_guess,
)
from ai_agents.agents.email_finder.io.contract_models import WebsiteGuessInput
from ai_agents.agents.email_finder.state import CanonicalLead, Identity


def _lead(**raw):
    return CanonicalLead(identity=Identity(host_name="Stan Phelps"), raw=raw)


def test_build_context_skips_source_noise_and_urls() -> None:
    inp = WebsiteGuessInput(
        lead=_lead(**{"Speaker Description": "CX keynote speaker", "Asset Value": "267",
                      "Img": "https://x/y.jpg", "_dedup": "k"}),
        trace_id="", source="e-speakers.com",
    )
    ctx = wg._build_context(inp, "Stan Phelps")
    assert "Speaker Description: CX keynote speaker" in ctx
    assert "e-speakers.com" not in ctx      # source must NOT leak in (it suppresses guesses)
    assert "Asset Value" not in ctx          # noise skipped
    assert "Img" not in ctx                  # url-valued / noise skipped


def _run_with_guess(monkeypatch, website, confidence):
    monkeypatch.setattr(wg, "_guess", lambda name, ctx, tid: (website, confidence, 0.0))
    out = wg.website_guesser_run(WebsiteGuessInput(lead=_lead(Bio="x"), trace_id="", source=None))
    return out.lead


def test_guesser_accepts_confident_nonsocial(monkeypatch) -> None:
    lead = _run_with_guess(monkeypatch, "https://www.stanphelps.com", 0.9)
    assert lead.website == "https://www.stanphelps.com"
    assert lead.raw.get("_website_source") == "llm_guess"


def test_guesser_rejects_low_confidence(monkeypatch) -> None:
    assert _run_with_guess(monkeypatch, "https://site.com", 0.5).website is None


def test_guesser_rejects_social(monkeypatch) -> None:
    assert _run_with_guess(monkeypatch, "https://www.linkedin.com/in/x", 0.95).website is None


def test_guesser_abstains_on_null(monkeypatch) -> None:
    assert _run_with_guess(monkeypatch, None, 0.0).website is None


def test_route_after_website_guess() -> None:
    assert route_after_website_guess({"lead": {"website": "https://s.com"}}) == "discover_urls"
    assert route_after_website_guess({"lead": {}}) == "perplexity_discovery"


def test_no_website_fallback_runs_guesser_once() -> None:
    # no website, guesser not yet run → guess first
    assert _no_website_fallback({"lead": {}, "nodes_executed": ["canonical_builder"]}) == "website_guesser"
    # guesser already ran → straight to perplexity (no loop)
    assert _no_website_fallback({"lead": {}, "nodes_executed": ["website_guesser"]}) == "perplexity_discovery"
    # already has a website → perplexity (shouldn't re-guess)
    assert _no_website_fallback({"lead": {"website": "https://s.com"}, "nodes_executed": []}) == "perplexity_discovery"


# ── derive website from email domain (canonical_builder) ──────────────────────

from ai_agents.agents.email_finder.nodes.canonical_builder import (
    _website_from_email,
    _build_from_llm_output,
)
from ai_agents.agents.email_finder.state import SourceType


def test_website_from_email_company_domain() -> None:
    assert _website_from_email("jane@acme.com") == "https://acme.com"
    # case-normalised, subdomain + multi-part TLD preserved
    assert _website_from_email("a.b+x@Sub.Acme.CO.UK") == "https://sub.acme.co.uk"


def test_website_from_email_skips_free_and_malformed() -> None:
    for e in ["jane@gmail.com", "x@outlook.com", "y@icloud.com", "z@proton.me",
              "notanemail", "@nodomain", "a@b", "", None]:
        assert _website_from_email(e) is None


def test_build_derives_website_from_email_when_missing() -> None:
    out = _build_from_llm_output(
        {"host_name": "Jane", "existing_email": "jane@acme.com"},  # no website
        SourceType.OTHER, {"Name": "Jane"},
    )
    assert out.website == "https://acme.com"
    assert out.raw.get("_website_source") == "email_domain"  # provenance tagged


def test_build_does_not_override_real_website() -> None:
    out = _build_from_llm_output(
        {"host_name": "Jane", "website": "https://real.com", "existing_email": "jane@acme.com"},
        SourceType.OTHER, {},
    )
    assert out.website == "https://real.com"
    assert out.raw.get("_website_source") is None


def test_build_skips_free_provider_email() -> None:
    out = _build_from_llm_output(
        {"host_name": "Jane", "existing_email": "jane@gmail.com"},
        SourceType.OTHER, {},
    )
    assert out.website is None


def test_routing_email_derived_website_validates_first() -> None:
    # derived website must NOT preempt validation of the existing email
    derived = {"existing_email": "jane@acme.com", "website": "https://acme.com",
               "raw": {"_website_source": "email_domain"}}
    assert route_after_canonical({"lead": derived}) == "validate_existing_email"
    # a real website alongside an existing email still crawls first (unchanged)
    real = {"existing_email": "jane@acme.com", "website": "https://acme.com", "raw": {}}
    assert route_after_canonical({"lead": real}) == "discover_urls"


# ── cost_mode flag + terminal fallback (control plane) ────────────────────────

from ai_agents.agents.email_finder.graph import (
    _terminal_fallback,
    _has_discovery_urls,
    route_after_bio_links,
)


def test_terminal_fallback_respects_cost_mode() -> None:
    assert _terminal_fallback({}) == "perplexity_discovery"                    # absent → high
    assert _terminal_fallback({"cost_mode": "high"}) == "perplexity_discovery"
    assert _terminal_fallback({"cost_mode": "low"}) == END                     # skip the paid node


def test_website_guess_low_cost_ends() -> None:
    # low-cost: no site guessed → END instead of paying for Perplexity
    assert route_after_website_guess({"lead": {}, "cost_mode": "low"}) == END
    assert route_after_website_guess({"lead": {}, "cost_mode": "high"}) == "perplexity_discovery"
    # a guessed site still crawls in both modes
    assert route_after_website_guess({"lead": {"website": "https://s.com"}, "cost_mode": "low"}) == "discover_urls"


# ── bio-link resolver routing (Locators) ──────────────────────────────────────


def test_has_discovery_urls() -> None:
    assert _has_discovery_urls({"discovery_urls": ["https://linktr.ee/x"]}) is True
    assert _has_discovery_urls({"discovery_urls": ["  "]}) is False
    assert _has_discovery_urls({"discovery_urls": []}) is False
    assert _has_discovery_urls({}) is False


def test_no_website_fallback_prefers_bio_links() -> None:
    lead = {"discovery_urls": ["https://linktr.ee/jane"]}
    # free bio-link resolution runs before the cheap guesser
    assert _no_website_fallback({"lead": lead, "nodes_executed": ["canonical_builder"]}) == "resolve_bio_links"
    # already resolved → fall through to the guesser
    assert _no_website_fallback({"lead": lead, "nodes_executed": ["resolve_bio_links"]}) == "website_guesser"
    # resolved + guessed, low-cost → END (no paid research)
    assert _no_website_fallback(
        {"lead": lead, "nodes_executed": ["resolve_bio_links", "website_guesser"], "cost_mode": "low"}
    ) == END


def test_route_after_bio_links() -> None:
    # resolved a website → crawl it
    assert route_after_bio_links({"lead": {"website": "https://jane.com/"}}) == "discover_urls"
    # only harvested aggregator-page emails → resolve them
    assert route_after_bio_links({"lead": {}, "website_scrape_candidates": [{"email": "a@b.com"}]}) == "resolve_best_email"
    # nothing found → continue the no-website chain (bio already ran → guesser)
    assert route_after_bio_links(
        {"lead": {"discovery_urls": ["https://linktr.ee/x"]}, "nodes_executed": ["resolve_bio_links"]}
    ) == "website_guesser"
    # a plaintext email was found → done
    assert route_after_bio_links({"status": LeadStatus.EMAIL_FOUND.value, "lead": {}}) == END


# ── link_resolver helpers (Locators / free hop-resolution) ────────────────────

from ai_agents.agents.email_finder.nodes.link_resolver import (
    _bucket,
    _clean_emails,
    _name_related,
    _name_tokens,
    _reg_domain,
    _root_url,
)


def test_reg_domain_dependency_free() -> None:
    assert _reg_domain("https://www.chloeting.com/program") == "chloeting.com"
    assert _reg_domain("https://foo.co.uk/x") == "foo.co.uk"
    assert _reg_domain("https://a.b.example.com") == "example.com"


def test_bucket_classifies_urls() -> None:
    assert _bucket("https://linktr.ee/jane") == "agg"
    assert _bucket("https://bit.ly/abc") == "short"
    assert _bucket("https://x.com/jane") == "social"
    assert _bucket("https://ncs.io") == "thirdparty"
    assert _bucket("https://amazon.com/dp/x") == "market"
    assert _bucket("https://i.ytimg.com/x.jpg") == "infra"
    assert _bucket("https://janedoe.com/") == "real"


def test_root_url_normalizes_to_root() -> None:
    assert _root_url("https://site.com/deep/page") == "https://site.com/"


def test_name_related_prioritizes_own_domain() -> None:
    toks = _name_tokens("Chloe Ting")
    assert _name_related("https://chloeting.com/", toks) is True
    assert _name_related("https://ncs.io/", toks) is False


def test_clean_emails_drops_vendor_and_junk() -> None:
    text = "reach jane@chloeting.com or noreply@sentry.io and a@schema.org plus x@example.com"
    got = _clean_emails(text)
    assert "jane@chloeting.com" in got
    assert all(j not in got for j in ["noreply@sentry.io", "a@schema.org", "x@example.com"])


# ── develop-branch additions re-applied after the ai-agent-v2 merge ──────────


def test_terminal_fallback_defaults_to_paid_research() -> None:
    from langgraph.graph import END as _END

    from ai_agents.agents.email_finder.graph import _terminal_fallback

    assert _terminal_fallback({}) == "perplexity_discovery"
    assert _terminal_fallback({"cost_mode": "high"}) == "perplexity_discovery"
    assert _terminal_fallback({"cost_mode": "low"}) == _END


def test_route_after_resolve_low_cost_ends_instead_of_escalating() -> None:
    from langgraph.graph import END as _END

    from ai_agents.agents.email_finder.graph import route_after_resolve

    found = {"status": LeadStatus.EMAIL_FOUND.value}
    assert route_after_resolve(found) == _END  # found: done in any mode
    # No candidates, no FB signal → terminal fallback, gated by cost_mode.
    assert route_after_resolve({"status": "pending", "cost_mode": "low"}) == _END
    assert route_after_resolve({"status": "pending"}) == "perplexity_discovery"


def test_placeholder_emails_rejected() -> None:
    from ai_agents.agents.email_finder.nodes.page_extract import (
        extract_emails_from_text as _extract,
    )

    text = "Subscribe: your@email.com. Docs: name@example.com. Real: tony@beastmodecamping.com"
    assert _extract(text) == ["tony@beastmodecamping.com"]


def test_placeholder_locals_match_exactly_not_substring() -> None:
    from ai_agents.agents.email_finder.nodes.email_utils import is_placeholder_email

    assert is_placeholder_email("your@email.com") is True
    assert is_placeholder_email("greatest@gmail.com") is False  # contains "test@"
    assert is_placeholder_email("brandname@gmail.com") is False  # contains "name@"
