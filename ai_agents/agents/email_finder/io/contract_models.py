"""
Pydantic input/output contracts per email-finder node.

Downstream nodes depend only on these types, not ad-hoc dict shapes.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ai_agents.agents.email_finder.state import (
    CanonicalLead,
    EmailCandidate,
    LeadStatus,
    SourceType,
)


class CanonicalBuilderInput(BaseModel):
    """Input for canonical_builder."""

    raw_row: dict[str, Any]
    source_type: SourceType


class CanonicalBuilderOutput(BaseModel):
    """Output from canonical_builder."""

    lead: CanonicalLead
    status: LeadStatus = LeadStatus.PENDING
    errors: list[str] = Field(default_factory=list)
    nodes_executed_delta: list[str] = Field(default_factory=lambda: ["canonical_builder"])
    trace_id: str = Field(description="Langfuse trace id for downstream spans")


class DiscoveryInput(BaseModel):
    """Input for url discovery (sitemap + homepage links)."""

    lead: CanonicalLead
    max_urls: int = Field(default=5, ge=1, le=20)
    trace_id: str


class DiscoveryMeta(BaseModel):
    """Structured discovery diagnostics for Langfuse and debugging."""

    sitemap_urls_found: int = 0
    homepage_links_found: int = 0
    strategies: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DiscoveryOutput(BaseModel):
    """Output from url discovery."""

    scrape_plan: list[str] = Field(default_factory=list)
    discovery_meta: DiscoveryMeta = Field(default_factory=DiscoveryMeta)
    errors: list[str] = Field(default_factory=list)
    nodes_executed_delta: list[str] = Field(default_factory=lambda: ["url_discovery"])
    # Emails/FB links harvested from the homepage HTML that discovery already
    # fetched for link extraction — so the homepage isn't crawled a second time
    # as the first fan-out page. Merged into the same reducers crawl_page feeds.
    homepage_candidates: list[EmailCandidate] = Field(default_factory=list)
    homepage_fb_links: list[str] = Field(default_factory=list)


class CrawlPageInput(BaseModel):
    """Input for a single parallel crawl worker (Send payload)."""

    url: str
    lead: CanonicalLead
    trace_id: str
    page_timeout_ms: int = Field(default=60000)


class CrawlPageOutput(BaseModel):
    """Output from one page crawl (merged via reducer)."""

    candidates: list[EmailCandidate] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    page_url: str = ""
    fb_links: list[str] = Field(default_factory=list)


class ResolveInput(BaseModel):
    """Input for resolve_best_email (website-sourced candidates only)."""

    lead: CanonicalLead
    website_candidates: list[EmailCandidate]
    trace_id: str


class ResolveOutput(BaseModel):
    """Output from resolver LLM."""

    best_email: Optional[EmailCandidate] = None
    status: LeadStatus = LeadStatus.EMAIL_NOT_FOUND
    errors: list[str] = Field(default_factory=list)
    nodes_executed_delta: list[str] = Field(default_factory=lambda: ["resolve_best_email"])
    # USD spent by this node's paid calls (feeds the graph's cost_usd reducer)
    cost_usd: float = 0.0


class PerplexityInput(BaseModel):
    """Input for perplexity_discovery."""

    lead: CanonicalLead
    trace_id: str
    prior_email_candidates: list[EmailCandidate] = Field(default_factory=list)
    # Provenance of the lead (e.g. "speakerhub.com", "youtube api tool") — used as
    # extra context to help identify/disambiguate the person.
    source: Optional[str] = None


class WebsiteGuessInput(BaseModel):
    """Input for website_guesser."""

    lead: CanonicalLead
    trace_id: str
    source: Optional[str] = None


class WebsiteGuessOutput(BaseModel):
    """
    Output from website_guesser.

    `lead` is returned (with `website` filled in when the model confidently knew
    it). `status` stays PENDING either way — the graph continues to the website
    crawl if a site was guessed, otherwise to Perplexity.
    """

    lead: CanonicalLead
    status: LeadStatus = LeadStatus.PENDING
    errors: list[str] = Field(default_factory=list)
    nodes_executed_delta: list[str] = Field(default_factory=lambda: ["website_guesser"])
    cost_usd: float = 0.0


class PerplexityOutput(BaseModel):
    """Output from perplexity_discovery."""

    email_candidates: list[EmailCandidate]
    best_email: Optional[EmailCandidate] = None
    status: LeadStatus
    errors: list[str] = Field(default_factory=list)
    nodes_executed_delta: list[str] = Field(default_factory=lambda: ["perplexity_discovery"])
    cost_usd: float = 0.0


class YouTubeEnrichInput(BaseModel):
    """Input for youtube_about_enricher (ScrapingBee channel About page)."""

    lead: CanonicalLead
    trace_id: str


class YouTubeEnrichOutput(BaseModel):
    """
    Output from youtube_about_enricher.

    `lead` is the enriched lead (website / social_links / discovery_urls filled
    from the channel About page). `status` is PENDING when only enrichment
    happened (graph continues), or EMAIL_FOUND if a plaintext email was on the page.
    """

    lead: CanonicalLead
    email_candidates: list[EmailCandidate] = Field(default_factory=list)
    best_email: Optional[EmailCandidate] = None
    status: LeadStatus = LeadStatus.PENDING
    errors: list[str] = Field(default_factory=list)
    nodes_executed_delta: list[str] = Field(
        default_factory=lambda: ["youtube_about_enricher"]
    )
    cost_usd: float = 0.0


class ValidateEmailInput(BaseModel):
    """Input for validate_existing_email."""

    lead: CanonicalLead
    trace_id: str


class ValidateEmailOutput(BaseModel):
    """Output from validate_existing_email."""

    best_email: Optional[EmailCandidate] = None
    status: LeadStatus = LeadStatus.EMAIL_NOT_FOUND
    errors: list[str] = Field(default_factory=list)
    nodes_executed_delta: list[str] = Field(
        default_factory=lambda: ["validate_existing_email"]
    )
    sub_threshold_candidate: Optional[EmailCandidate] = None


class FBCrawlerInput(BaseModel):
    """Input for fb_crawler node."""

    lead: CanonicalLead
    fb_link: str
    trace_id: str


class FBCrawlerOutput(BaseModel):
    """Output from fb_crawler node."""

    best_email: Optional[EmailCandidate] = None
    status: LeadStatus = LeadStatus.EMAIL_NOT_FOUND
    errors: list[str] = Field(default_factory=list)
    nodes_executed_delta: list[str] = Field(
        default_factory=lambda: ["fb_crawler"]
    )
