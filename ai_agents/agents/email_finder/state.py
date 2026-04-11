from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class SourceType(str, Enum):
    PODSCAN_HOST = "podscan_host"
    PODSCAN_GUEST = "podscan_guest"
    YOUTUBE_SCRIPT_TOOL = "youtube_script_tool"
    OTHER = "other"


class LeadStatus(str, Enum):
    PENDING = "pending"
    DISCOVERING = "discovering"
    EMAIL_FOUND = "email_found"
    EMAIL_NOT_FOUND = "email_not_found"
    FAILED = "failed"


class SocialLinks(BaseModel):
    facebook: Optional[str] = None
    twitter: Optional[str] = None
    instagram: Optional[str] = None
    youtube: Optional[str] = None
    linkedin: Optional[str] = None


class Identity(BaseModel):
    host_name: Optional[str] = None
    podcast_name: Optional[str] = None
    channel_name: Optional[str] = None
    brand_name: Optional[str] = None

    def best_name(self) -> Optional[str]:
        """Returns the most specific name available."""
        return (
            self.host_name
            or self.podcast_name
            or self.channel_name
            or self.brand_name
        )

    def context_name(self) -> Optional[str]:
        """Returns show/brand context to support the best name."""
        if self.host_name:
            return self.podcast_name or self.channel_name or self.brand_name
        return None


class EmailCandidate(BaseModel):
    email: str
    source: str                    # e.g. "perplexity", "website_scraper"
    confidence: float = Field(ge=0.0, le=1.0)
    note: Optional[str] = None    # e.g. "found on contact page"


class CanonicalLead(BaseModel):
    identity: Identity
    website: Optional[str] = None
    # Non-social, non-standalone URLs (linktr.ee, beacons.ai, carrd.co, etc.)
    # that are useful as discovery hints for Perplexity but aren't the host's own domain
    discovery_urls: list[str] = Field(default_factory=list)
    existing_email: Optional[str] = None
    social_links: SocialLinks = Field(default_factory=SocialLinks)
    source_type: SourceType = SourceType.OTHER
    raw: dict = Field(default_factory=dict)


class EmailFinderState(BaseModel):
    """
    The full graph state passed between nodes.
    Lead is the canonical input.
    Everything else is written by nodes as they run.
    """
    lead: CanonicalLead

    # Written by nodes
    status: LeadStatus = LeadStatus.PENDING
    email_candidates: list[EmailCandidate] = Field(default_factory=list)
    best_email: Optional[EmailCandidate] = None

    # Tracking
    errors: list[str] = Field(default_factory=list)
    nodes_executed: list[str] = Field(default_factory=list)