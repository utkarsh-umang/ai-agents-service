"""
LangGraph TypedDict state with reducers for parallel crawl_fan-out.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired

from typing_extensions import TypedDict


class EmailFinderGraphState(TypedDict, total=False):
    """Shared LangGraph state. Optional keys use NotRequired where needed."""

    # Invocation (canonical_builder consumes these)
    raw_row: dict[str, Any]
    source_type: str

    # When True, the ScrapingBee youtube_about_enricher node is enabled for this
    # run (used for lists known to be YouTube channels). Gated further on the
    # lead actually having a YouTube URL. Defaults to False when absent.
    youtube_list: NotRequired[bool]

    # Propagated observability
    trace_id: NotRequired[str]

    # Canonical lead (serialized CanonicalLead)
    lead: dict[str, Any]

    # Pipeline status + outputs
    status: str
    email_candidates: list[dict[str, Any]]
    best_email: NotRequired[dict[str, Any] | None]

    # Discovery / scrape
    scrape_plan: list[str]
    discovery_meta: NotRequired[dict[str, Any]]

    # Reducer: parallel crawl_page workers append EmailCandidate dicts
    website_scrape_candidates: Annotated[list[dict[str, Any]], operator.add]

    # Reducer: parallel crawl_page workers append FB links found in page HTML
    scraped_fb_links: Annotated[list[str], operator.add]

    errors: Annotated[list[str], operator.add]
    nodes_executed: Annotated[list[str], operator.add]

    # crawl_page Send payload (NotRequired — only present in worker arg)
    url: NotRequired[str]
    page_timeout_ms: NotRequired[int]