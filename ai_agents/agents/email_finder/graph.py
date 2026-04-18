from __future__ import annotations

import asyncio
import nest_asyncio
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from ai_agents.agents.email_finder.adapters import source_type_from_state
from ai_agents.agents.email_finder.graph_state import EmailFinderGraphState
from ai_agents.agents.email_finder.nodes.canonical_builder import (
    canonical_builder_to_graph_dict,
)
from ai_agents.agents.email_finder.nodes.crawl_page import crawl_page_node_async
from ai_agents.agents.email_finder.nodes.perplexity_discovery import (
    perplexity_discovery_node_async,
)
from ai_agents.agents.email_finder.nodes.resolve_best_email import (
    resolve_best_email_node_async,
)
from ai_agents.agents.email_finder.nodes.url_discovery import discover_urls_node_async
from ai_agents.agents.email_finder.state import LeadStatus, SourceType


# ── Node wrappers ────────────────────────────────────────────────────────────


def run_canonical_builder(state: EmailFinderGraphState) -> dict[str, Any]:
    st = source_type_from_state(state)
    return canonical_builder_to_graph_dict(state["raw_row"], st)


def route_after_canonical(state: EmailFinderGraphState) -> str:
    lead = state.get("lead") or {}
    w = lead.get("website")
    if w and str(w).strip():
        return "discover_urls"
    return "perplexity_discovery"


def route_after_discover(state: EmailFinderGraphState) -> str | list[Send]:
    plan = state.get("scrape_plan") or []
    if not plan:
        return "resolve_best_email"
    tid = state.get("trace_id") or ""
    lead = state["lead"]
    return [
        Send(
            "crawl_page",
            {
                "url": u,
                "lead": lead,
                "trace_id": tid,
                "page_timeout_ms": 60000,
            },
        )
        for u in plan
    ]


def route_after_resolve(state: EmailFinderGraphState) -> str:
    if state.get("status") == LeadStatus.EMAIL_FOUND.value:
        return END
    return "perplexity_discovery"


# ── Graph definition ─────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    graph = StateGraph(EmailFinderGraphState)

    graph.add_node("canonical_builder", run_canonical_builder)
    graph.add_node("discover_urls", discover_urls_node_async)
    graph.add_node("crawl_page", crawl_page_node_async)
    graph.add_node("resolve_best_email", resolve_best_email_node_async)
    graph.add_node("perplexity_discovery", perplexity_discovery_node_async)

    graph.set_entry_point("canonical_builder")

    graph.add_conditional_edges(
        "canonical_builder",
        route_after_canonical,
        {
            "discover_urls": "discover_urls",
            "perplexity_discovery": "perplexity_discovery",
        },
    )

    graph.add_conditional_edges(
        "discover_urls",
        route_after_discover,
        {
            "crawl_page": "crawl_page",
            "resolve_best_email": "resolve_best_email",
        },
    )

    graph.add_edge("crawl_page", "resolve_best_email")

    graph.add_conditional_edges(
        "resolve_best_email",
        route_after_resolve,
        {
            END: END,
            "perplexity_discovery": "perplexity_discovery",
        },
    )

    graph.add_edge("perplexity_discovery", END)

    return graph.compile()


# ── Single lead runner ────────────────────────────────────────────────────────


def run_single(raw_row: dict[str, Any], source_type: SourceType) -> dict[str, Any]:
    """
    Run the graph for a single lead.
    Returns the final state as a dict.
    """
    graph = build_graph()

    async def _run() -> dict[str, Any]:
        return await graph.ainvoke(
            {
                "raw_row": raw_row,
                "source_type": source_type.value
                if isinstance(source_type, SourceType)
                else source_type,
            }
        )

    try:
        return asyncio.run(_run())
    except RuntimeError:
        # Nested loop (e.g. Jupyter)
        nest_asyncio.apply()
        return asyncio.run(_run())


# ── Batch runner ──────────────────────────────────────────────────────────────


async def run_single_async(
    graph: Any,
    raw_row: dict[str, Any],
    source_type: SourceType,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run a single lead through the graph with semaphore rate limiting."""
    async with semaphore:
        st_val = source_type.value if isinstance(source_type, SourceType) else source_type
        return await graph.ainvoke({"raw_row": raw_row, "source_type": st_val})


async def run_batch_async(
    rows: list[dict[str, Any]],
    source_type: SourceType,
    concurrency: int = 3,
) -> list[dict[str, Any]]:
    """
    Run multiple leads concurrently.
    Lower default concurrency due to Playwright + Perplexity load.
    """
    graph = build_graph()
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        run_single_async(graph, row, source_type, semaphore)
        for row in rows
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            output.append({
                "status": LeadStatus.FAILED.value,
                "errors": [str(result)],
                "raw_row": rows[i],
            })
        else:
            output.append(result)

    return output


def run_batch(
    rows: list[dict[str, Any]],
    source_type: SourceType,
    concurrency: int = 3,
) -> list[dict[str, Any]]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            nest_asyncio.apply()
        return asyncio.run(run_batch_async(rows, source_type, concurrency))
    except RuntimeError:
        return asyncio.get_event_loop().run_until_complete(
            run_batch_async(rows, source_type, concurrency)
        )
