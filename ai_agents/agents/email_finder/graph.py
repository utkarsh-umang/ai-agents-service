from __future__ import annotations

import asyncio
from typing import Any
from langgraph.graph import StateGraph, END
from agents.email_finder.state import (
    EmailFinderState,
    LeadStatus,
    SourceType,
)
from agents.email_finder.nodes.canonical_builder import canonical_builder_node
from agents.email_finder.nodes.perplexity_discovery import perplexity_discovery_node


# ── Node wrappers ────────────────────────────────────────────────────────────
# LangGraph passes state into nodes, so we wrap our functions
# to match that signature

def run_canonical_builder(state: dict) -> dict:
    """
    Entry node — not a real state transition.
    Called before graph starts with raw input.
    """
    result = canonical_builder_node(
        raw_row=state["raw_row"],
        source_type=state["source_type"],
    )
    return result.model_dump()


def run_perplexity_discovery(state: dict) -> dict:
    email_finder_state = EmailFinderState(**state)
    result = perplexity_discovery_node(email_finder_state)
    return result.model_dump()


# ── Routing logic ────────────────────────────────────────────────────────────

def route_after_canonical_build(state: dict) -> str:
    """
    After canonical build decide what to do next.
    Right now always goes to perplexity.
    When scraper + social nodes exist, this is where you add routing logic.
    """
    return "perplexity_discovery"


def route_after_perplexity(state: dict) -> str:
    """
    After perplexity discovery decide next step.
    - EMAIL_FOUND → end, we have what we need
    - EMAIL_NOT_FOUND → end for now, scraper subgraph will plug in here in v2
    - FAILED → end, log the error
    """
    status = state.get("status")

    if status == LeadStatus.EMAIL_FOUND.value:
        return END

    if status == LeadStatus.EMAIL_NOT_FOUND.value:
        # v2: route to website scraper subgraph here
        return END

    if status == LeadStatus.FAILED.value:
        return END

    return END


# ── Graph definition ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(dict)

    # Nodes
    graph.add_node("canonical_builder", run_canonical_builder)
    graph.add_node("perplexity_discovery", run_perplexity_discovery)

    # Entry point
    graph.set_entry_point("canonical_builder")

    # Edges
    graph.add_conditional_edges(
        "canonical_builder",
        route_after_canonical_build,
        {
            "perplexity_discovery": "perplexity_discovery",
        },
    )

    graph.add_conditional_edges(
        "perplexity_discovery",
        route_after_perplexity,
        {
            END: END,
        },
    )

    return graph.compile()


# ── Single lead runner ────────────────────────────────────────────────────────

def run_single(raw_row: dict[str, Any], source_type: SourceType) -> dict:
    """
    Run the graph for a single lead.
    Returns the final state as a dict.
    """
    graph = build_graph()
    result = graph.invoke({
        "raw_row": raw_row,
        "source_type": source_type,
    })
    return result


# ── Batch runner ──────────────────────────────────────────────────────────────

async def run_single_async(
    graph: StateGraph,
    raw_row: dict[str, Any],
    source_type: SourceType,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single lead through the graph with semaphore rate limiting."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: graph.invoke({
                "raw_row": raw_row,
                "source_type": source_type,
            })
        )
        return result


async def run_batch_async(
    rows: list[dict[str, Any]],
    source_type: SourceType,
    concurrency: int = 5,
) -> list[dict]:
    """
    Run multiple leads concurrently.
    concurrency controls how many leads are processed in parallel.
    Increase carefully — Perplexity has rate limits.
    """
    graph = build_graph()
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        run_single_async(graph, row, source_type, semaphore)
        for row in rows
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successes from failures
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
    concurrency: int = 5,
) -> list[dict]:
    """
    Sync wrapper for batch runner.
    This is what you call from your main service.
    """
    return asyncio.run(run_batch_async(rows, source_type, concurrency))