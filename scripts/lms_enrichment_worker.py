"""Pull worker: drains LMS's enrichment queue through the email finder.

LMS owns the ledger (enrichment_attempts) and derives the queue from it;
this worker just pulls a page of leads, runs the finder at the requested
cost tier, and posts one result per lead back. Crash-safe by design: a
lead only leaves the queue when its result row lands, so a killed worker
re-does at most one in-flight page, and nothing is ever attempted twice
at the same tier.

Usage:
    python scripts/lms_enrichment_worker.py                 # one page, low cost
    python scripts/lms_enrichment_worker.py --loop          # poll until queue empty
    python scripts/lms_enrichment_worker.py --cost-mode high --limit 5

Env: LMS_API_URL (default http://localhost:8000)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_agents.agents.email_finder.graph import build_graph, run_single_async  # noqa: E402
from ai_agents.agents.email_finder.state import LeadStatus, SourceType  # noqa: E402

LMS_API = os.environ.get("LMS_API_URL", "http://localhost:8000").rstrip("/")
PROVIDER = "email_finder@ai-agents-service"


def _queue_item_to_raw_row(item: dict) -> dict:
    """LMS canonical fields → the raw_row shape the finder's ingestion
    (LLM-classified) consumes. Keys are human-readable on purpose — the
    canonical_builder prompt reads them as a table row."""
    return {
        "channel_name": item.get("youtube_channel_name"),
        "youtube_handle": item.get("youtube_handle"),
        "website": item.get("website"),
        "youtube_url": item.get("social_youtube"),
        "twitter_url": item.get("social_twitter"),
        "instagram_url": item.get("social_instagram"),
        "tiktok_url": item.get("social_tiktok"),
        "facebook_url": item.get("social_facebook"),
        "linkedin_url": item.get("social_linkedin"),
        "niche": item.get("niche"),
        "country": item.get("country"),
    }


def _result_from_final_state(item: dict, state: dict, cost_mode: str) -> dict:
    best = state.get("best_email") or {}
    email = best.get("email") if isinstance(best, dict) else None
    found = state.get("status") == LeadStatus.EMAIL_FOUND.value and bool(email)
    failed = state.get("status") == LeadStatus.FAILED.value
    return {
        "lead_id": item["lead_id"],
        "type": "email",
        "cost_mode": cost_mode,
        "status": "found" if found else ("failed" if failed else "not_found"),
        "value": email if found else None,
        "confidence": best.get("confidence") if found else None,
        "provider": PROVIDER,
    }


async def process_page(client: httpx.AsyncClient, cost_mode: str, limit: int, concurrency: int) -> int:
    """Pull one page, run it, post results. Returns how many leads were processed."""
    resp = await client.get(
        f"{LMS_API}/api/v1/enrichment/queue",
        params={"cost_mode": cost_mode, "limit": limit},
    )
    resp.raise_for_status()
    items = resp.json()
    if not items:
        return 0

    print(f"[worker] pulled {len(items)} leads (cost_mode={cost_mode})")
    graph = build_graph()
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(item: dict) -> tuple[dict, dict | Exception]:
        try:
            state = await run_single_async(
                graph,
                _queue_item_to_raw_row(item),
                SourceType.YOUTUBE_SCRIPT_TOOL,
                semaphore,
                cost_mode=cost_mode,
            )
            return item, state
        except Exception as exc:  # noqa: BLE001 — one bad lead must not sink the page
            return item, exc

    results = await asyncio.gather(*(run_one(i) for i in items))

    for item, outcome in results:
        if isinstance(outcome, Exception):
            payload = {
                "lead_id": item["lead_id"],
                "type": "email",
                "cost_mode": cost_mode,
                "status": "failed",
                "provider": PROVIDER,
            }
        else:
            payload = _result_from_final_state(item, outcome, cost_mode)
        post = await client.post(f"{LMS_API}/api/v1/enrichment/results", json=payload)
        post.raise_for_status()
        print(f"[worker]   {item.get('youtube_channel_name') or item['lead_id']}: {payload['status']}"
              + (f" -> {payload['value']}" if payload.get("value") else ""))

    return len(items)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-mode", choices=["low", "high"], default="low")
    parser.add_argument("--limit", type=int, default=10, help="leads per page")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--loop", action="store_true", help="keep pulling until the queue is empty")
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=30.0) as client:
        total = 0
        while True:
            n = await process_page(client, args.cost_mode, args.limit, args.concurrency)
            total += n
            if n == 0 or not args.loop:
                break
        print(f"[worker] done — {total} leads processed" + (", queue empty" if args.loop else ""))


if __name__ == "__main__":
    asyncio.run(main())
