"""ICP classification worker daemon.

Polls the LMS for lists the user marked "Classify" (GET /classification/requested)
and drains each one — crawl + gpt-4o-mini + write the verdict back. Resumable and
idempotent: it only ever pulls leads not yet classified, and a list drops out of
the queue once its pending count hits 0. Pressing "Stop" in the UI clears the
list's request flag, which halts the worker mid-list on the next page.

Mirrors the email-finder worker: runs on the host (this .venv has crawl4ai +
Playwright), managed by launchd.

  python scripts/classification_worker.py [--concurrency 8] [--limit 100] [--once]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_agents.agents import icp_classifier as icp  # noqa: E402

LMS_API = os.environ.get("LMS_API_URL", "http://localhost:8000").rstrip("/")
POLL_S = 15.0
TRANSIENT_BACKOFF_S = 15.0


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=100, help="leads per page")
    ap.add_argument("--once", action="store_true", help="drain the queue once and exit")
    args = ap.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    print(f"[icp-worker] starting — LMS={LMS_API} concurrency={args.concurrency}")

    async with httpx.AsyncClient(timeout=90.0) as client:

        async def still_requested(batch_id: str) -> bool:
            try:
                r = await client.get(f"{LMS_API}/api/v1/classification/status",
                                     params={"batch_id": batch_id})
                j = r.json()
                # Global pause (user closed the laptop) halts mid-list too, not
                # just between lists — same status call already carries it.
                if j.get("paused"):
                    return False
                return bool(j.get("classify_requested"))
            except Exception:
                return True  # a blip shouldn't halt a long run

        try:
            while True:
                try:
                    resp = await client.get(f"{LMS_API}/api/v1/classification/requested")
                    resp.raise_for_status()
                    batches = resp.json()
                except (httpx.HTTPError, OSError) as exc:
                    print(f"[icp-worker] transient: {type(exc).__name__}: {exc}")
                    await asyncio.sleep(TRANSIENT_BACKOFF_S)
                    continue

                if not batches:
                    if args.once:
                        print("[icp-worker] nothing requested — done")
                        break
                    await asyncio.sleep(POLL_S)
                    continue

                for b in batches:
                    bid = b["batch_id"]
                    print(f"[icp-worker] classifying {b['filename']} — {b['pending']} pending")
                    try:
                        done, acc = await icp.drain_batch(
                            client, LMS_API, bid, sem, limit=args.limit,
                            should_continue=lambda bid=bid: still_requested(bid),
                        )
                        print(f"[icp-worker] {b['filename']}: {done} classified, {acc} paid-ads this pass")
                    except (httpx.HTTPError, OSError) as exc:
                        print(f"[icp-worker] transient during drain: {type(exc).__name__}: {exc}")
                        await asyncio.sleep(TRANSIENT_BACKOFF_S)

                if args.once:
                    break
        finally:
            await icp.close_crawler()

    print("[icp-worker] stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[icp-worker] interrupted")
