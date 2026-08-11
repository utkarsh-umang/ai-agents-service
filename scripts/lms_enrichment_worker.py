"""Persistent enrichment worker: drains LMS's email queue via the email
finder, driven by long-polling — no idle chatter, instant pickup.

The loop is a single blocking call from the worker's point of view:
`GET /enrichment/queue/wait` holds until work exists (or ~55s timeout,
then we immediately re-call). LMS wakes held requests the moment an
ingestion commits. While the human-in-the-loop gate is closed (pause),
the same call simply keeps blocking — pressing Resume in the UI wakes it.

Error taxonomy (the part that keeps automation trustworthy):
- lead-level  — one lead's crawl/parse blew up → record status="failed",
                move on. That lead is done at this tier.
- transient   — LMS/network briefly unreachable → short local backoff,
                nothing recorded, retry.
- hard block  — LLM credits exhausted / invalid API key → record NOTHING
                (leads stay queued), POST /pause, and wait for a human to
                press Resume. Never a timed retry against an error that
                cannot self-heal.

Concurrency defaults (12 in flight, pages of 50) come straight from the
proven `email-finder-youtube-19k-no-perplexity` notebook run — the same
low-cost workload profile.

Usage:
    python scripts/lms_enrichment_worker.py            # daemon (long-poll loop)
    python scripts/lms_enrichment_worker.py --once     # drain once, then exit
Env: LMS_API_URL (default http://localhost:8000)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_agents.agents.email_finder.graph import build_graph, run_single_async  # noqa: E402
from ai_agents.agents.email_finder.nodes import browser_pool  # noqa: E402
from ai_agents.agents.email_finder.guest_finder import find_guest_email  # noqa: E402
from ai_agents.agents.email_finder.company_finder import find_company_contact  # noqa: E402
from ai_agents.agents.email_finder.state import LeadStatus, SourceType  # noqa: E402

LMS_API = os.environ.get("LMS_API_URL", "http://localhost:8000").rstrip("/")
PROVIDER = "email_finder@ai-agents-service"

# Server holds /queue/wait up to this long; client must out-wait it.
WAIT_TIMEOUT_S = 55.0
CLIENT_TIMEOUT_S = 75.0
TRANSIENT_BACKOFF_S = 20.0
# Hard per-lead budget (the lite notebook's PER_LEAD_BUDGET_S lesson).
# One un-timeouted network call anywhere in the graph must never jam the
# whole pipeline: a lead that exceeds this is recorded "failed" and the
# page moves on. Generous — a normal lead with heavy crawling takes 1-3min.
PER_LEAD_BUDGET_S = 300.0

# Fully recycle the shared crawl browser (browser_pool) every N pages, between
# pages where no crawl is in flight — a deterministic memory valve on top of the
# in-browser page recycle. Replaces the old pkill reaper; the persistent browser
# no longer orphans on cancel, so this is housekeeping, not leak-control.
RECYCLE_EVERY_PAGES = 10

# Signatures of errors that can never self-heal — pausing for a human is
# the only correct response. Matched against the exception's full repr.
HARD_BLOCK_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "invalid_api_key",
    "incorrect api key",
    "authenticationerror",
    "billing",
)


class HardBlock(Exception):
    """LLM provider says stop: quota/billing/key. Human required."""


def _classify(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in HARD_BLOCK_MARKERS):
        return "hard_block"
    return "lead_level"


def _queue_item_to_raw_row(item: dict) -> dict:
    return {
        "channel_name": item.get("youtube_channel_name"),
        "youtube_handle": item.get("youtube_handle"),
        "website": item.get("website"),
        # both spellings: canonical_builder maps this into social_links, and
        # the graph's _youtube_enabled gate also reads raw "channel_url".
        "youtube_url": item.get("social_youtube"),
        "channel_url": item.get("social_youtube"),
        "twitter_url": item.get("social_twitter"),
        "instagram_url": item.get("social_instagram"),
        "tiktok_url": item.get("social_tiktok"),
        "facebook_url": item.get("social_facebook"),
        "linkedin_url": item.get("social_linkedin"),
        "niche": item.get("niche"),
        "country": item.get("country"),
    }


def _queue_item_to_guest_lead(item: dict) -> dict:
    """Queue item -> the identity dict the search-first guest finder needs."""
    name = f"{(item.get('first_name') or '').strip()} {(item.get('last_name') or '').strip()}".strip()
    return {
        "name": name,
        "company": item.get("company_name"),
        "occupation": item.get("job_title"),
        "industry": item.get("industry"),
        "website": item.get("website"),
        "linkedin": item.get("social_linkedin"),
        "twitter": item.get("social_twitter"),
        "instagram": item.get("social_instagram"),
    }


def _queue_item_to_company_lead(item: dict) -> dict:
    """Queue item -> the company-as-lead dict the Clutch contact resolver needs."""
    return {
        "company": item.get("company_name"),
        "clutch_profile_url": item.get("clutch_profile_url"),
        "website": item.get("website"),
    }


def _company_result_payload(item: dict, state: dict, cost_mode: str) -> dict:
    """Like _result_payload, but a company lead's result also carries WHO the
    address belongs to (the resolver picked one senior person) and the verifier's
    deliverability verdict."""
    best = state.get("best_email") or {}
    email = best.get("email") if isinstance(best, dict) else None
    found = state.get("status") == LeadStatus.EMAIL_FOUND.value and bool(email)
    failed = state.get("status") == LeadStatus.FAILED.value
    name = (best.get("person_name") or "").strip()
    first, last = (name.split(" ", 1) + [""])[:2] if name else ("", "")
    return {
        "lead_id": item["lead_id"],
        "type": "email",
        "cost_mode": cost_mode,
        "status": "found" if found else ("failed" if failed else "not_found"),
        "value": email if found else None,
        "confidence": best.get("confidence") if found else None,
        "provider": PROVIDER,
        "cost_incurred": round(float(state.get("cost_usd") or 0.0), 6),
        "evidence": state.get("evidence"),
        # company-lead extras (person the address belongs to + verifier verdict)
        "person_first_name": (first or None) if found else None,
        "person_last_name": (last or None) if found else None,
        "person_job_title": (best.get("person_title") or None) if found else None,
        "person_seniority": (best.get("seniority") or None) if found else None,
        "email_status": (best.get("email_status") or None) if found else None,
    }


def _result_payload(item: dict, state: dict, cost_mode: str) -> dict:
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
        # Real per-lead spend accumulated by the graph (exact litellm costs +
        # estimated ScrapingBee/Perplexity per-call costs). Rounded: sub-cent
        # precision matters when a lead costs $0.0003.
        "cost_incurred": round(float(state.get("cost_usd") or 0.0), 6),
        # Candidate context (guest finder) so a later logic change can re-score
        # this attempt offline — no re-search, no re-scrape.
        "evidence": state.get("evidence"),
        # Verifier verdict when present (the pattern tier's Mailin result:
        # "ok" | "catch_all"). Stored on lead.email_status; None for the free
        # search tier, which does no paid verification.
        "email_status": (best.get("email_status") if found else None),
    }


def _reap_stale_browsers() -> None:
    """Clear any Crawl4AI/Playwright chromium left by a PREVIOUS worker instance.

    Steady-state there are no orphans: the finder shares one persistent browser
    (browser_pool) that a cancelled crawl can't orphan, and a clean shutdown
    closes it. But launchd stops the daemon with SIGTERM/SIGKILL, which runs no
    Python cleanup — so a hard kill can leave the previous instance's single
    browser behind. Running this ONCE at startup (before our pool is created)
    guarantees at most one live browser at any time across restarts.

    Path-scoped to Playwright's own chromium build so a user's Chrome/Brave is
    untouched. Best-effort — pkill exit 1 just means nothing to reap."""
    try:
        subprocess.run(
            ["pkill", "-f", "ms-playwright/chromium"],
            capture_output=True,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001 — startup reap must never break boot
        print(f"[worker] startup browser reap skipped: {type(exc).__name__}: {exc}")


async def _heartbeat(client: httpx.AsyncClient, state: str, detail: str | None = None, in_flight: int = 0) -> None:
    try:
        await client.post(
            f"{LMS_API}/api/v1/enrichment/heartbeat",
            json={"state": state, "detail": detail, "in_flight": in_flight},
        )
    except httpx.HTTPError:
        pass  # heartbeats are best-effort; the queue semantics don't depend on them


async def _pause(client: httpx.AsyncClient, reason: str) -> None:
    await client.post(f"{LMS_API}/api/v1/enrichment/pause", json={"reason": reason})


async def process_page(
    client: httpx.AsyncClient, items: list[dict], cost_mode: str, concurrency: int
) -> None:
    """Run one page through the finder, posting each lead's result AS IT
    FINISHES (not after the whole page): a killed worker loses only the
    in-flight leads' spend, the queue/dashboard move in real time, and a
    50-lead page can't sit for an hour looking dead. Raises HardBlock at
    the end if a systemic LLM error was detected — those leads get nothing
    recorded and stay queued."""
    graph = build_graph()
    semaphore = asyncio.Semaphore(concurrency)
    done_count = 0
    hard_block_reason: str | None = None

    async def post_result(payload: dict, item: dict) -> None:
        resp = await client.post(f"{LMS_API}/api/v1/enrichment/results", json=payload)
        resp.raise_for_status()
        label = item.get("youtube_channel_name") or item["lead_id"]
        print(f"[worker]   {label}: {payload['status']}"
              + (f" -> {payload['value']}" if payload.get("value") else ""))

    async def run_one(item: dict) -> None:
        nonlocal done_count, hard_block_reason
        try:
            # Hold the concurrency slot OUTSIDE the budget timer — a lead
            # queued behind others must not burn budget while waiting. The
            # inner call gets a fresh single-use semaphore that never blocks.
            # Routing: a Clutch company-lead (has a profile URL, no person/site)
            # goes to the company-contact resolver; a podscan GUEST (a tagged
            # person to search for by name+company) to the guest finder;
            # everything else — including a podscan HOST (the podcast itself, a
            # site to crawl, no person) — to the crawl graph.
            is_company = bool(item.get("clutch_profile_url"))
            # podscan-host carries lead_tag='podcast_host' but no person, so it
            # must NOT go to the person-search guest finder — the crawl graph
            # scrapes its website for the show's email instead.
            is_guest = bool(item.get("lead_tag")) and item.get("lead_tag") != "podcast_host"
            async with semaphore:
                if is_company:
                    # Sync (drives Crawl4AI + async I/O on its own loop) -> thread.
                    state = await asyncio.wait_for(
                        asyncio.to_thread(find_company_contact, _queue_item_to_company_lead(item), cost_mode),
                        timeout=PER_LEAD_BUDGET_S,
                    )
                elif is_guest:
                    # Tagged (podscan) lead -> the search-first guest finder.
                    # It's sync (drives Crawl4AI on a private loop), so run it in
                    # a thread to keep this event loop free.
                    state = await asyncio.wait_for(
                        asyncio.to_thread(find_guest_email, _queue_item_to_guest_lead(item), cost_mode),
                        timeout=PER_LEAD_BUDGET_S,
                    )
                else:
                    state = await asyncio.wait_for(
                        run_single_async(
                            graph,
                            _queue_item_to_raw_row(item),
                            SourceType.YOUTUBE_SCRIPT_TOOL,
                            asyncio.Semaphore(1),
                            # youtube_list enables the About-page enricher for
                            # leads with a YouTube URL and no website — the graph
                            # gates per-lead, so it's safe unconditionally.
                            youtube_list=True,
                            source="lms",
                            cost_mode=cost_mode,
                        ),
                        timeout=PER_LEAD_BUDGET_S,
                    )
            payload = (_company_result_payload if is_company else _result_payload)(item, state, cost_mode)
            await post_result(payload, item)
        except asyncio.TimeoutError:
            label = item.get("youtube_channel_name") or item["lead_id"]
            print(f"[worker]   {label}: exceeded {PER_LEAD_BUDGET_S:.0f}s budget — recording failed")
            await post_result(
                {
                    "lead_id": item["lead_id"],
                    "type": "email",
                    "cost_mode": cost_mode,
                    "status": "failed",
                    "provider": PROVIDER,
                },
                item,
            )
        except HardBlock:
            raise
        except Exception as exc:  # noqa: BLE001 — classified, never swallowed
            if _classify(exc) == "hard_block":
                # Record nothing — this lead must stay in the queue.
                hard_block_reason = f"{type(exc).__name__}: {exc}"
                return
            await post_result(
                {
                    "lead_id": item["lead_id"],
                    "type": "email",
                    "cost_mode": cost_mode,
                    "status": "failed",
                    "provider": PROVIDER,
                },
                item,
            )
        finally:
            done_count += 1

    # Background heartbeat while the page runs — without it, a long page
    # makes the dashboard claim "worker not seen" while it's hard at work.
    async def heartbeat_loop() -> None:
        while True:
            await _heartbeat(
                client, "processing",
                f"processing page: {done_count}/{len(items)} done",
                len(items) - done_count,
            )
            await asyncio.sleep(30)

    hb = asyncio.create_task(heartbeat_loop())
    try:
        await asyncio.gather(*(run_one(i) for i in items))
    finally:
        hb.cancel()

    if hard_block_reason:
        raise HardBlock(hard_block_reason)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-mode", choices=["low", "high"], default="low")
    parser.add_argument("--limit", type=int, default=50, help="leads per page")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--once", action="store_true", help="drain the current queue and exit")
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="stop after N pages regardless of queue depth (use --pages 1 for a bounded test; "
        "--once alone drains the ENTIRE queue)",
    )
    args = parser.parse_args()

    print(f"[worker] starting — LMS={LMS_API} cost_mode={args.cost_mode} "
          f"page={args.limit} concurrency={args.concurrency} "
          f"mode={'once' if args.once else 'daemon'}")

    # Clear any browser a previous (hard-killed) instance left behind before we
    # start our own persistent one.
    _reap_stale_browsers()

    pages_done = 0
    client = httpx.AsyncClient(timeout=CLIENT_TIMEOUT_S)
    try:
        while True:
            if args.pages is not None and pages_done >= args.pages:
                print(f"[worker] page limit ({args.pages}) reached — done")
                return
            try:
                await _heartbeat(client, "waiting")
                resp = await client.get(
                    f"{LMS_API}/api/v1/enrichment/queue/wait",
                    params={
                        "cost_mode": args.cost_mode,
                        "limit": args.limit,
                        "timeout": WAIT_TIMEOUT_S,
                    },
                )
                resp.raise_for_status()
                items = resp.json()

                if not items:
                    if args.once:
                        print("[worker] queue empty — done")
                        return
                    continue  # timeout tick or just-resumed; re-enter the wait

                print(f"[worker] pulled {len(items)} leads")
                await _heartbeat(client, "processing",
                                 f"processing {len(items)} leads", len(items))
                await process_page(client, items, args.cost_mode, args.concurrency)
                pages_done += 1
                # Between pages, no crawl is in flight. The shared browser is
                # persistent (no per-crawl orphans anymore), so this is just a
                # periodic full recycle to bound memory — not leak-control.
                if pages_done % RECYCLE_EVERY_PAGES == 0:
                    await browser_pool.recycle_crawler()

            except HardBlock as exc:
                reason = f"Email finder paused: {exc}"
                print(f"[worker] HARD BLOCK — {reason}")
                try:
                    await _pause(client, reason)
                    await _heartbeat(client, "blocked", reason)
                except httpx.HTTPError:
                    pass
                if args.once:
                    sys.exit(2)
                # No sleep needed: /queue/wait now blocks server-side until
                # a human presses Resume. Just re-enter the loop.

            except (httpx.HTTPError, OSError) as exc:
                # LMS down / network blip — transient by definition here.
                print(f"[worker] transient: {type(exc).__name__}: {exc} — "
                      f"retrying in {TRANSIENT_BACKOFF_S:.0f}s")
                if args.once:
                    sys.exit(1)
                await asyncio.sleep(TRANSIENT_BACKOFF_S)
    finally:
        # Clean shutdown (once-mode / page-limit / Ctrl-C): close the shared
        # browser so it isn't orphaned. (A SIGKILL skips this — the startup
        # reap covers that case on the next boot.)
        await browser_pool.close_crawler()
        await client.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[worker] stopped")
