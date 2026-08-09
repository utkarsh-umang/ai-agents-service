"""Process-wide shared Crawl4AI browser for the email-finder graph.

Why this exists
---------------
The graph used to open a fresh ``AsyncWebCrawler`` — a new headless Chromium
process — for every URL: the homepage crawl in ``url_discovery`` plus one per
fan-out page in ``crawl_page`` (~6 per lead). That was expensive twice over:

* **Latency.** Every crawl paid a browser cold-start (~1-3s) before it even
  navigated — ~5 of them per lead on the common path.
* **The orphaned-chromium leak.** A crawl cancelled at the worker's
  ``PER_LEAD_BUDGET_S`` deadline interrupted the ``async with AsyncWebCrawler``
  teardown, so the browser subprocess was never closed. They piled up until the
  box was exhausted — the incident that forced worker concurrency 4 -> 2.

One long-lived browser fixes both. Nothing is created or destroyed per crawl,
so a cancelled ``arun`` just abandons a page — the browser stays warm and
reusable, and there is nothing to orphan. Memory is bounded two ways: a
per-navigation ``max_pages_before_recycle`` backstop inside the browser, and a
deterministic ``recycle_crawler()`` the worker calls between pages (when no
crawl is in flight), which replaces the old ``pkill`` reaper.

Lifecycle is owned by the worker: the crawler starts lazily on the first crawl
and is closed on shutdown. The pool is keyed to the event loop it was started
on, so notebook / ``run_batch`` runs that spin a fresh loop transparently get a
fresh crawler instead of reusing one bound to a dead loop.
"""
from __future__ import annotations

import asyncio
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig

# One Chromium serving up to this many concurrent pages. Worker lead-concurrency
# x fan-out (URLs/lead) sets real demand (2 x 5 = 10 today); this ceiling keeps
# page count bounded if lead-concurrency is raised later.
MAX_CONCURRENT_PAGES = 12

# In-browser backstop: recycle pages after this many navigations to bound
# memory even if the worker never calls recycle_crawler(). The worker's
# between-pages recycle is the primary valve; this just guards a long single
# page from leaking unbounded.
_MAX_PAGES_BEFORE_RECYCLE = 200

_crawler: AsyncWebCrawler | None = None
_crawler_loop: asyncio.AbstractEventLoop | None = None
_page_sem: asyncio.Semaphore | None = None


def _browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        verbose=False,
        max_pages_before_recycle=_MAX_PAGES_BEFORE_RECYCLE,
    )


async def _safe_close(crawler: AsyncWebCrawler) -> None:
    try:
        await crawler.close()
    except Exception:
        pass


async def _ensure_crawler() -> AsyncWebCrawler:
    """Return the shared crawler for the running loop, starting it if needed.

    Race-safe without a loop-bound lock: concurrent first-callers may each start
    a browser, but only one wins the module slot and the losers close theirs, so
    at most one transient extra browser exists for a moment at startup.
    """
    global _crawler, _crawler_loop, _page_sem
    loop = asyncio.get_running_loop()
    if _crawler is not None and _crawler_loop is loop:
        return _crawler

    crawler = AsyncWebCrawler(config=_browser_config())
    await crawler.start()
    if _crawler is not None and _crawler_loop is loop:
        # Another coroutine won the slot while we were starting — discard ours.
        await _safe_close(crawler)
        return _crawler
    _page_sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
    _crawler = crawler
    _crawler_loop = loop
    return _crawler


async def crawl(url: str, run_config: Any):
    """Run one navigation on the shared browser.

    No timeout/retry here — callers keep their own wall-clock cap (crawl_page's
    hard timeout, url_discovery's page_timeout). Concurrency is bounded by
    MAX_CONCURRENT_PAGES so a burst of fan-out pages can't overwhelm the browser.
    """
    crawler = await _ensure_crawler()
    sem = _page_sem
    if sem is None:  # defensive; _ensure_crawler always sets it alongside _crawler
        return await crawler.arun(url=url, config=run_config)
    async with sem:
        return await crawler.arun(url=url, config=run_config)


async def close_crawler() -> None:
    """Close the shared browser (worker shutdown). Safe to call repeatedly."""
    global _crawler, _crawler_loop, _page_sem
    crawler, _crawler = _crawler, None
    _crawler_loop = None
    _page_sem = None
    if crawler is not None:
        await _safe_close(crawler)


async def recycle_crawler() -> None:
    """Close the browser so the next crawl lazily starts a fresh one.

    The worker calls this between pages — a point where no crawl is in flight —
    as the deterministic memory valve that replaces the old pkill reaper.
    """
    await close_crawler()
