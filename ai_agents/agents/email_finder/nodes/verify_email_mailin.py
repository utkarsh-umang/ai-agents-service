"""Mailin email-verification tool — a deliverability check for a candidate
address (does this mailbox actually exist / accept mail).

Distinct from the finder's own confidence, which scores whether an address
*belongs* to the lead. This asks the orthogonal question: is it *deliverable*.
The two compose — a name-matched address that Mailin says is `invalid` should
not be shipped, and a blind `first.last@` pattern guess only becomes trustworthy
once Mailin returns `ok`.

Auth: a bearer token (Mailin's "API key", minted once via POST /auth/login per
their docs) read from config.MAILIN_API_TOKEN — calls never carry the login
password. The API sits behind Cloudflare, so a browser User-Agent is required
(a plain client signature gets a 1010 ban). Verification is effectively
synchronous (the submit call returns the verdict); we still poll get-single-email
as a fallback for any address that comes back `verifying`.

Cost: `standard` = 1 credit/email; `catch_all` = 7 credits (run only after a
standard `ok` on a catch-all domain, if you need to disambiguate). Leave
MAILIN_API_TOKEN blank to disable — the tool then returns result="unverified".
"""
from __future__ import annotations

import asyncio

import httpx

from ai_agents.core import config

# result string -> deliverable tri-state. True = send it, False = drop it,
# None = uncertain (catch-all / unknown / not run) — caller decides policy.
_DELIVERABLE = {
    "ok": True,
    "valid": True,
    "invalid": False,
    "bad": False,
    "catch_all": None,
    "catch-all": None,
    "accept_all": None,
    "unknown": None,
    "risky": None,
    "unverified": None,
}

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)


def _headers() -> dict[str, str]:
    return {
        "User-Agent": _UA,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "https://app.mailin.ai",
        "Referer": "https://app.mailin.ai/",
        "Authorization": f"Bearer {config.MAILIN_API_TOKEN}",
    }


def _normalize(payload: dict) -> dict:
    """Flatten Mailin's response (fields sometimes nested under `data`) into a
    stable verdict dict."""
    d = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    result = (d.get("result") or "").lower() or "unverified"
    return {
        "result": result,                                   # ok | invalid | catch_all | ...
        "score": (d.get("score") or "").lower() or None,    # good | bad | ...
        "deliverable": _DELIVERABLE.get(result),            # True | False | None
        "credits_used": d.get("credits_used") or d.get("credit_used") or 0,
        "task_id": d.get("task_id"),
        "raw": d,
    }


async def verify_email_mailin(
    email: str, kind: str = "standard", poll_timeout_s: float = 45.0
) -> dict:
    """Verify one address. Returns a verdict dict (see _normalize). Never
    raises for a normal API/network error — returns result="unverified" so a
    verification outage degrades to "unknown", never blocks the pipeline."""
    if not config.MAILIN_API_TOKEN:
        return {"result": "unverified", "score": None, "deliverable": None,
                "credits_used": 0, "task_id": None, "raw": {"reason": "no_token"}}
    base = config.MAILIN_BASE_URL.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=60.0, headers=_headers()) as client:
            resp = await client.post(
                f"{base}/verify-single-email", json={"email": email, "type": kind}
            )
            resp.raise_for_status()
            body = resp.json()
            # Synchronous case: the submit call already carries the verdict.
            status = (body.get("status") or (body.get("data") or {}).get("status") or "").lower()
            has_result = "result" in body or "result" in (body.get("data") or {})
            if status != "verifying" and has_result:
                return _normalize(body)
            # Async fallback: poll get-single-email until it settles.
            task_id = body.get("task_id") or (body.get("data") or {}).get("task_id")
            if not task_id:
                return _normalize(body)
            deadline = asyncio.get_event_loop().time() + poll_timeout_s
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(3)
                det = await client.get(f"{base}/get-single-email", params={"task_id": task_id})
                det.raise_for_status()
                dj = det.json()
                st = (dj.get("status") or (dj.get("data") or {}).get("status") or "").lower()
                if st and st != "verifying":
                    return _normalize(dj)
            return {"result": "unverified", "score": None, "deliverable": None,
                    "credits_used": 1, "task_id": task_id, "raw": {"reason": "poll_timeout"}}
    except Exception as exc:  # network / HTTP / JSON — degrade to unknown
        return {"result": "unverified", "score": None, "deliverable": None,
                "credits_used": 0, "task_id": None, "raw": {"error": f"{type(exc).__name__}: {exc}"}}


def verify_email_mailin_sync(email: str, kind: str = "standard") -> dict:
    """Blocking wrapper for scripts/notebooks outside an event loop."""
    return asyncio.run(verify_email_mailin(email, kind))


async def get_mailin_credits() -> int | None:
    """Remaining verification credits, or None if unavailable."""
    if not config.MAILIN_API_TOKEN:
        return None
    base = config.MAILIN_BASE_URL.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=40.0, headers=_headers()) as client:
            r = await client.get(f"{base}/get-credits")
            r.raise_for_status()
            j = r.json()
            return (j.get("total_credits") or (j.get("data") or {}).get("total_credits"))
    except Exception:
        return None
