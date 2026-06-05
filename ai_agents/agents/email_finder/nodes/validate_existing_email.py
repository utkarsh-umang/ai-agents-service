"""Validate a lead's existing email against its own homepage."""
from __future__ import annotations

from urllib.parse import urlparse

import httpx

from ai_agents.agents.email_finder.adapters import validate_email_input_from_state
from ai_agents.agents.email_finder.io.contract_models import (
    ValidateEmailInput,
    ValidateEmailOutput,
)
from ai_agents.agents.email_finder.nodes.email_utils import confidence_for_email, is_plus_addressed
from ai_agents.agents.email_finder.state import EmailCandidate, LeadStatus
from ai_agents.core.llm import langfuse


def _structural_confidence(email: str, website: str) -> tuple[float, list[str]]:
    """
    Score using signals that don't require an HTTP call.

    These work on JS-rendered sites too, so they're the reliable baseline.
    Domain match alone can push a non-generic email to 0.90 (past the 0.85 threshold).
    """
    base = confidence_for_email(email)
    notes: list[str] = []
    conf = base

    if not website:
        return conf, notes

    try:
        netloc = urlparse(website if website.startswith("http") else f"https://{website}").netloc.lower()
        website_domain = netloc.removeprefix("www.")
    except Exception:
        return conf, notes

    email_domain = email.split("@")[-1].lower() if "@" in email else ""
    local_part = email.split("@")[0].lower().rstrip("0123456789")

    # Email domain matches website domain — strongest structural signal
    if email_domain and website_domain and email_domain == website_domain:
        conf += 0.15
        notes.append("email domain matches website domain")

    # Local part appears in the website domain (e.g. sarah@sarahspodcast.com)
    if len(local_part) >= 4 and local_part in website_domain:
        conf += 0.10
        notes.append(f"local part '{local_part}' matched in website domain")

    return min(conf, 0.95), notes


def _page_confidence_boost(email: str, website: str, page_text: str) -> tuple[float, list[str]]:
    """Additional boost from homepage content — best-effort only."""
    notes: list[str] = []
    boost = 0.0
    local_part = email.split("@")[0].lower().rstrip("0123456789")

    if len(local_part) < 4 or not page_text:
        return boost, notes

    if local_part in website.lower():
        boost += 0.05
        notes.append(f"local part '{local_part}' matched in URL")

    if local_part in page_text.lower():
        boost += 0.05
        notes.append(f"local part '{local_part}' matched in page text")

    return boost, notes


def _validate(inp: ValidateEmailInput) -> ValidateEmailOutput:
    trace = langfuse.trace(
        id=inp.trace_id,
        name="email_finder",
        session_id="email_finder",
    )
    span = trace.span(
        name="validate_existing_email",
        metadata={"existing_email": inp.lead.existing_email, "website": inp.lead.website},
    )

    email = inp.lead.existing_email
    if is_plus_addressed(email):
        span.end()
        trace.event(name="validate_email_skipped", metadata={"reason": "plus-addressed email"})
        return ValidateEmailOutput(errors=["plus-addressed email skipped"])

    website = inp.lead.website or ""

    # ── Structural signals (always available, no HTTP needed) ─────────────────
    conf, notes = _structural_confidence(email, website)

    # ── Best-effort homepage fetch for additional signal ──────────────────────
    # RC3 fix: fetch failure no longer drops the email — we fall back to
    # structural confidence and always produce at least a sub_threshold_candidate.
    if website:
        try:
            with httpx.Client(timeout=12, follow_redirects=True) as client:
                resp = client.get(website)
                resp.raise_for_status()
                page_text = resp.text
            boost, page_notes = _page_confidence_boost(email, website, page_text)
            conf = min(conf + boost, 0.95)
            notes.extend(page_notes)
        except Exception as exc:
            notes.append("homepage fetch failed — using structural signals only")
            trace.event(name="homepage_fetch_failed", metadata={"website": website, "error": str(exc)})

    span.end()

    note = "; ".join(notes) if notes else "structured data email"
    candidate = EmailCandidate(
        email=email,
        source="structured_data — existing email",
        confidence=conf,
        note=note,
    )

    # ── Threshold decision ────────────────────────────────────────────────────
    if conf >= 0.85:
        trace.event(name="validate_email_found", metadata={"email": email, "confidence": conf})
        return ValidateEmailOutput(best_email=candidate, status=LeadStatus.EMAIL_FOUND)

    trace.event(name="validate_email_below_threshold", metadata={"email": email, "confidence": conf})
    return ValidateEmailOutput(
        best_email=None,
        status=LeadStatus.EMAIL_NOT_FOUND,
        sub_threshold_candidate=candidate,
    )


async def validate_existing_email_node_async(state: dict) -> dict:
    inp = validate_email_input_from_state(state)
    output = _validate(inp)
    result: dict = {
        "best_email": output.best_email.model_dump(mode="json") if output.best_email else None,
        "status": output.status.value,
        "errors": output.errors,
        "nodes_executed": output.nodes_executed_delta,
    }
    if output.sub_threshold_candidate:
        result["email_candidates"] = [output.sub_threshold_candidate.model_dump(mode="json")]
    return result
