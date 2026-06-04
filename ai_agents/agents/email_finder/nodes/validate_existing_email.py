"""Validate a lead's existing email against its own homepage."""
from __future__ import annotations

import httpx

from ai_agents.agents.email_finder.adapters import validate_email_input_from_state
from ai_agents.agents.email_finder.io.contract_models import (
    ValidateEmailInput,
    ValidateEmailOutput,
)
from ai_agents.agents.email_finder.state import EmailCandidate, LeadStatus
from ai_agents.core.llm import langfuse

_GENERIC = frozenset(
    {"info", "contact", "hello", "support", "admin", "team", "mail", "enquiries", "enquiry", "noreply"}
)


def _confidence_for_email(email: str) -> float:
    local = email.split("@")[0].lower()
    if local in _GENERIC:
        return 0.55
    return 0.75


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
    website = inp.lead.website or ""

    # ── Fetch homepage ────────────────────────────────────────────────────────
    page_text = ""
    if website:
        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(website)
                resp.raise_for_status()
                page_text = resp.text
        except Exception as exc:
            span.end()
            trace.event(
                name="homepage_fetch_failed",
                metadata={"website": website, "error": str(exc)},
            )
            return ValidateEmailOutput(
                errors=[f"homepage_fetch_failed: {exc}"],
            )

    # ── Confidence enrichment (same logic as crawl_page.py) ──────────────────
    base_confidence = _confidence_for_email(email)
    enriched_confidence = base_confidence
    note_parts: list[str] = []

    raw_local = email.split("@")[0].lower()
    local_part = raw_local.rstrip("0123456789")

    if len(local_part) >= 4:
        if local_part in website.lower():
            enriched_confidence += 0.15
            note_parts.append(f"local part '{local_part}' matched in URL")

        if local_part in page_text.lower():
            enriched_confidence += 0.10
            note_parts.append(f"local part '{local_part}' matched in page text")

    enriched_confidence = min(enriched_confidence, 0.95)
    note = "; ".join(note_parts) if note_parts else "extracted from page content"

    span.end()

    # ── Threshold decision ────────────────────────────────────────────────────
    if enriched_confidence > 0.75:
        trace.event(
            name="validate_email_found",
            metadata={"email": email, "confidence": enriched_confidence},
        )
        return ValidateEmailOutput(
            best_email=EmailCandidate(
                email=email,
                source="structured_data — existing email",
                confidence=enriched_confidence,
                note=note,
            ),
            status=LeadStatus.EMAIL_FOUND,
        )

    trace.event(
        name="validate_email_not_found",
        metadata={"email": email, "confidence": enriched_confidence},
    )
    return ValidateEmailOutput(
        best_email=None,
        status=LeadStatus.EMAIL_NOT_FOUND,
    )


async def validate_existing_email_node_async(state: dict) -> dict:
    inp = validate_email_input_from_state(state)
    output = _validate(inp)
    return {
        "best_email": output.best_email.model_dump(mode="json") if output.best_email else None,
        "status": output.status.value,
        "errors": output.errors,
        "nodes_executed": output.nodes_executed_delta,
    }
