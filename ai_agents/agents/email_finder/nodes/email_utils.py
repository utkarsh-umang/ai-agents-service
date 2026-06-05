"""Shared utilities for email extraction and confidence scoring across nodes."""
from __future__ import annotations

_GENERIC = frozenset({
    "info", "contact", "hello", "support", "admin",
    "team", "mail", "enquiries", "enquiry", "noreply",
})


def is_plus_addressed(email: str) -> bool:
    """Return True for tagged/plus-addressed emails like info+xyz@domain.com."""
    local = email.split("@")[0]
    return "+" in local


def confidence_for_email(email: str) -> float:
    local = email.split("@")[0].lower()
    if local in _GENERIC:
        return 0.55
    return 0.75


def enrich_confidence(
    email: str,
    url: str,
    blob: str,
    base_confidence: float,
) -> tuple[float, str]:
    local_part = email.split("@")[0].lower().rstrip("0123456789")

    if len(local_part) < 4:
        return base_confidence, "extracted from page content"

    confidence = base_confidence
    notes: list[str] = []

    if local_part in url.lower():
        confidence += 0.15
        notes.append(f"local part '{local_part}' matched in URL")

    if local_part in blob.lower():
        confidence += 0.10
        notes.append(f"local part '{local_part}' matched in page text")

    confidence = min(confidence, 0.95)
    note = "; ".join(notes) if notes else "extracted from page content"
    return confidence, note


def filter_emails(emails: list[str]) -> list[str]:
    """Remove plus-addressed and obviously invalid emails."""
    return [e for e in emails if not is_plus_addressed(e)]
