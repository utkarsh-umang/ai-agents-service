from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse
import httpx
from ai_agents.core.llm import langfuse, get_prompt
from ai_agents.agents.email_finder.io.contract_models import PerplexityInput, PerplexityOutput
from ai_agents.agents.email_finder.state import (
    EmailFinderState,
    EmailCandidate,
    LeadStatus,
    SourceType,
)

# Perplexity Agent API — same backend as the browser chat
_AGENT_API_URL = "https://api.perplexity.ai/v1/agent"

# pro-search: openai/gpt-5.1, up to 3 steps, web_search + fetch_url tools
# This matches the "Pro Search" mode in the Perplexity browser chat
_PRESET = "pro-search"

# Tells the Agent API to skip citation markers and return pure JSON
_AGENT_INSTRUCTIONS = (
    "Do not add any citation markers (e.g. [web:1], [page:2]) to your response. "
    "Return ONLY a valid JSON object. No markdown fences, no prose, no citations."
)


def _infer_website_from_linktree(url: str) -> str | None:
    """
    If a Linktree URL's slug looks like a domain (e.g. linktr.ee/jenrichardson.co),
    return that domain as the likely personal website (https://jenrichardson.co).
    Returns None for slugs that are plain handles (e.g. linktr.ee/johnsmith).
    """
    try:
        parsed = urlparse(url)
        if "linktr.ee" not in parsed.netloc:
            return None
        slug = parsed.path.strip("/")
        if "." in slug and not slug.startswith("."):
            return f"https://{slug}"
    except Exception:
        pass
    return None


def _build_query(state: EmailFinderState) -> str:
    """
    Construct the search query from available lead information.
    More context = better results.
    """
    lead = state.lead
    identity = lead.identity
    parts = []

    best_name = identity.best_name()
    context_name = identity.context_name()

    if best_name:
        parts.append(f"Name: {best_name}")
    if context_name:
        parts.append(f"Show/Channel: {context_name}")

    if state.lead.source_type in (
        SourceType.PODSCAN_HOST,
        SourceType.PODSCAN_GUEST,
    ):
        parts.append("Type: Podcast")
    elif state.lead.source_type == SourceType.YOUTUBE_SCRIPT_TOOL:
        parts.append("Type: YouTube Channel")

    if lead.website:
        parts.append(f"Website: {lead.website}")

    for url in lead.discovery_urls:
        parts.append(f"Profile/Link page: {url}")
        inferred = _infer_website_from_linktree(url)
        if inferred and inferred != lead.website:
            parts.append(f"Possible personal website (inferred from Linktree slug): {inferred}")

    if lead.existing_email:
        parts.append(
            f"Existing email (unverified): {lead.existing_email} "
            f"— check if this is correct or find a better one"
        )

    social = lead.social_links
    if social.youtube:
        parts.append(f"YouTube: {social.youtube}")
    if social.linkedin:
        parts.append(f"LinkedIn: {social.linkedin}")
    if social.twitter:
        parts.append(f"Twitter: {social.twitter}")
    if social.instagram:
        parts.append(f"Instagram: {social.instagram}")
    if social.facebook:
        parts.append(f"Facebook: {social.facebook}")

    return "\n".join(parts)


def _call_perplexity_agent(prompt: str) -> str:
    """
    Call the Perplexity Agent API with the pro-search preset.
    This is the same backend as the Perplexity browser chat Pro Search mode:
    - Model: openai/gpt-5.1
    - Tools: web_search + fetch_url (can visit specific URLs)
    - Max steps: 3
    """
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            _AGENT_API_URL,
            headers={
                "Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "preset": _PRESET,
                "input": prompt,
                "instructions": _AGENT_INSTRUCTIONS,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Collect all output_text content items from the response
    text_parts: list[str] = []
    for item in data.get("output", []):
        for content_block in item.get("content", []):
            if content_block.get("type") == "output_text":
                text_parts.append(content_block.get("text", ""))

    return "".join(text_parts).strip()


def _parse_perplexity_response(content: str) -> list[EmailCandidate]:
    # Strip markdown fences if present
    content = re.sub(
        r"^```json|^```|```$", "", content.strip(), flags=re.MULTILINE
    ).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    candidates = []
    for item in data.get("emails_found", []):
        email = item.get("email", "").strip()
        if not email or "@" not in email:
            continue
        candidates.append(
            EmailCandidate(
                email=email,
                source=f"perplexity — {item.get('source', 'unknown')}",
                confidence=float(item.get("confidence", 0.5)),
                note=item.get("note"),
            )
        )

    return candidates


def _rank_candidates(candidates: list[EmailCandidate]) -> list[EmailCandidate]:
    """Sort candidates by confidence, penalising generic prefixes."""
    generic_prefixes = {
        "info", "contact", "hello", "support",
        "admin", "team", "mail", "enquiries", "enquiry",
    }

    def score(candidate: EmailCandidate) -> float:
        prefix = candidate.email.split("@")[0].lower()
        penalty = 0.2 if prefix in generic_prefixes else 0.0
        return candidate.confidence - penalty

    return sorted(candidates, key=score, reverse=True)


def perplexity_discovery_run(inp: PerplexityInput) -> PerplexityOutput:
    """Contract-based Perplexity discovery."""
    trace = langfuse.trace(
        id=inp.trace_id,
        name="email_finder",
        session_id="email_finder",
    )
    span = trace.span(name="perplexity_discovery")

    try:
        ef_state = EmailFinderState(
            lead=inp.lead,
            status=LeadStatus.PENDING,
            email_candidates=inp.prior_email_candidates,
        )
        available_info = _build_query(ef_state)

        filled_prompt = get_prompt(
            "perplexity_email_discovery",
            available_info=available_info
        )

        raw_content = _call_perplexity_agent(filled_prompt)

        candidates = _parse_perplexity_response(raw_content)
        combined = inp.prior_email_candidates + candidates
        ranked = _rank_candidates(combined)

        span.end()
        trace.event(
            name="perplexity_complete",
            metadata={"new_candidates": len(candidates), "ranked_total": len(ranked)},
        )

        if not ranked:
            return PerplexityOutput(
                email_candidates=combined,
                best_email=None,
                status=LeadStatus.EMAIL_NOT_FOUND,
                errors=[],
            )

        return PerplexityOutput(
            email_candidates=combined,
            best_email=ranked[0],
            status=LeadStatus.EMAIL_FOUND,
            errors=[],
        )

    except Exception as e:
        span.end()
        trace.event(name="perplexity_failed", metadata={"error": str(e)})
        return PerplexityOutput(
            email_candidates=inp.prior_email_candidates,
            best_email=None,
            status=LeadStatus.FAILED,
            errors=[f"perplexity_discovery: {str(e)}"],
        )


def perplexity_discovery_node(state: EmailFinderState) -> EmailFinderState:
    """Legacy EmailFinderState API — builds PerplexityInput with a synthetic trace id."""

    trace = langfuse.trace(name="perplexity_discovery", session_id="email_finder")
    inp = PerplexityInput(
        lead=state.lead,
        trace_id=trace.id,
        prior_email_candidates=list(state.email_candidates),
    )
    out = perplexity_discovery_run(inp)
    return state.model_copy(update={
        "status": out.status,
        "email_candidates": out.email_candidates,
        "best_email": out.best_email,
        "errors": state.errors + out.errors,
        "nodes_executed": state.nodes_executed + out.nodes_executed_delta,
    })


async def perplexity_discovery_node_async(state: dict) -> dict:
    """LangGraph async node."""
    from ai_agents.agents.email_finder.adapters import perplexity_input_from_state

    inp = perplexity_input_from_state(state)
    out = perplexity_discovery_run(inp)
    return {
        "status": out.status.value,
        "email_candidates": [c.model_dump(mode="json") for c in out.email_candidates],
        "best_email": out.best_email.model_dump(mode="json") if out.best_email else None,
        "errors": out.errors,
        "nodes_executed": out.nodes_executed_delta,
    }
