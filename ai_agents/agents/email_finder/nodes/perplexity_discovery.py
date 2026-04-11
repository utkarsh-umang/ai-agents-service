from __future__ import annotations

import json
import re
from typing import Optional
import litellm
from core.llm import langfuse, get_prompt
from agents.email_finder.state import (
    EmailFinderState,
    EmailCandidate,
    LeadStatus,
    SourceType,
)


FALLBACK_PROMPT = """
You are an expert at finding contact emails for podcasts and YouTube channels.

You will be given information about a podcast or YouTube channel.
Your job is to find the best contact email for reaching the host or creator directly.

Available information:
{available_info}

Instructions:
- Search for the contact email of this person or show
- Prefer personal emails over generic ones (avoid info@, contact@, hello@, support@)
- If you find multiple emails, list all of them
- For each email found, mention where you found it (website, social media, etc.)
- If you cannot find any email, say so clearly

Return your response as JSON only:
{{
    "emails_found": [
        {{
            "email": "string",
            "source": "string — where you found it",
            "confidence": 0.0 to 1.0,
            "note": "string or null"
        }}
    ],
    "search_summary": "brief summary of what you found and where you looked"
}}

Confidence scoring guide:
- 1.0 — found directly on their personal website or LinkedIn
- 0.8 — found on podcast/channel website contact page
- 0.6 — found on social media bio or linktree
- 0.4 — found mentioned in a third party source
- 0.2 — inferred or uncertain
"""


def _build_query(state: EmailFinderState) -> str:
    """
    Construct the search query from available lead information.
    More context = better Perplexity results.
    """
    lead = state.lead
    identity = lead.identity
    parts = []

    # Primary identity
    best_name = identity.best_name()
    context_name = identity.context_name()

    if best_name:
        parts.append(f"Name: {best_name}")
    if context_name:
        parts.append(f"Show/Channel: {context_name}")

    # Source type context
    if state.lead.source_type in (
        SourceType.PODSCAN_HOST,
        SourceType.PODSCAN_GUEST,
    ):
        parts.append("Type: Podcast")
    elif state.lead.source_type == SourceType.YOUTUBE_SCRIPT_TOOL:
        parts.append("Type: YouTube Channel")

    # Website
    if lead.website:
        parts.append(f"Website: {lead.website}")

    # Existing email as a hint
    if lead.existing_email:
        parts.append(
            f"Existing email (unverified): {lead.existing_email} "
            f"— check if this is correct or find a better one"
        )

    # Social links — only include ones that exist
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


def _parse_perplexity_response(content: str) -> list[EmailCandidate]:
    """
    Parse Perplexity response into EmailCandidate list.
    Handles cases where response has markdown fences.
    """
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
    """
    Sort candidates by confidence descending.
    Penalise generic email prefixes.
    """
    generic_prefixes = {
        "info", "contact", "hello", "support",
        "admin", "team", "mail", "enquiries", "enquiry"
    }

    def score(candidate: EmailCandidate) -> float:
        prefix = candidate.email.split("@")[0].lower()
        penalty = 0.2 if prefix in generic_prefixes else 0.0
        return candidate.confidence - penalty

    return sorted(candidates, key=score, reverse=True)


def perplexity_discovery_node(state: EmailFinderState) -> EmailFinderState:
    """
    Uses Perplexity via LiteLLM to find email candidates
    from all available lead information.
    """
    trace = langfuse.trace(
        name="perplexity_discovery",
        metadata={
            "source_type": state.lead.source_type.value,
            "best_name": state.lead.identity.best_name(),
        },
    )

    try:
        # Build query
        available_info = _build_query(state)
        trace.event(
            name="query_built",
            metadata={"available_info": available_info},
        )

        # Fetch prompt
        prompt_text = get_prompt(
            prompt_name="perplexity_email_discovery",
            fallback=FALLBACK_PROMPT,
        )

        filled_prompt = prompt_text.format(available_info=available_info)

        # Call Perplexity via LiteLLM
        span = trace.span(name="perplexity_call")
        response = litellm.completion(
            model="perplexity/sonar",
            messages=[{"role": "user", "content": filled_prompt}],
            metadata={
                "langfuse_session_id": "perplexity_discovery",
                "langfuse_trace_name": "perplexity_email_discovery",
            },
        )
        span.end()

        # Parse response
        content = response.choices[0].message.content.strip()
        candidates = _parse_perplexity_response(content)
        ranked = _rank_candidates(candidates)

        trace.event(
            name="candidates_found",
            metadata={"count": len(ranked)},
        )

        # Update state
        updated_candidates = state.email_candidates + ranked
        best_email = ranked[0] if ranked else state.best_email
        status = (
            LeadStatus.EMAIL_FOUND if ranked else LeadStatus.EMAIL_NOT_FOUND
        )

        return state.model_copy(
            update={
                "status": status,
                "email_candidates": updated_candidates,
                "best_email": best_email,
                "nodes_executed": state.nodes_executed + ["perplexity_discovery"],
            }
        )

    except Exception as e:
        trace.event(
            name="perplexity_discovery_failed",
            metadata={"error": str(e)},
        )
        return state.model_copy(
            update={
                "status": LeadStatus.FAILED,
                "errors": state.errors + [f"perplexity_discovery: {str(e)}"],
                "nodes_executed": state.nodes_executed + ["perplexity_discovery"],
            }
        )