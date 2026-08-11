"""
Cheaply guess a person's website from a small LLM's own knowledge, BEFORE paying
for Perplexity's research mode.

Idea: for a lead with no website, ask a cheap model (gpt-4o-mini) whether it
already knows the person's official website from its training knowledge. If it
does, we set that as the lead's website and let the FREE Crawl4AI website-crawl
path find the email. If it doesn't know, it must abstain (return null) and we
fall through to Perplexity. A single mini call is ~150-300x cheaper than a
Perplexity research call, so even a modest hit-rate deflects real cost.

Safety: the model is instructed to return null when not confident (this keeps
hallucination near zero — verified empirically). The guessed site is still
*unverified*, so we only accept it above a confidence threshold and lean on the
downstream resolver's name/domain matching to reject a wrong-but-real domain.
"""
from __future__ import annotations

import json

import litellm

from ai_agents.agents.email_finder.io.contract_models import (
    WebsiteGuessInput,
    WebsiteGuessOutput,
)
from ai_agents.agents.email_finder.nodes.cost_utils import llm_call_cost
from ai_agents.agents.email_finder.nodes.canonical_builder import _detect_social_platform
from ai_agents.agents.email_finder.state import LeadStatus
from ai_agents.core.llm import langfuse

_MODEL = "gpt-4o-mini"

# Only accept a guess at/above this self-reported confidence. Empirically the
# model returns 1.0 for sites it actually knows and abstains (null) otherwise.
_MIN_CONFIDENCE = 0.7

# NB: keep the directory/social exclusion abstract — naming example domains
# (e.g. "speakerhub.com") made the model abstain ~1 in 5 even for sites it knew.
_PROMPT = """You are given a person's name and details about them. If you are CONFIDENT, \
from your own training knowledge, that you know their OWN official personal or company website, \
return it. If you are not confident, return null — do not guess or invent a domain. Do not \
return a social-media profile or a speaker/agency directory page as their website.

Name: {name}
{context}

Return ONLY JSON: {{"website": "https://... or null", "confidence": 0.0 to 1.0}}"""


# Row keys that are noise for identification (don't help the model place the person).
_CONTEXT_SKIP = {
    "img", "image", "asset value", "pricing", "country", "email", "email status",
    "channel id", "channel url", "search term id", "discovered at", "last upload",
    "subscribers", "avg views", "uploads last 30d", "score", "subsheet",
}


def _build_context(inp: WebsiteGuessInput, name: str) -> str:
    """
    Schema-agnostic: surface every descriptive field on the row (title, bio,
    location, occupation, etc.) so this works for any sheet, not a fixed allowlist.
    """
    lead = inp.lead
    parts: list[str] = []
    ctx_name = lead.identity.context_name()
    if ctx_name:
        parts.append(f"Show/Brand: {ctx_name}")

    for key, val in (lead.raw or {}).items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if key.strip().lower() in _CONTEXT_SKIP:
            continue
        text = str(val or "").strip()
        if not text or text == name or text.lower().startswith("http"):
            continue
        parts.append(f"{key}: {text[:240]}")

    # NB: deliberately do NOT mention `inp.source` here. Telling the model the
    # person is "listed on <a speaker directory>" makes it defer to the directory
    # and abstain from naming their personal site (verified empirically). The
    # source is still useful for Perplexity, which actually searches the web.
    return "\n".join(parts) if parts else "(no extra details)"


def _guess(name: str, context: str, trace_id: str) -> tuple[str | None, float, float]:
    """Returns (website, confidence, cost_usd)."""
    filled = _PROMPT.format(name=name, context=context)
    resp = litellm.completion(
        model=_MODEL,
        # 60s cap — without it the OpenAI SDK default (600s × retries) can hang
        # an executor thread ~20min and jam the whole page (the Joseph llinas RCA).
        timeout=60,
        messages=[{"role": "user", "content": filled}],
        response_format={"type": "json_object"},
        temperature=0,
        metadata={"langfuse_trace_id": trace_id, "langfuse_session_id": "email_finder"},
    )
    cost = llm_call_cost(resp)
    data = json.loads(resp.choices[0].message.content.strip())
    website = data.get("website")
    if not isinstance(website, str) or not website.lower().startswith("http"):
        return None, 0.0, cost
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return website.strip(), confidence, cost


def website_guesser_run(inp: WebsiteGuessInput) -> WebsiteGuessOutput:
    trace = langfuse.trace(id=inp.trace_id, name="email_finder", session_id="email_finder")
    span = trace.span(name="website_guesser")

    lead = inp.lead
    name = lead.identity.best_name()

    # Nothing to guess from, or already have a website — pass through untouched.
    if not name or (lead.website and str(lead.website).strip()):
        span.end()
        return WebsiteGuessOutput(lead=lead, status=LeadStatus.PENDING)

    try:
        website, confidence, cost = _guess(name, _build_context(inp, name), inp.trace_id)

        accepted = bool(
            website
            and confidence >= _MIN_CONFIDENCE
            and not _detect_social_platform(website)
        )
        if accepted:
            lead.website = website
            # Mark provenance so downstream/analytics know this site was a guess.
            lead.raw = {**(lead.raw or {}), "_website_source": "llm_guess"}

        span.end()
        trace.event(
            name="website_guess_complete",
            metadata={
                "name": name,
                "guessed": website,
                "confidence": confidence,
                "accepted": accepted,
            },
        )
        return WebsiteGuessOutput(lead=lead, status=LeadStatus.PENDING, cost_usd=cost)

    except Exception as e:
        span.end()
        trace.event(name="website_guess_failed", metadata={"error": str(e)})
        # Graceful: keep the lead as-is and let the graph continue to Perplexity.
        return WebsiteGuessOutput(
            lead=lead,
            status=LeadStatus.PENDING,
            errors=[f"website_guesser: {e}"],
        )


async def website_guesser_node_async(state: dict) -> dict:
    """LangGraph async node."""
    from ai_agents.agents.email_finder.adapters import website_guess_input_from_state

    inp = website_guess_input_from_state(state)
    out = website_guesser_run(inp)
    return {
        "lead": out.lead.model_dump(mode="json"),
        "status": out.status.value,
        "errors": out.errors,
        "nodes_executed": out.nodes_executed_delta,
        "cost_usd": out.cost_usd,
    }
