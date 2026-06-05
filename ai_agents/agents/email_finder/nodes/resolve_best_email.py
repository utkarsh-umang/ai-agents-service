"""LLM resolution of best email from website-sourced candidates."""
from __future__ import annotations

import json
import re

import litellm

from ai_agents.agents.email_finder.adapters import lead_to_dict, merge_candidate_dicts
from ai_agents.agents.email_finder.io.contract_models import ResolveInput, ResolveOutput
from ai_agents.agents.email_finder.state import EmailCandidate, LeadStatus
from ai_agents.core.llm import get_prompt, langfuse


def _parse_json_response(raw: str) -> dict:
    content = re.sub(
        r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE
    ).strip()
    return json.loads(content)


def _dedupe_candidates(items: list[EmailCandidate]) -> list[EmailCandidate]:
    seen: set[str] = set()
    out: list[EmailCandidate] = []
    for c in items:
        k = c.email.lower().strip()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


async def resolve_best_email_async(inp: ResolveInput) -> ResolveOutput:
    trace = langfuse.trace(
        id=inp.trace_id,
        name="email_finder",
        session_id="email_finder",
    )
    span = trace.span(name="resolve_best_email")

    ranked = _dedupe_candidates(inp.website_candidates)
    if not ranked:
        span.end()
        trace.event(name="resolver_empty", metadata={})
        return ResolveOutput(
            best_email=None,
            status=LeadStatus.EMAIL_NOT_FOUND,
            errors=[],
        )

    filled = get_prompt(
        "email_resolver",
        canonical_lead_json=json.dumps(lead_to_dict(inp.lead), ensure_ascii=False),
        candidates_json=json.dumps(
            [c.model_dump(mode="json") for c in ranked],
            ensure_ascii=False,
        ),
    )

    try:
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": filled}],
            response_format={"type": "json_object"},
            metadata={
                "langfuse_trace_id": inp.trace_id,
                "langfuse_session_id": "email_finder",
            },
        )
        raw = response.choices[0].message.content or "{}"
        data = _parse_json_response(raw)
    except Exception as e:
        span.end()
        return ResolveOutput(
            best_email=None,
            status=LeadStatus.EMAIL_NOT_FOUND,
            errors=[f"resolve_best_email: {e}"],
        )

    chosen = (data.get("chosen_email") or "").strip() or None
    if not chosen:
        reason = data.get("reason") or "Resolver rejected all candidates"
        span.end()
        trace.event(name="resolver_empty", metadata={"reason": reason})
        return ResolveOutput(
            best_email=None,
            status=LeadStatus.EMAIL_NOT_FOUND,
            errors=[f"resolver: {reason}"],
        )

    conf = float(data.get("confidence", 0.7))
    note = data.get("reason")

    best = EmailCandidate(
        email=chosen,
        source="resolver — website crawl",
        confidence=max(0.0, min(1.0, conf)),
        note=note,
    )
    span.end()
    return ResolveOutput(
        best_email=best,
        status=LeadStatus.EMAIL_FOUND,
        errors=[],
    )


async def resolve_best_email_node_async(state: dict) -> dict:
    from ai_agents.agents.email_finder.adapters import resolve_input_from_state

    inp = resolve_input_from_state(state)
    out = await resolve_best_email_async(inp)

    merged = merge_candidate_dicts(
        list(state.get("email_candidates") or []),
        list(state.get("website_scrape_candidates") or []),
    )
    if out.best_email:
        merged = merge_candidate_dicts(
            merged,
            [out.best_email.model_dump(mode="json")],
        )

    return {
        "status": out.status.value,
        "best_email": out.best_email.model_dump(mode="json") if out.best_email else None,
        "email_candidates": merged,
        "errors": out.errors,
        "nodes_executed": out.nodes_executed_delta,
    }
