"""Per-call USD cost helpers for the graph's `cost_usd` accumulator.

LLM calls get exact costs from litellm's pricing table. The two non-LLM paid
providers can't self-report USD, so they use per-call estimates overridable by
env (set them to match your actual plan's effective rate):

  SCRAPINGBEE_COST_USD_PER_CALL  default 0.001  (render_js=false = 1 credit;
                                  ~$0.0008-0.001/credit on the Freelance plan)
  PERPLEXITY_COST_USD_PER_CALL   default 0.005  (Agent API pro-search base
                                  request fee; token cost is small next to it)
"""
from __future__ import annotations

import os

import litellm

SCRAPINGBEE_COST_USD = float(os.environ.get("SCRAPINGBEE_COST_USD_PER_CALL", "0.001"))
PERPLEXITY_COST_USD = float(os.environ.get("PERPLEXITY_COST_USD_PER_CALL", "0.005"))


def llm_call_cost(response) -> float:
    """Exact USD cost of one litellm completion; 0.0 if pricing is unknown.

    Never raises — cost accounting must not be able to fail a lead.
    """
    try:
        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0
