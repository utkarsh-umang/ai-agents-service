from __future__ import annotations
import os
import litellm
from langfuse import Langfuse
from ai_agents.core.config import (
    PERPLEXITY_API_KEY,
    LANGFUSE_EMAIL_FINDER_PUBLIC_KEY,
    LANGFUSE_EMAIL_FINDER_SECRET_KEY,
    LANGFUSE_HOST,
)

# Langfuse client for email finder
langfuse = Langfuse(
    public_key=LANGFUSE_EMAIL_FINDER_PUBLIC_KEY,
    secret_key=LANGFUSE_EMAIL_FINDER_SECRET_KEY,
    host=LANGFUSE_HOST,
)

# Tell LiteLLM to use Langfuse for tracing
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_EMAIL_FINDER_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_EMAIL_FINDER_SECRET_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST
os.environ["PERPLEXITY_API_KEY"] = PERPLEXITY_API_KEY


def get_prompt(prompt_name: str, fallback: str) -> str:
    """Fetch prompt from Langfuse, fall back to local if unavailable."""
    try:
        prompt = langfuse.get_prompt(prompt_name)
        return prompt.compile()
    except Exception:
        return fallback