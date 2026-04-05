"""Public API for the ai-agents-service package.

Required environment variables (validated on import): ``OPENAI_API_KEY``,
``GEMINI_API_KEY``. See ``.env.example``.

Exports:
    run_thumbnail_agent(model, reference_image_url, base_image_urls, title,
        include_title, creative_comments) -> dict[str, object]
        Returns a dict with ``image_bytes`` (bytes) and ``prompt_used`` (str).
"""

from __future__ import annotations

from . import _bootstrap  # noqa: F401 - env validation side effect
from .agents.thumbnail_generator.agent import run_thumbnail_agent

__all__ = ["run_thumbnail_agent"]
