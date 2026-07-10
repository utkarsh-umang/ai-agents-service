"""Public API for the ai-agents-service package.

Only ``run_thumbnail_agent`` is supported as public API. Subpackages and
modules such as ``agents``, ``generator``, ``gpt_image_generator``, and
``prompts`` are internal implementation details and are not re-exported here.

Required environment variables (validated on import): ``OPENAI_API_KEY``.
Also used, but not enforced at import time (raised lazily on first call):
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``), ``BFL_API_KEY``. See ``.env.example``.

**Signature** (``model`` is the literal ``"gptimage"``, ``"nanobanana"``, or ``"fluxkontext"``)::

    def run_thumbnail_agent(
        model: Literal["gptimage", "nanobanana", "fluxkontext"],
        reference_image_url: str,
        base_image_urls: list[str],
        title: str,
        include_title: bool,
        creative_comments: str,
        shorts_or_reels: bool = False,
        num_candidates: int = 1,
    ) -> dict[str, object]:
        ...

**Return value:** a dict with ``images`` (``list[bytes]``, length ``num_candidates``)
and ``prompt_used`` (``str``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load env checks without registering ``ai_agents._bootstrap`` on this package.
_bootstrap_path = Path(__file__).with_name("_bootstrap.py")
_spec = importlib.util.spec_from_file_location(
    "_ai_agents_service_env_check",
    _bootstrap_path,
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load ai_agents env validation module")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
del _bootstrap_path, _spec, _mod

from .agents.thumbnail_generator.agent import run_thumbnail_agent

__all__ = ["run_thumbnail_agent"]
