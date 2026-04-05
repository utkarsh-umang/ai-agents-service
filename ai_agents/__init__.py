"""Public API for the ai-agents-service package.

Only ``run_thumbnail_agent`` is supported as public API. Subpackages and
modules such as ``agents``, ``generator``, ``gpt_image_generator``, and
``prompts`` are internal implementation details and are not re-exported here.

Required environment variables (validated on import): ``OPENAI_API_KEY``,
``GEMINI_API_KEY``. See ``.env.example``.

**Signature** (``model`` is the literal ``"gptimage"`` or ``"nanobanana"``)::

    def run_thumbnail_agent(
        model: Literal["gptimage", "nanobanana"],
        reference_image_url: str,
        base_image_urls: list[str],
        title: str,
        include_title: bool,
        creative_comments: str,
    ) -> dict[str, object]:
        ...

**Return value:** a dict with ``image_bytes`` (``bytes``) and ``prompt_used`` (``str``).
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
