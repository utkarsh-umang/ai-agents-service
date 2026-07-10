"""Import-time side effects: validate required environment variables.

Only ``OPENAI_API_KEY`` is validated eagerly here — it's the one thumbnail_generator
dependency without its own lazy getter (``client = OpenAI()`` resolves it implicitly).
Everything else is agent-specific and validated lazily where it's actually used:
``GEMINI_API_KEY``/``BFL_API_KEY`` via explicit getters in thumbnail_generator,
``PERPLEXITY_API_KEY``/``LANGFUSE_*`` via ``ai_agents.core.config`` when email_finder
is imported. Requiring those eagerly here would force every consumer of this package
(including ones that only use thumbnail_generator) to configure secrets for agents
they never call.
"""

from __future__ import annotations

import os

_REQUIRED = ("OPENAI_API_KEY",)

for _name in _REQUIRED:
    _raw = os.environ.get(_name)
    if _raw is None or not str(_raw).strip():
        raise EnvironmentError(
            f"Missing required environment variable: {_name}"
        )

del _name, _raw, _REQUIRED