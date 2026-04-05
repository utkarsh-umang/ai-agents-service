"""Import-time side effects: validate required environment variables."""

from __future__ import annotations

import os

_REQUIRED = ("OPENAI_API_KEY", "GEMINI_API_KEY")

for _name in _REQUIRED:
    _raw = os.environ.get(_name)
    if _raw is None or not str(_raw).strip():
        raise EnvironmentError(
            f"Missing required environment variable: {_name}"
        )

del _name, _raw, _REQUIRED
