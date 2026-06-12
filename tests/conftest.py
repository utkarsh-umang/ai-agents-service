"""Load .env before tests import ai_agents (its bootstrap validates env vars)."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
