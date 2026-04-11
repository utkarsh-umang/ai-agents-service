from __future__ import annotations
import os

# Perplexity
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]

# Langfuse - Email Finder Project
LANGFUSE_EMAIL_FINDER_PUBLIC_KEY = os.environ["LANGFUSE_EMAIL_FINDER_PUBLIC_KEY"]
LANGFUSE_EMAIL_FINDER_SECRET_KEY = os.environ["LANGFUSE_EMAIL_FINDER_SECRET_KEY"]
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com")