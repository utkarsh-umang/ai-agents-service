from __future__ import annotations
import os

# Perplexity
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]

# ScrapingBee — OPTIONAL fallback for the youtube_about_enricher node. That node
# fetches the YouTube About page with a free HTTP GET first; ScrapingBee's
# residential proxies are only used if YouTube serves a consent/bot wall (e.g.
# from a datacenter IP). Leave blank to run the free path only.
SCRAPINGBEE_API_KEY = os.environ.get("SCRAPINGBEE_API_KEY", "")
# Override only if you proxy ScrapingBee or use a different base path.
SCRAPINGBEE_BASE_URL = os.environ.get(
    "SCRAPINGBEE_BASE_URL", "https://app.scrapingbee.com/api/v1/"
)

# Langfuse - Email Finder Project
LANGFUSE_EMAIL_FINDER_PUBLIC_KEY = os.environ["LANGFUSE_EMAIL_FINDER_PUBLIC_KEY"]
LANGFUSE_EMAIL_FINDER_SECRET_KEY = os.environ["LANGFUSE_EMAIL_FINDER_SECRET_KEY"]
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

# Mailin — email deliverability verification (mailin.ai). The "API key" is a
# bearer token minted once via POST /auth/login (per their docs) and stored in
# .env as MAILIN_API_TOKEN, so calls carry only the token, never the login
# password. Optional: leave MAILIN_API_TOKEN blank to disable verification (the
# tool then reports "unverified" instead of a valid/invalid/catch-all verdict).
MAILIN_API_TOKEN = os.environ.get("MAILIN_API_TOKEN", "")
MAILIN_BASE_URL = os.environ.get("MAILIN_BASE_URL", "https://api.mailin.ai/api/v1/public")