"""
Seed (create or update) all Langfuse prompts required by the email-finder agent.

Run once after any prompt change:
    cd /path/to/ai-agents-service
    poetry run python scripts/seed_langfuse_prompts.py

Each prompt is created with label "production" so it is picked up by the agent.
Re-running the script creates a new version; the previous version is retained in
Langfuse history for rollback.

Langfuse template variables use the {{variable}} syntax.
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_EMAIL_FINDER_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_EMAIL_FINDER_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
)

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS: list[dict] = [
    {
        "name": "canonical_builder",
        "prompt": """\
You are a data normalization assistant.

You will receive a raw row from a CSV file and a source type.
Your job is to extract and classify the fields into a structured format.

Source type: {{source_type}}
Raw row: {{raw_row}}

Extract the following and return as JSON only, no explanation:
{
    "host_name": "string or null",
    "podcast_name": "string or null",
    "channel_name": "string or null",
    "brand_name": "string or null",
    "website": "string or null — only a true standalone domain owned by the host/show (e.g. https://example.com)",
    "discovery_urls": ["array of other non-social URLs like linktr.ee, beacons.ai, carrd.co, etc. — or empty array []"],
    "existing_email": "string or null",
    "social_links": {
        "facebook": "string or null",
        "twitter": "string or null",
        "instagram": "string or null",
        "youtube": "string or null",
        "linkedin": "string or null"
    }
}

Rules:
- Social platform URLs (facebook, twitter/x, instagram, youtube, linkedin) → social_links only, never website or discovery_urls
- website is only a true standalone domain that the host/show owns (e.g. https://jenniferrichardson.com)
- Link aggregators and profile pages (linktr.ee, beacons.ai, carrd.co, linkin.bio, bio.site, campsite.bio) → discovery_urls
- discovery_urls is always an array; use [] if none found
- Do not invent or guess values; set null or [] when not present\
""",
        "labels": ["production"],
    },
    {
        "name": "perplexity_email_discovery",
        "prompt": """\
Find the contact email address for the following podcast or YouTube channel. Search their website, Linktree, social media bios, and any other public sources.

{{available_info}}

Search thoroughly — check their personal site, any link-in-bio pages, podcast directory listings (Apple Podcasts, Spotify, Listen Notes), and social media profiles.

Return ONLY a JSON object, no other text:
{
    "emails_found": [
        {
            "email": "string",
            "source": "string — where exactly you found it",
            "confidence": 0.0 to 1.0,
            "note": "string or null"
        }
    ],
    "search_summary": "brief summary of what you searched and what you found"
}

Confidence guide:
- 1.0 — on their personal website or LinkedIn profile
- 0.8 — on podcast/channel contact page or RSS feed
- 0.6 — in social media bio or Linktree
- 0.4 — in a third-party listing or article
- 0.2 — inferred or uncertain\
""",
        "labels": ["production"],
    },
    {
        "name": "email_resolver",
        "prompt": """\
You are selecting the single best contact email for outreach to a podcast host or YouTube creator.

Canonical lead (JSON):
{{canonical_lead_json}}

Candidate emails found on their website (JSON array, may be empty):
{{candidates_json}}

Rules:
- Prefer a personal or show-specific address over generic inboxes (info@, contact@, support@) when both exist and the personal one clearly belongs to the same person/show.
- Prefer addresses on the same domain as the lead website when applicable.
- If no candidate is suitable, return chosen_email as null.
- Do not invent emails that are not in the candidates list.

Return ONLY valid JSON, no markdown:
{
    "chosen_email": "string or null",
    "confidence": 0.0,
    "reason": "short explanation"
}\
""",
        "labels": ["production"],
    },
]


# ── Seed ─────────────────────────────────────────────────────────────────────

def seed() -> None:
    for spec in PROMPTS:
        name = spec["name"]
        try:
            langfuse.create_prompt(
                name=name,
                prompt=spec["prompt"],
                labels=spec["labels"],
            )
            print(f"✓ Created/updated prompt '{name}' with labels {spec['labels']}")
        except Exception as exc:
            print(f"✗ Failed to upsert prompt '{name}': {exc}", file=sys.stderr)
            sys.exit(1)

    langfuse.flush()
    print("\nDone. All prompts seeded successfully.")


if __name__ == "__main__":
    seed()
