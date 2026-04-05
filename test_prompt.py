"""Smoke test: Gemini (Nano Banana) thumbnail generation from public image URLs."""

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies. Activate the project venv and install:\n"
        "  python3 -m venv .venv && source .venv/bin/activate\n"
        "  pip install -U pip && pip install -e .\n"
        "  python test_prompt.py"
    ) from exc

from ai_agents.agents.thumbnail_generator.generator import generate_with_nanobanana

load_dotenv()

# Public HTTPS images (replace with S3 presigned URLs for private objects).
REFERENCE_URL = "https://thumbnail-generator-ai-agent.s3.ap-south-1.amazonaws.com/reference.jpeg"
BASE_URL = "https://thumbnail-generator-ai-agent.s3.ap-south-1.amazonaws.com/base.jpeg"

image_bytes = generate_with_nanobanana(
    reference_image_url=REFERENCE_URL,
    base_image_urls=[BASE_URL],
    title="I Survived 30 Days",
    include_title=True,
    creative_comments="Make it feel intense and dramatic",
)

with open("output_thumbnail.png", "wb") as f:
    f.write(image_bytes)

print("Saved to output_thumbnail.png")
