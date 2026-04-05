"""Smoke test: OpenAI GPT Image thumbnail (reference + base images → PNG bytes)."""

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies. Activate the project venv and install:\n"
        "  python3 -m venv .venv && source .venv/bin/activate\n"
        "  pip install -U pip && pip install -e .\n"
        "  python test_prompt.py"
    ) from exc

load_dotenv()

from ai_agents import run_thumbnail_agent

REFERENCE_URL = "https://thumbnail-generator-ai-agent.s3.ap-south-1.amazonaws.com/reference.jpeg"
BASE_URL = "https://thumbnail-generator-ai-agent.s3.ap-south-1.amazonaws.com/base.jpeg"

result = run_thumbnail_agent(
    model="gptimage",
    reference_image_url=REFERENCE_URL,
    base_image_urls=[BASE_URL],
    title="I Survived 30 Days",
    include_title=True,
    creative_comments="Strong focal point, energetic but mainstream YouTube style",
)

raw = result["image_bytes"]
if not isinstance(raw, (bytes, bytearray)):
    raise TypeError("Expected image_bytes to be bytes")
out_path = "output_thumbnail.png"
with open(out_path, "wb") as f:
    f.write(bytes(raw))

print(f"Saved to {out_path}")
print("Prompt length:", len(str(result["prompt_used"])))
