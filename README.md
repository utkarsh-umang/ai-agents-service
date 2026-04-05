# ai-agents-service

## Do not use `poetry run` here if you see pyenv **exit 127**

Poetry is calling a broken `~/.pyenv/shims/python`. Use the venv below instead.

## Quickest way to run the thumbnail test

From the repo root:

```bash
chmod +x run_test.sh
./run_test.sh
```

That creates `.venv` with **Homebrew or `/usr/local` Python 3.10+** (not pyenv), installs deps with `pip install -e .`, and runs `test_prompt.py`.

`test_prompt.py` uses **`run_thumbnail_agent(model="gptimage", ...)`**: **OpenAI GPT Image** (`images.edit` with `gpt-image-1` or similar) takes the **reference + base images** plus a text brief and returns **PNG bytes** (base64 in the API response—there is no permanent URL). You need **`OPENAI_API_KEY`** and org access to GPT Image.

To try **Gemini (Nano Banana)** instead, call `run_thumbnail_agent(model="nanobanana", ...)` (requires **`GEMINI_API_KEY`**).

## Manual setup (same as the script)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python test_prompt.py
```

Use a real **3.10+** interpreter to create the venv (e.g. `/opt/homebrew/bin/python3.12`), not a broken pyenv shim.

## Environment

Create `.env` (not committed) with:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

## Notes

- The old **GPT-4o analyst + DALL-E 3** text-only path was removed in favor of true **image-conditioned** OpenAI **GPT Image** (same architectural idea as Nano Banana: images + prompt → image).
- **`analyst.py`** is unused by the default pipeline; kept for reference or experiments.
- Gemini image models may return **429** if quota or billing is insufficient—see [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).
- OpenAI image endpoints have their own quotas; see [OpenAI limits](https://platform.openai.com/settings/organization/limits).

Poetry is optional; only use it after fixing pyenv or putting a working `python` ahead of `~/.pyenv/shims` in `PATH`.

If `./run_test.sh` fails with import or version errors, remove the old env and retry: `rm -rf .venv && ./run_test.sh`.
