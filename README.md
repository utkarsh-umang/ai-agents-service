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

Gemini image models may return **429** if quota or billing is insufficient—see [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

Poetry is optional; only use it after fixing pyenv or putting a working `python` ahead of `~/.pyenv/shims` in `PATH`.

If `./run_test.sh` fails with import or version errors, remove the old env and retry: `rm -rf .venv && ./run_test.sh`.
