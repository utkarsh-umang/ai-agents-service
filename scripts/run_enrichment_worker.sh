#!/usr/bin/env bash
# launchd entrypoint for the LMS enrichment worker: sources the project
# .env (launchd doesn't do that) and execs the venv python.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a; source .env; set +a
# Unbuffered so launchd's log file fills in real time, not on exit.
export PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" scripts/lms_enrichment_worker.py "$@"
