#!/usr/bin/env bash
# Runs test_prompt.py using a project .venv — no Poetry (avoids broken pyenv shims).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

pick_python() {
  for candidate in \
    "/opt/homebrew/bin/python3.12" \
    "/opt/homebrew/bin/python3" \
    "/usr/local/bin/python3.12" \
    "/usr/local/bin/python3" \
    "/usr/bin/python3"; do
    if [[ -x "$candidate" ]]; then
      "$candidate" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null && echo "$candidate" && return
    fi
  done
  echo ""
}

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY_BOOT="$(pick_python)"
  if [[ -z "$PY_BOOT" ]]; then
    echo "Need Python 3.10+. Install with: brew install python@3.12"
    exit 1
  fi
  echo "Creating .venv with: $PY_BOOT"
  "$PY_BOOT" -m venv "$ROOT/.venv"
  PY="$ROOT/.venv/bin/python"
fi

"$ROOT/.venv/bin/pip" install -q -U pip
"$ROOT/.venv/bin/pip" install -q -e "$ROOT"
exec "$PY" "$ROOT/test_prompt.py"
