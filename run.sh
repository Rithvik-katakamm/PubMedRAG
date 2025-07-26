#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/env/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "Error: Python executable not found at $PYTHON" >&2
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/src/app.py" "$@"
