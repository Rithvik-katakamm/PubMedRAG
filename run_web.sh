#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Check if virtual environment exists
if [ ! -x "$PYTHON" ]; then
  echo "Error: Python executable not found at $PYTHON" >&2
  echo "Please create a virtual environment and install dependencies:" >&2
  echo "  python -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -r requirements.txt" >&2
  exit 1
fi

echo "Starting PubMed RAG Web Application..."
echo "Open your browser to: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src"
exec "$PYTHON" -m uvicorn src.web_app:app --host 0.0.0.0 --port 8000 --reload 


# http://localhost:8000